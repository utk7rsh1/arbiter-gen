"""FastAPI REST Server for ARBITER — OpenEnv-compliant.

Endpoints:
  POST /sessions                    → create session, returns session_id
  POST /sessions/{id}/reset         → reset episode
  POST /sessions/{id}/step          → take action, returns (obs, reward, done, info)
  GET  /sessions/{id}/render        → graph + claim chain visualization data
  GET  /sessions/{id}/metrics       → session aggregate metrics
  GET  /metrics                     → global aggregate metrics
  GET  /explain/{session_id}        → post-hoc ground truth explanation
  GET  /health                      → health check

Usage:
    pip install fastapi uvicorn
    python -m arbiter.server
    # Then open http://localhost:8000/docs for auto-generated Swagger UI
"""
from __future__ import annotations

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import time

from arbiter.env.environment import (
    ArbiterEnv, create_session, get_session, list_sessions
)
try:
    from arbiter.env.domain_config import DomainConfig
    _HAS_DOMAIN_CONFIG = True
except ImportError:
    _HAS_DOMAIN_CONFIG = False
from arbiter.env.dual_env import (
    DualArbiterEnv, create_dual_session, get_dual_session, list_dual_sessions
)
from arbiter.env.domain_config import DomainConfig
try:
    from arbiter.env.openenv_wrapper import ArbiterEnvironment, ArbiterAction
    _HAS_OPENENV_WRAPPER = True
except ImportError:
    _HAS_OPENENV_WRAPPER = False

# ── OpenEnv-native app (for TRL / Unsloth GRPO integration) ─────────────
openenv_app = None
if _HAS_OPENENV_WRAPPER:
    try:
        from openenv.core import create_app as _openenv_create_app
        _arbiter_openenv_env = ArbiterEnvironment(level=1)
        openenv_app = _openenv_create_app(_arbiter_openenv_env, ArbiterAction)
    except Exception as _e:
        print(f"[server] OpenEnv native app skipped: {_e}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ARBITER — AI Oversight Training Environment",
    description=(
        "OpenEnv-compatible REST API for the ARBITER multi-agent RL environment. "
        "Implements co-evolutionary auditing: Auditor vs Adaptive Defender over 7 curriculum levels."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static frontend ──────────────────────────────────────────────────────────
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "demo", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        """Serve the React frontend index.html."""
        return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))

_global_start = time.time()
_global_stats: Dict[str, Any] = {
    "total_sessions":  0,
    "total_episodes":  0,
    "total_rewards":   [],
    "verdicts_correct": 0,
}


# ── Schemas ────────────────────────────────────────────────────────────────────

def _parse_domain(domain_json: Optional[Dict]) -> Optional["DomainConfig"]:
    """Deserialise a plain dict into a DomainConfig, or return None."""
    if not domain_json:
        return None
    if not _HAS_DOMAIN_CONFIG:
        raise HTTPException(status_code=501, detail="DomainConfig module not available")
    try:
        from arbiter.env.domain_config import DomainConfig
        return DomainConfig(**domain_json)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid domain_json: {exc}")


class GenerateDomainRequest(BaseModel):
    description: str
    seed: int = 42


class CreateSessionRequest(BaseModel):
    level: int = 1
    seed: Optional[int] = None
    domain_json: Optional[Dict[str, Any]] = None


class ResetRequest(BaseModel):
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


class SessionResponse(BaseModel):
    session_id: str
    level: int
    created: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - _global_start, 1)}


@app.post("/generate-domain")
def generate_domain_endpoint(req: GenerateDomainRequest):
    """
    Generate a DomainConfig from a plain-English description via Groq.
    Returns the full DomainConfig as JSON so the frontend can display
    and edit it before starting a session.
    """
    import os
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY not set on the server. Set the environment variable and restart."
        )
    try:
        from arbiter.env.groq_generator import GroqGraphGenerator
        gen    = GroqGraphGenerator(api_key=groq_key)
        config = gen.generate_cached(req.description, seed=req.seed)
        return config.model_dump()
    except ImportError:
        raise HTTPException(status_code=501, detail="groq package not installed (pip install groq)")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Domain generation failed: {exc}")


@app.post("/sessions", response_model=SessionResponse)
def create_session_endpoint(req: CreateSessionRequest):
    domain = _parse_domain(req.domain_json)
    sid = create_session(level=req.level, seed=req.seed, domain=domain)
    _global_stats["total_sessions"] += 1
    return {"session_id": sid, "level": req.level, "created": time.time()}


@app.get("/sessions")
def list_sessions_endpoint():
    return {"sessions": list_sessions(), "count": len(list_sessions())}


@app.post("/sessions/{session_id}/reset")
def reset_endpoint(session_id: str, req: ResetRequest):
    env = _get_env(session_id)
    obs = env.reset(seed=req.seed)
    _global_stats["total_episodes"] += 1
    return {"observation": obs}


@app.post("/sessions/{session_id}/step")
def step_endpoint(session_id: str, req: StepRequest):
    env = _get_env(session_id)
    obs, reward, done, info = env.step(req.action)

    if done and "episode_reward" in info:
        r = info["episode_reward"]["total"]
        _global_stats["total_rewards"].append(r)
        if info["episode_reward"]["terminal"].get("verdict_correct"):
            _global_stats["verdicts_correct"] += 1

    return {
        "observation": obs,
        "reward":      reward,
        "done":        done,
        "info":        _serialize(info),
    }


@app.get("/sessions/{session_id}/render")
def render_endpoint(session_id: str):
    env = _get_env(session_id)
    return env.render()


@app.get("/sessions/{session_id}/metrics")
def session_metrics_endpoint(session_id: str):
    env = _get_env(session_id)
    return env.get_metrics()


@app.get("/explain/{session_id}")
def explain_endpoint(session_id: str):
    """Post-hoc ground truth explanation for the current episode."""
    env = _get_env(session_id)
    if env._anomaly_info is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return {
        "anomaly_info":    env._anomaly_info,
        "decoy_a":         env._ep.get("decoy_a", {}),
        "decoy_b":         env._ep.get("decoy_b", {}),
        "defender_log":    env.defender.action_log,
        "curriculum":      env.curriculum.get_stats(),
    }


@app.get("/metrics")
def global_metrics():
    """Global aggregate metrics across all sessions."""
    rewards = _global_stats["total_rewards"]
    return {
        "total_sessions":   _global_stats["total_sessions"],
        "total_episodes":   _global_stats["total_episodes"],
        "mean_reward":      round(sum(rewards) / max(1, len(rewards)), 2),
        "verdict_accuracy": round(_global_stats["verdicts_correct"] / max(1, len(rewards)), 3),
        "uptime_s":         round(time.time() - _global_start, 1),
    }


@app.get("/arms-race")
def arms_race_endpoint():
    """
    Return training episode data from grpo_training.jsonl as chart-ready points.
    Each point: { ep, auditor (mean_reward), defender (defender_evasion * 100) }
    """
    import pathlib
    log_path = pathlib.Path(__file__).parent.parent / "logs" / "grpo_training.jsonl"
    data = []
    try:
        with open(log_path) as f:
            for line in f:
                try:
                    e = __import__("json").loads(line)
                    data.append({
                        "ep":       e.get("episode", len(data)),
                        "auditor":  round(float(e.get("mean_reward", 0)), 2),
                        "defender": round(float(e.get("defender_evasion", 0)) * 100, 1),
                    })
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return {"data": data, "count": len(data)}


@app.get("/training-log")
def training_log_endpoint(last: int = 100):
    """Return the last N lines of the GRPO training stdout log."""
    import pathlib
    log_path = pathlib.Path(__file__).parent.parent / "logs" / "grpo_stdout.log"
    lines = []
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        pass
    return {"lines": [l.rstrip() for l in lines[-last:]], "total": len(lines)}


# ── Training subprocess management ────────────────────────────────────────────

_training_proc: Optional[Any] = None
_training_log_buffer: list = []


class StartTrainingRequest(BaseModel):
    checkpoint: str = "lora_sft_v4"
    level:      int = 3
    episodes:   int = 120
    output:     str = "lora_grpo_v2"
    kl_coef:    float = 0.1
    lr:         float = 5e-6


@app.post("/training/start")
def training_start_endpoint(req: StartTrainingRequest):
    """Launch the GRPO trainer as a subprocess (mirrors Gradio training tab)."""
    import pathlib, subprocess, threading, sys as _sys
    global _training_proc, _training_log_buffer
    if _training_proc and _training_proc.poll() is None:
        return {"status": "already_running", "pid": _training_proc.pid}

    root     = pathlib.Path(__file__).parent.parent
    log_path = root / "logs" / "grpo_training.jsonl"
    out_path = root / "logs" / "grpo_stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)

    cmd = [
        _sys.executable, "-m", "arbiter.training.grpo_trainer",
        "--checkpoint", str(root / req.checkpoint),
        "--level",      str(req.level),
        "--episodes",   str(req.episodes),
        "--output",     str(root / req.output),
        "--log_file",   str(log_path),
        "--kl_coef",    str(req.kl_coef),
        "--lr",         str(req.lr),
    ]

    out_file = open(out_path, "w")
    _training_proc = subprocess.Popen(
        cmd, cwd=str(root),
        stdout=out_file, stderr=subprocess.STDOUT
    )
    _training_log_buffer = []

    def _watch():
        _training_proc.wait()
        out_file.close()

    threading.Thread(target=_watch, daemon=True).start()
    return {"status": "started", "pid": _training_proc.pid, "cmd": " ".join(cmd)}


@app.post("/training/abort")
def training_abort_endpoint():
    """Terminate the running GRPO subprocess."""
    global _training_proc
    if _training_proc and _training_proc.poll() is None:
        _training_proc.terminate()
        return {"status": "aborted", "pid": _training_proc.pid}
    return {"status": "not_running"}


@app.get("/training/status")
def training_status_endpoint():
    """Return whether training is running + last entry from the JSONL log."""
    import pathlib, json as _json
    global _training_proc
    running = bool(_training_proc and _training_proc.poll() is None)
    pid     = _training_proc.pid if _training_proc else None
    rc      = _training_proc.poll() if _training_proc else None

    log_path = pathlib.Path(__file__).parent.parent / "logs" / "grpo_training.jsonl"
    last_entry = None
    try:
        lines = open(log_path).readlines()
        if lines:
            last_entry = _json.loads(lines[-1])
    except Exception:
        pass

    return {
        "running":    running,
        "pid":        pid,
        "returncode": rc,
        "last_entry": last_entry,
        "total_episodes": last_entry.get("episode", 0) if last_entry else 0,
    }




@app.get("/leaderboard")
def leaderboard():
    """Top-10 sessions by mean episode reward."""
    from arbiter.env.environment import _sessions
    rows = []
    for sid, env in _sessions.items():
        m = env.get_metrics()
        rows.append({
            "session_id":   sid,
            "level":        m["current_level"],
            "mean_reward":  m["mean_reward"],
            "accuracy":     m["accuracy"],
            "episodes":     m["episodes_completed"],
        })
    rows.sort(key=lambda r: r["mean_reward"], reverse=True)
    return {"leaderboard": rows[:10]}


# ── Level 7 — Dual-Auditor Endpoints ─────────────────────────────────────────

class CreateDualSessionRequest(BaseModel):
    level:       int = 7
    mode:        str = "collaborative"   # "collaborative" or "competitive"
    seed:        Optional[int] = None
    domain_json: Optional[Dict[str, Any]] = None


class DualStepRequest(BaseModel):
    auditor_id: str          # "A" or "B"
    action:     Dict[str, Any]


@app.post("/dual-sessions")
def create_dual_session_endpoint(req: CreateDualSessionRequest):
    """Create a Level-7 dual-auditor session. Returns session_id."""
    domain = _parse_domain(req.domain_json)
    sid = create_dual_session(level=req.level, mode=req.mode, seed=req.seed, domain=domain)
    _global_stats["total_sessions"] += 1
    return {"session_id": sid, "level": req.level, "mode": req.mode, "created": time.time()}


@app.get("/dual-sessions")
def list_dual_sessions_endpoint():
    return {"sessions": list_dual_sessions(), "count": len(list_dual_sessions())}


@app.post("/dual-sessions/{session_id}/reset")
def dual_reset_endpoint(session_id: str, req: ResetRequest):
    dual = _get_dual(session_id)
    obs_a, obs_b = dual.reset(seed=req.seed)
    _global_stats["total_episodes"] += 1
    return {"observation_A": obs_a, "observation_B": obs_b}


@app.post("/dual-sessions/{session_id}/step")
def dual_step_endpoint(session_id: str, req: DualStepRequest):
    """Step for one auditor (A or B). Other auditor steps independently."""
    dual = _get_dual(session_id)
    obs, reward, done, info = dual.step(req.auditor_id, req.action)
    return {
        "observation": obs,
        "reward":      reward,
        "done":        done,
        "info":        _serialize(info),
    }


@app.get("/dual-sessions/{session_id}/render")
def dual_render_endpoint(session_id: str, auditor_id: str = "A"):
    dual = _get_dual(session_id)
    return dual.render(auditor_id=auditor_id)


@app.get("/dual-sessions/{session_id}/metrics")
def dual_metrics_endpoint(session_id: str):
    dual = _get_dual(session_id)
    return dual.get_metrics()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_dual(session_id: str) -> DualArbiterEnv:
    dual = get_dual_session(session_id)
    if dual is None:
        raise HTTPException(status_code=404,
                            detail=f"Dual session '{session_id}' not found.")
    return dual


def _get_env(session_id: str) -> ArbiterEnv:
    env = get_session(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env


def _serialize(obj):
    """Make objects JSON-safe."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(i) for i in obj]
    if isinstance(obj, bool):
        return obj
    if hasattr(obj, "item"):   # numpy scalars
        return obj.item()
    return obj


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Starting ARBITER API server...")
    print("REST API docs:   http://localhost:8000/docs")
    if openenv_app:
        print("OpenEnv app:     http://localhost:8000/openenv")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


# Mount OpenEnv sub-app if available (enables /openenv/* WebSocket endpoints)
if openenv_app:
    app.mount("/openenv", openenv_app)


if __name__ == "__main__":
    main()
