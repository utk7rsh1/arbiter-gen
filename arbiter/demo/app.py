"""ARBITER Gradio Demo — Live episode + GRPO Training Monitor."""
import sys
import json
import copy
import subprocess
import threading
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from arbiter.env.environment import ArbiterEnv

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent.parent
GRPO_LOG = ROOT / "logs" / "grpo_training.jsonl"

# ── Scripted action sequence (mirrors useEpisode.js _getScriptedAction) ────────
SCRIPTED_ACTIONS = [
    {"type": "QUERY_RECORDS", "feature_filter": {}, "outcome_filter": "denied", "time_range": None},
    {"type": "QUERY_FEATURE_DISTRIBUTION", "feature_id": "zip_code_cluster", "group_by": "loan_outcome"},
    {"type": "CLAIM_CAUSAL", "claim": {
        "cause_feature": "zip_code_cluster", "effect_outcome": "denial_rate_overall",
        "mechanism": "internal_risk_score", "direction": "positive", "confidence": "HIGH",
        "basis_records": ["rec_001", "rec_007", "rec_012"], "anomaly_type": "proxy_discrimination",
    }},
    {"type": "QUERY_COUNTERFACTUAL", "record_id": "rec_001", "feature_id": "zip_code_cluster", "counterfactual_value": 3},
    {"type": "CLAIM_COUNTERFACTUAL", "claim": {
        "subject_record": "rec_001", "counterfactual_feature": "zip_code_cluster",
        "predicted_outcome_change": "approved", "confidence": "HIGH", "basis": "cf_query_step_3",
    }},
    {"type": "QUERY_FEATURE_DISTRIBUTION", "feature_id": "credit_score", "group_by": "zip_code_cluster"},
    {"type": "CLAIM_CAUSAL", "claim": {
        "cause_feature": "zip_code_cluster", "effect_outcome": "internal_risk_score",
        "mechanism": "proxy_laundering", "direction": "positive", "confidence": "HIGH",
        "basis_records": ["rec_001", "rec_007", "rec_012", "rec_019"], "anomaly_type": "proxy_discrimination",
    }},
    {"type": "FLAG_HYPOTHESIS", "hypothesis_type": "proxy_discrimination", "status": "CONFIRMED"},
    {"type": "FLAG_HYPOTHESIS", "hypothesis_type": "model_drift", "status": "ELIMINATED"},
    {"type": "CLAIM_THEORY_OF_MIND", "claim": {
        "defender_action": "obfuscating", "target_link": "zip_code_cluster->internal_risk_score",
        "obfuscation_method": "timestamp_manipulation", "confidence": "HIGH",
        "basis": "timestamp_clustering_around_denial_events",
    }},
    {"type": "QUERY_COUNTERFACTUAL", "record_id": "rec_007", "feature_id": "zip_code_cluster", "counterfactual_value": 1},
    {"type": "SUBMIT_REPORT",
        "anomaly_type": "proxy_discrimination",
        "primary_evidence_chain": ["zip_code_cluster", "internal_risk_score", "denial_rate_overall"],
        "affected_demographic": "zip_code_cluster_7",
        "recommended_action": "audit_risk_score_model",
    },
]

# ── Episode global state ────────────────────────────────────────────────────────
_env            = None
_obs            = None
_render         = None
_step_index     = 0
_claims         = []
_hypotheses     = {
    "proxy_discrimination":  "ACTIVE",
    "adversarial_injection": "ACTIVE",
    "model_drift":           "ACTIVE",
}
_budget              = 20
_reward_total        = 0.0
_reward_components   = {"claim": 0.0, "counterfactual": 0.0, "tom": 0.0,
                         "chain": 0.0, "consistency": 0.0, "budget": 0.0}
_episode_done        = False
_verdict_correct     = None
_current_level       = 3

# ── Training global state ───────────────────────────────────────────────────────
_training_process = None
_log_buffer       = []

# ── CSS ─────────────────────────────────────────────────────────────────────────
CSS = """
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container, .main, footer { background: #020617 !important; color: #f1f5f9 !important; }
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }

.arbiter-header {
    background: linear-gradient(135deg, #0f172a 0%, #0c1a3a 50%, #0f172a 100%);
    border: 1px solid #1e3a6e; border-radius: 14px;
    padding: 24px 32px; margin-bottom: 16px;
    position: relative; overflow: hidden;
}
.arbiter-header::before {
    content: ''; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(59,130,246,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(139,92,246,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.arbiter-title {
    font-size: 2rem; font-weight: 900; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 4px 0;
}
.arbiter-subtitle { color: #64748b; font-size: 0.8rem; letter-spacing: 0.5px; text-transform: uppercase; margin: 0; }
.arbiter-badge {
    display: inline-block; background: #0f2a4a; border: 1px solid #1e4a8a;
    color: #60a5fa; font-size: 10px; font-weight: 700; letter-spacing: 1px;
    padding: 3px 8px; border-radius: 4px; text-transform: uppercase;
}

.tab-nav { background: #0f172a !important; border-bottom: 1px solid #1e293b !important; }
.tab-nav button { color: #64748b !important; font-weight: 500 !important; }
.tab-nav button.selected { color: #60a5fa !important; border-bottom: 2px solid #3b82f6 !important; }

.block { background: #0f172a !important; border: 1px solid #1e293b !important; border-radius: 10px !important; }
.label-wrap span { color: #94a3b8 !important; font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: 0.5px !important; text-transform: uppercase !important; }

textarea, input[type=text], input[type=number] {
    background: #020617 !important; color: #e2e8f0 !important;
    border: 1px solid #334155 !important; border-radius: 8px !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important; font-size: 12px !important;
}
textarea:focus, input:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important; }

button.primary { background: linear-gradient(135deg, #1d4ed8, #2563eb) !important; border: none !important;
    color: white !important; font-weight: 600 !important; border-radius: 8px !important; transition: all 0.2s !important; }
button.primary:hover { background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    box-shadow: 0 0 16px rgba(59,130,246,0.3) !important; transform: translateY(-1px) !important; }
button.secondary { background: #1e293b !important; border: 1px solid #334155 !important;
    color: #94a3b8 !important; border-radius: 8px !important; }
button.stop { background: linear-gradient(135deg, #7f1d1d, #991b1b) !important; border: none !important;
    color: white !important; font-weight: 600 !important; border-radius: 8px !important; }

select, .wrap { background: #0f172a !important; color: #e2e8f0 !important; border-color: #334155 !important; }
input[type=range] { accent-color: #3b82f6; }

.stats-bar {
    display: flex; flex-wrap: wrap; gap: 8px;
    background: #0a1628; border: 1px solid #1e3a6e;
    border-radius: 10px; padding: 10px 14px; margin: 8px 0;
}
.stat-pill {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 8px; padding: 6px 14px;
    display: flex; flex-direction: column; align-items: center; min-width: 80px;
}
.stat-label { color: #475569; font-size: 9px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 2px; }
.stat-value { font-size: 18px; font-weight: 800; font-family: monospace; }

.training-log {
    background: #020617; border: 1px solid #1e293b; border-radius: 8px;
    padding: 12px; font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: #94a3b8; max-height: 320px; overflow-y: auto; line-height: 1.6;
}
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.status-dot.active { background: #4ade80; box-shadow: 0 0 6px #4ade80; animation: pulse 1.5s infinite; }
.status-dot.idle { background: #475569; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.plot-container { background: #0f172a !important; border-radius: 10px !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 2px; }
"""

# ── Header ──────────────────────────────────────────────────────────────────────
HEADER_HTML = """
<div class="arbiter-header">
  <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
    <div>
      <h1 class="arbiter-title">ARBITER</h1>
      <p class="arbiter-subtitle">Autonomous Reasoning-Based Inspector &nbsp;·&nbsp;
         Training Environments with Recursive Oversight</p>
    </div>
    <div style="margin-left:auto; display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
      <span class="arbiter-badge">Multi-Agent</span>
      <span class="arbiter-badge">Causal RL</span>
      <span class="arbiter-badge">Scalable Oversight</span>
    </div>
  </div>
</div>
"""

# ── Hypothesis HTML ─────────────────────────────────────────────────────────────
_HYP_LABELS = {
    "proxy_discrimination":  "PROXY DISCRIM.",
    "adversarial_injection": "ADV. INJECTION",
    "model_drift":           "MODEL DRIFT",
}

def _hyp_html(hyp: dict) -> str:
    def card(k, status):
        label  = _HYP_LABELS.get(k, k.upper())
        status = (status or "ACTIVE").upper()
        if   status == "CONFIRMED":  dot, sl = "#4ade80", "Confirmed"
        elif status == "ACTIVE":     dot, sl = "#fbbf24", "Active"
        elif status == "WEAKENED":   dot, sl = "#f97316", "Weakened"
        else:                        dot, sl = "#475569", "Eliminated"
        strike  = "text-decoration:line-through;" if status == "ELIMINATED" else ""
        glow    = f"box-shadow:0 0 8px {dot};" if status in ("ACTIVE", "CONFIRMED") else ""
        border  = "#4ade80" if status == "CONFIRMED" else "#1e293b"
        weight  = "700" if status in ("ACTIVE", "CONFIRMED") else "400"
        return (
            f"<div style='flex:1;background:#0f172a;border:1px solid {border};"
            f"border-radius:8px;padding:12px;'>"
            f"<div style='font-size:9px;color:#475569;letter-spacing:1px;margin-bottom:5px;font-weight:600;'>"
            f"{k[:3].upper()}</div>"
            f"<div style='font-size:12px;color:#e2e8f0;font-weight:600;margin-bottom:8px;{strike}'>{label}</div>"
            f"<div style='display:flex;align-items:center;gap:6px;'>"
            f"<span style='width:7px;height:7px;border-radius:50%;background:{dot};{glow}display:inline-block;'></span>"
            f"<span style='font-size:11px;color:{dot};font-weight:{weight};'>{sl}</span>"
            f"</div></div>"
        )
    cards = "".join(card(k, v) for k, v in hyp.items())
    return (
        "<div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:12px;'>"
        "<div style='font-size:9px;color:#475569;letter-spacing:1px;font-weight:700;"
        "text-transform:uppercase;margin-bottom:10px;'>Hypothesis Tracker</div>"
        f"<div style='display:flex;gap:8px;'>{cards}</div></div>"
    )


# ── Reward HTML ─────────────────────────────────────────────────────────────────
def _reward_html(rc: dict, done: bool, verdict: bool | None) -> str:
    total = sum(rc.values())
    comps = [
        ("claim",          "Claim Reward",   12, "#00c4e0"),
        ("counterfactual", "Counterfactual",  6, "#a78bfa"),
        ("tom",            "Theory of Mind",  3, "#a78bfa"),
        ("chain",          "Chain Bonus",     8, "#4ade80"),
        ("consistency",    "Consistency",     3, "#f87171"),
        ("budget",         "Budget Eff.",     4, "#00c4e0"),
    ]
    def bar(key, label, mx, color):
        val  = rc.get(key, 0.0)
        pct  = min(100, abs(val / mx) * 100) if mx else 0
        neg  = val < 0
        bc   = "#f87171" if neg else color
        sign      = "" if neg else "+"
        val_color = "#f87171" if neg else "#e2e8f0"
        return (
            f"<div style='margin-bottom:9px;'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>"
            f"<span style='font-size:10px;color:#64748b;'>{label}</span>"
            f"<span style='font-size:10px;font-family:monospace;color:{val_color};'>"
            f"{sign}{val:.1f} / {mx:.1f}</span></div>"
            f"<div style='height:4px;background:#1e293b;border-radius:2px;'>"
            f"<div style='height:100%;width:{pct:.1f}%;background:{bc};border-radius:2px;"
            f"transition:width 0.4s ease;'></div></div></div>"
        )
    bars = "".join(bar(*c) for c in comps)
    score_color = ("#4ade80" if verdict else "#f87171") if done else ("#00c4e0" if total > 10 else "#e2e8f0")
    badge = ""
    if done and verdict is not None:
        bg  = "rgba(16,185,129,0.08)" if verdict else "rgba(239,68,68,0.08)"
        bc2 = "#4ade80" if verdict else "#f87171"
        txt = "VERDICT CORRECT" if verdict else "VERDICT WRONG"
        badge = (
            f"<div style='margin-top:8px;padding:3px 10px;border-radius:12px;"
            f"background:{bg};border:1px solid {bc2};font-size:9px;color:{bc2};"
            f"font-family:monospace;text-align:center;'>{txt}</div>"
        )
    return (
        "<div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;"
        "padding:14px;margin-top:8px;'>"
        "<div style='font-size:9px;color:#475569;letter-spacing:1px;font-weight:700;"
        "text-transform:uppercase;margin-bottom:12px;'>Reward Breakdown</div>"
        "<div style='display:flex;gap:14px;'>"
        "<div style='flex:0 0 90px;display:flex;flex-direction:column;align-items:center;justify-content:center;'>"
        "<div style='font-size:9px;color:#475569;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;'>Total</div>"
        f"<div style='font-size:38px;font-weight:800;font-family:monospace;color:{score_color};line-height:1;'>{total:.1f}</div>"
        "<div style='font-size:11px;color:#475569;font-family:monospace;'>/ 35.0</div>"
        f"{badge}</div>"
        "<div style='width:1px;background:#1e293b;align-self:stretch;'></div>"
        f"<div style='flex:1;'>{bars}</div></div></div>"
    )


# ── Bottom stats bar ────────────────────────────────────────────────────────────
def _bottom_stats(step: int, budget: int, reward: float, claims: list, level: str) -> str:
    total  = len(claims)
    correct = sum(1 for c in claims if c.get("correct"))
    acc    = round(correct / total * 100) if total else 0
    bpct   = budget / 20
    bc     = "#4ade80" if bpct > 0.6 else ("#fbbf24" if bpct > 0.3 else "#f87171")
    rc     = "#4ade80" if reward >= 0 else "#f87171"
    def pill(lbl, val, col):
        return (
            f"<div class='stat-pill' style='border-color:{col}22;'>"
            f"<span class='stat-label'>{lbl}</span>"
            f"<span class='stat-value' style='color:{col};'>{val}</span></div>"
        )
    return (
        "<div class='stats-bar'>"
        + pill("Step",     f"{step}/20",    "#60a5fa")
        + pill("Budget",   str(budget),     bc)
        + pill("Reward",   f"{reward:+.2f}", rc)
        + pill("Claims",   str(total),      "#a78bfa")
        + pill("Accuracy", f"{acc}%",       "#34d399")
        + pill("Level",    level,           "#34d399")
        + "</div>"
    )


# ── Claims HTML ─────────────────────────────────────────────────────────────────
def _claims_html(claims: list) -> str:
    if not claims:
        return (
            "<div style='color:#334155;font-style:italic;font-size:12px;"
            "padding:20px;text-align:center;background:#0f172a;"
            "border:1px solid #1e293b;border-radius:10px;'>"
            "No claims yet — the investigation will build them automatically.</div>"
        )
    html = (
        "<div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;"
        "padding:12px;max-height:340px;overflow-y:auto;'>"
        "<div style='font-size:9px;color:#475569;letter-spacing:1px;font-weight:700;"
        "text-transform:uppercase;margin-bottom:10px;'>Claim Chain</div>"
    )
    for c in claims:
        ct      = c.get("claim_type", "causal").upper()
        correct = c.get("correct", True)
        rd      = c.get("reward_delta", 0.0)
        col     = "#4ade80" if correct else "#f87171"
        bg      = "rgba(20,83,45,0.35)" if correct else "rgba(69,10,10,0.35)"
        step_n  = c.get("step", "?")
        if c.get("claim_type") == "causal":
            detail = (f"{c.get('cause_feature','?')} &rarr; {c.get('effect_outcome','?')} "
                      f"via <em>{c.get('mechanism','?')}</em> [{c.get('confidence','?')}]")
        elif c.get("claim_type") == "counterfactual":
            detail = (f"If <em>{c.get('counterfactual_feature','?')}</em> changed on "
                      f"{c.get('subject_record','?')} &rarr; {c.get('predicted_outcome_change','?')}")
        else:
            detail = (f"Defender {c.get('defender_action','?')} "
                      f"<em>{c.get('target_link','?')}</em>")
        html += (
            f"<div style='background:{bg};border-left:3px solid {col};"
            f"padding:6px 10px;margin:4px 0;border-radius:0 6px 6px 0;font-size:11px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;'>"
            f"<span style='color:{col};font-weight:700;font-size:10px;letter-spacing:0.5px;'>"
            f"[{ct}] step {step_n}</span>"
            f"<span style='color:{col};font-weight:700;font-family:monospace;'>{rd:+.2f}</span></div>"
            f"<span style='color:#cbd5e1;'>{detail}</span></div>"
        )
    html += "</div>"
    return html


# ── Graph drawing ───────────────────────────────────────────────────────────────
def draw_graph(render_data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")
    if not render_data or not render_data.get("graph_nodes"):
        ax.text(0.5, 0.5, "Start an episode to see the causal graph.",
                color="#334155", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, style="italic")
        ax.axis("off")
        return fig
    G = nx.DiGraph()
    for node in render_data["graph_nodes"]:
        G.add_node(node["id"], **node)
    for edge in render_data["graph_edges"]:
        G.add_edge(edge["source"], edge["target"], **edge)
    queried = set(render_data.get("queried_nodes", []))
    node_colors, node_sizes, edge_widths = [], [], []
    for n, d in G.nodes(data=True):
        if n in queried:
            node_colors.append("#fbbf24"); node_sizes.append(700)
        elif d.get("proxy"):
            node_colors.append("#f87171"); node_sizes.append(500)
        elif d.get("node_type") == "outcome":
            node_colors.append("#60a5fa"); node_sizes.append(600)
        elif d.get("node_type") == "policy":
            node_colors.append("#a78bfa"); node_sizes.append(550)
        else:
            node_colors.append("#1e3a5f"); node_sizes.append(400)
    for u, v, d in G.edges(data=True):
        edge_widths.append(1.5 if d.get("anomalous") else 0.8)
    try:
        pos = nx.spring_layout(G, seed=42, k=2.2)
    except Exception:
        pos = nx.random_layout(G)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="#1e3a5f",
                           arrows=True, arrowsize=12, width=edge_widths, alpha=0.7)
    nx.draw_networkx_nodes(G, pos=pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes, alpha=0.95)
    nx.draw_networkx_labels(G, pos=pos, ax=ax,
                            font_color="#cbd5e1", font_size=6, font_weight="bold")
    legend = [
        mpatches.Patch(color="#fbbf24", label="Queried"),
        mpatches.Patch(color="#f87171", label="Proxy / Anomalous"),
        mpatches.Patch(color="#60a5fa", label="Outcome"),
        mpatches.Patch(color="#a78bfa", label="Policy"),
        mpatches.Patch(color="#1e3a5f", label="Feature"),
    ]
    ax.legend(handles=legend, loc="upper left", facecolor="#0f172a",
              edgecolor="#334155", labelcolor="#94a3b8", fontsize=7, framealpha=0.95)
    ax.axis("off")
    ax.set_title(
        f"Causal Graph  ·  {len(G.nodes)} nodes  ·  {len(G.edges)} edges  ·  {len(queried)} queried",
        color="#475569", fontsize=8, pad=8)
    fig.tight_layout(pad=0.5)
    return fig


# ── Episode step logic ──────────────────────────────────────────────────────────
def _do_step():
    global _step_index, _claims, _hypotheses, _budget, _reward_total
    global _reward_components, _episode_done, _verdict_correct, _obs, _render

    if _env is None or _episode_done or _step_index >= len(SCRIPTED_ACTIONS):
        return None, False

    action = copy.deepcopy(SCRIPTED_ACTIONS[_step_index])
    try:
        _obs, reward, done, info = _env.step(action)
        _render = _env.render()
    except Exception:
        return None, False

    _step_index += 1
    cost = 2 if action["type"] == "QUERY_COUNTERFACTUAL" else (1 if action["type"].startswith("QUERY_") else 0)
    _budget = max(0, _budget - cost)

    if action["type"] == "FLAG_HYPOTHESIS":
        _hypotheses[action["hypothesis_type"]] = action["status"]
    if _obs and _obs.get("hypothesis_flags"):
        for k, v in _obs["hypothesis_flags"].items():
            if k in _hypotheses:
                _hypotheses[k] = v

    if action["type"].startswith("CLAIM_"):
        ver = info.get("verification", {})
        true_c  = sum(1 for x in ver.values() if x is True)
        false_c = sum(1 for x in ver.values() if x is False)
        correct = true_c >= false_c
        cd      = action.get("claim", {})
        ct      = ("causal"       if "CAUSAL"      in action["type"] else
                   "counterfactual" if "COUNTERFACTUAL" in action["type"] else
                   "theory_of_mind")
        _claims.append({"claim_type": ct, "step": _step_index - 1,
                         "correct": correct, "reward_delta": reward, **cd})
        comp = ("tom" if "THEORY" in action["type"] else
                "counterfactual" if "COUNTERFACTUAL" in action["type"] else "claim")
        _reward_components[comp] = _reward_components.get(comp, 0.0) + reward
        _reward_total += reward

    if done:
        _episode_done = True
        er = info.get("episode_reward", {})
        if er:
            interm = er.get("intermediate", {})
            term   = er.get("terminal", {})
            _reward_components = {
                "claim":          interm.get("claim_reward",         0.0),
                "counterfactual": interm.get("counterfactual_reward", 0.0),
                "tom":            interm.get("tom_reward",            0.0),
                "chain":          term.get("chain_score",            0.0),
                "consistency":    term.get("consistency_penalty",    0.0),
                "budget":         term.get("budget_efficiency",      0.0),
            }
            _reward_total    = er.get("total", 0.0)
            _verdict_correct = term.get("verdict_correct")

    return reward, done


def _collect_outputs(level_str: str, status: str, timer_active: bool):
    """Return all 7 Investigate tab outputs."""
    return (
        draw_graph(_render or {}),
        _claims_html(_claims),
        _reward_html(_reward_components, _episode_done, _verdict_correct),
        _hyp_html(_hypotheses),
        _bottom_stats(_step_index, _budget, _reward_total, _claims, level_str),
        status,
        gr.update(active=timer_active),
    )


def _level_str() -> str:
    if _obs and _obs.get("level"):
        return f"L{_obs['level']}"
    return f"L{_current_level}"


# ── Episode handlers ────────────────────────────────────────────────────────────
def new_episode_fn(level_radio, seed_num):
    global _env, _obs, _render, _step_index, _claims, _hypotheses, _budget
    global _reward_total, _reward_components, _episode_done, _verdict_correct, _current_level

    level = int(str(level_radio).replace("L", "").strip()) if level_radio else 3
    _current_level = level
    _env   = ArbiterEnv(level=level)
    seed   = int(seed_num) if seed_num else None
    _obs   = _env.reset(seed=seed)
    _render = _env.render()
    _step_index    = 0
    _claims        = []
    _hypotheses    = {"proxy_discrimination": "ACTIVE",
                      "adversarial_injection": "ACTIVE",
                      "model_drift": "ACTIVE"}
    _budget        = 20
    _reward_total  = 0.0
    _reward_components = {"claim": 0.0, "counterfactual": 0.0, "tom": 0.0,
                           "chain": 0.0, "consistency": 0.0, "budget": 0.0}
    _episode_done    = False
    _verdict_correct = None
    return _collect_outputs(_level_str(), f"Episode started at Level {level}. Seed: {seed}.", False)


def step_fn():
    if _env is None:
        return _collect_outputs("—", "No episode — click New Episode first.", False)
    if _episode_done:
        return _collect_outputs(_level_str(), "Episode complete.", False)
    if _step_index >= len(SCRIPTED_ACTIONS):
        return _collect_outputs(_level_str(), "All scripted steps exhausted.", False)
    _do_step()
    atype   = SCRIPTED_ACTIONS[min(_step_index - 1, len(SCRIPTED_ACTIONS) - 1)]["type"]
    suffix  = " — EPISODE COMPLETE" if _episode_done else ""
    return _collect_outputs(_level_str(), f"Step {_step_index}: {atype}{suffix}", False)


def auto_run_fn():
    if _env is None:
        return _collect_outputs("—", "No episode — click New Episode first.", False)
    return _collect_outputs(_level_str(), "Auto-run started...", True)


def pause_fn():
    return _collect_outputs(_level_str(), "Paused.", False)


def auto_tick_fn():
    if _env is None or _episode_done or _step_index >= len(SCRIPTED_ACTIONS):
        return _collect_outputs(_level_str(), "Episode complete.", False)
    _do_step()
    atype   = SCRIPTED_ACTIONS[min(_step_index - 1, len(SCRIPTED_ACTIONS) - 1)]["type"]
    suffix  = " — EPISODE COMPLETE" if _episode_done else ""
    still   = not _episode_done and _step_index < len(SCRIPTED_ACTIONS)
    return _collect_outputs(_level_str(), f"Auto: step {_step_index} — {atype}{suffix}", still)


# ── GRPO Training monitor ───────────────────────────────────────────────────────
def _read_stdout_thread(proc):
    global _log_buffer
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            _log_buffer.append(line)
            if len(_log_buffer) > 300:
                _log_buffer.pop(0)
    except Exception:
        pass


def start_grpo_training(checkpoint, level, episodes, output):
    global _training_process, _log_buffer
    if _training_process and _training_process.poll() is None:
        return "Training already running. Click Abort first."
    checkpoint = (checkpoint or "").strip() or "lora_sft_v4"
    output     = (output     or "").strip() or "lora_grpo"
    log_path   = ROOT / "logs" / "grpo_training.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    _log_buffer = [f"[ARBITER] Launching GRPO training...",
                   f"[ARBITER] Checkpoint : {checkpoint}",
                   f"[ARBITER] Level      : {level}",
                   f"[ARBITER] Episodes   : {episodes}",
                   f"[ARBITER] Output     : {output}",
                   f"[ARBITER] Log file   : {log_path}", ""]
    cmd = [sys.executable, "-m", "arbiter.training.grpo_trainer",
           "--checkpoint", str(ROOT / checkpoint),
           "--level",      str(level),
           "--episodes",   str(episodes),
           "--output",     str(ROOT / output),
           "--log_file",   str(log_path)]
    _training_process = subprocess.Popen(cmd, cwd=str(ROOT),
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    threading.Thread(target=_read_stdout_thread, args=(_training_process,), daemon=True).start()
    return f"Training started (PID {_training_process.pid}). Monitor updates every 2 s."


def abort_grpo_training():
    global _training_process
    if _training_process and _training_process.poll() is None:
        _training_process.terminate()
        _log_buffer.append("[ARBITER] Training aborted by user.")
        return "Training aborted."
    return "No active training process."


def _read_grpo_log():
    entries = []
    try:
        with open(ROOT / "logs" / "grpo_training.jsonl") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return entries


def training_stats_html(entries: list) -> str:
    is_running = bool(_training_process and _training_process.poll() is None)
    dot_class  = "active" if is_running else "idle"
    status_str = "TRAINING" if is_running else ("IDLE" if not entries else "COMPLETE")
    status_col = "#4ade80" if is_running else ("#fbbf24" if entries else "#475569")
    if not entries:
        ep = reward = loss = evasion = level = "—"
    else:
        last    = entries[-1]
        ep      = last.get("episode", "—")
        reward  = f"{last.get('mean_reward', 0):.2f}"
        loss    = f"{last.get('grpo_loss', 0):.4f}"
        evasion = f"{last.get('defender_evasion', 0):.1%}"
        level   = last.get("level", "—")
    def pill(lbl, val, col):
        return (f"<div class='stat-pill' style='border-color:{col}33;'>"
                f"<span class='stat-label'>{lbl}</span>"
                f"<span class='stat-value' style='color:{col};'>{val}</span></div>")
    return (
        f"<div class='stats-bar'>"
        f"<div class='stat-pill' style='border-color:{status_col}33;'>"
        f"<span class='stat-label'>Status</span>"
        f"<span class='stat-value' style='color:{status_col};font-size:12px;display:flex;align-items:center;'>"
        f"<span class='status-dot {dot_class}'></span>{status_str}</span></div>"
        + pill("Episode", ep, "#60a5fa")
        + pill("Reward",  reward, "#4ade80")
        + pill("Loss",    loss, "#f87171")
        + pill("Evasion", evasion, "#fbbf24")
        + pill("Level",   level, "#a78bfa")
        + "</div>"
    )


def draw_training_chart(entries: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")
    if not entries:
        ax.text(0.5, 0.5, "Start GRPO training to see the live reward curve.",
                color="#334155", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, style="italic")
        ax.axis("off")
        return fig
    eps     = [e["episode"]         for e in entries]
    rewards = [e["mean_reward"]      for e in entries]
    evasion = [e["defender_evasion"] for e in entries]
    if len(rewards) >= 5:
        from scipy.ndimage import uniform_filter1d
        rewards_sm = uniform_filter1d(rewards, size=min(10, len(rewards) // 2 or 1))
    else:
        rewards_sm = rewards
    ax.plot(eps, rewards, color="#1e3a5f", linewidth=1, alpha=0.4)
    ax.plot(eps, rewards_sm, color="#3b82f6", linewidth=2.5, label="Auditor Reward", zorder=3)
    ax2 = ax.twinx()
    ax2.plot(eps, [e * 100 for e in evasion],
             color="#f87171", linewidth=1.8, linestyle="--", alpha=0.85, label="Defender Evasion %")
    ax2.set_ylabel("Defender Evasion %", color="#f87171", fontsize=8)
    ax2.tick_params(colors="#f87171", labelsize=7)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#1e293b")
    for e in entries:
        if e.get("level_advanced"):
            ax.axvline(e["episode"], color="#fbbf24", linewidth=1, linestyle=":", alpha=0.7)
            ax.text(e["episode"] + 1, min(rewards) * 0.95, f"L{e['level_advanced']}", color="#fbbf24", fontsize=7)
    ax.set_xlabel("Episode", color="#475569", fontsize=8)
    ax.set_ylabel("Mean Reward", color="#3b82f6", fontsize=8)
    final_r = rewards_sm[-1] if rewards_sm else 0
    ax.set_title(f"GRPO Training  ·  {len(entries)} episodes  ·  Latest reward: {final_r:.2f}",
                 color="#64748b", fontsize=9, pad=6)
    ax.tick_params(colors="#475569", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e293b")
    ax.grid(color="#0f172a", linewidth=0.8, zorder=0)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, facecolor="#0f172a", edgecolor="#334155",
              labelcolor="#94a3b8", fontsize=8, loc="upper left")
    fig.tight_layout(pad=0.5)
    return fig


def get_log_html() -> str:
    if not _log_buffer:
        return "<div class='training-log'>Waiting for training to start...</div>"
    lines_html = []
    for line in _log_buffer[-80:]:
        if "Advanced" in line or ("Level" in line and "***" in line):
            lines_html.append(f"<div style='color:#fbbf24;font-weight:700;'>{line}</div>")
        elif "Error" in line or "error" in line:
            lines_html.append(f"<div style='color:#f87171;'>{line}</div>")
        elif line.startswith("[ARBITER]"):
            lines_html.append(f"<div style='color:#60a5fa;'>{line}</div>")
        else:
            lines_html.append(f"<div>{line}</div>")
    return (
        "<div class='training-log' id='tlog'>"
        + "".join(lines_html)
        + "<script>document.getElementById('tlog').scrollTop=99999;</script></div>"
    )


_IDLE_CHART_HTML = """
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    height:280px;gap:12px;">
  <div style="font-size:2.5rem;">📡</div>
  <div style="color:#475569;font-size:1rem;font-weight:600;">No Training in Progress</div>
  <div style="color:#334155;font-size:0.8rem;text-align:center;max-width:320px;line-height:1.6;">
    Configure settings above and click
    <strong style="color:#60a5fa;">Start GRPO Training</strong> to begin.
  </div>
</div>
"""


def refresh_training(_checkpoint=None, _level=None, _episodes=None, _output=None):
    is_running = bool(_training_process and _training_process.poll() is None)
    entries    = _read_grpo_log()
    if not is_running and not entries:
        return (training_stats_html([]), gr.update(visible=True),
                gr.update(visible=False), get_log_html(), gr.update(active=False))
    chart_fig = draw_training_chart(entries)
    return (training_stats_html(entries), gr.update(visible=False),
            gr.update(value=chart_fig, visible=True), get_log_html(),
            gr.update(active=is_running))


# ── Arms race chart ─────────────────────────────────────────────────────────────
def draw_arms_race_from_files() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")
    entries = _read_grpo_log()
    if not entries:
        try:
            summary = json.loads((ROOT / "lora_grpo" / "training_summary.json").read_text())
            rewards = summary.get("reward_curve", [])
            evasion = summary.get("defender_evasion", [])
        except Exception:
            rewards, evasion = [], []
    else:
        rewards = [e["mean_reward"]      for e in entries]
        evasion = [e["defender_evasion"] for e in entries]
    if not rewards:
        ax.text(0.5, 0.5, "Run GRPO training to see the arms-race curve.",
                color="#334155", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, style="italic")
        ax.axis("off")
        return fig
    eps = range(len(rewards))
    ax.plot(eps, rewards, color="#3b82f6", linewidth=2.5, label="Auditor Reward", zorder=3)
    if evasion:
        ax2 = ax.twinx()
        ax2.plot(range(len(evasion)), [e * 100 for e in evasion],
                 color="#f87171", linewidth=2, linestyle="--", label="Defender Evasion %")
        ax2.set_ylabel("Defender Evasion %", color="#f87171", fontsize=8)
        ax2.tick_params(colors="#f87171", labelsize=7)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#1e293b")
    ax.set_xlabel("Episode", color="#475569", fontsize=8)
    ax.set_ylabel("Mean Reward", color="#3b82f6", fontsize=8)
    ax.set_title("Arms Race: Auditor vs Defender Co-Evolution", color="#64748b", fontsize=9)
    ax.tick_params(colors="#475569", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e293b")
    ax.grid(color="#0f172a", linewidth=0.8)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#94a3b8", fontsize=8)
    fig.tight_layout(pad=0.5)
    return fig


# ── Gradio app ──────────────────────────────────────────────────────────────────
def build_demo() -> gr.Blocks:
    init_hyp = {"proxy_discrimination": "ACTIVE", "adversarial_injection": "ACTIVE", "model_drift": "ACTIVE"}
    init_rc  = {"claim": 0.0, "counterfactual": 0.0, "tom": 0.0, "chain": 0.0, "consistency": 0.0, "budget": 0.0}

    with gr.Blocks(title="ARBITER — AI Oversight Training Environment") as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ── Tab 1: Investigate ─────────────────────────────────────────────
            with gr.Tab("Investigate"):

                # ── Controls row ──
                with gr.Row():
                    level_radio = gr.Radio(
                        choices=["L1", "L2", "L3", "L4", "L5"],
                        value="L3", label="Curriculum Level", scale=3,
                    )
                    model_radio = gr.Radio(
                        choices=["UNTRAINED", "SFT ONLY", "FULL ARBITER"],
                        value="UNTRAINED", label="Model Mode", scale=4,
                    )
                    seed_num = gr.Number(value=42, label="Seed", precision=0, scale=1)

                with gr.Row():
                    new_btn   = gr.Button("New Episode",  variant="primary",    scale=2)
                    step_btn  = gr.Button("Step  →",      variant="secondary",  scale=2)
                    auto_btn  = gr.Button("Auto-run  ▶",  variant="secondary",  scale=2)
                    pause_btn = gr.Button("Pause  ‖",     variant="secondary",  scale=1)

                status_md  = gr.Markdown("_Click **New Episode** to start. Then use **Step** or **Auto-run**._")
                stats_html = gr.HTML(_bottom_stats(0, 20, 0.0, [], "L3"))

                # ── Main layout ──
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6):
                        graph_plot = gr.Plot(label="Causal Decision Graph", show_label=True)

                    with gr.Column(scale=5):
                        claim_html  = gr.HTML(_claims_html([]))
                        reward_html = gr.HTML(_reward_html(init_rc, False, None))
                        hyp_html    = gr.HTML(_hyp_html(init_hyp))

                # ── Auto-step timer (inactive until Auto-run clicked) ──
                step_timer = gr.Timer(value=1.5, active=False)

                _all = [graph_plot, claim_html, reward_html, hyp_html, stats_html, status_md, step_timer]

                new_btn.click(fn=new_episode_fn,  inputs=[level_radio, seed_num], outputs=_all)
                step_btn.click(fn=step_fn,                                         outputs=_all)
                auto_btn.click(fn=auto_run_fn,                                     outputs=_all)
                pause_btn.click(fn=pause_fn,                                        outputs=_all)
                step_timer.tick(fn=auto_tick_fn,                                    outputs=_all)

                demo.load(fn=lambda: new_episode_fn("L3", 42), outputs=_all)

            # ── Tab 2: Training Monitor ────────────────────────────────────────
            with gr.Tab("Training Monitor"):

                gr.Markdown("## GRPO Training Control Center")
                gr.Markdown(
                    "Configure and launch GRPO reinforcement learning. "
                    "The reward curve, level advances, and defender evasion update live every 2 s."
                )
                with gr.Row():
                    t_checkpoint = gr.Textbox(value="lora_sft_v4", label="SFT Checkpoint", scale=3)
                    t_output     = gr.Textbox(value="lora_grpo",   label="Output Path",   scale=3)
                    t_level      = gr.Slider(1, 5, value=1, step=1, label="Start Level",  scale=2)
                    t_episodes   = gr.Slider(50, 500, value=300, step=50, label="Episodes", scale=2)
                with gr.Row():
                    train_btn = gr.Button("Start GRPO Training", variant="primary",    scale=2)
                    abort_btn = gr.Button("Abort Training",       variant="secondary", scale=1)

                train_status   = gr.Markdown("_Configure settings above and click Start._")
                train_stats_ht = gr.HTML(training_stats_html([]))

                train_chart_ph = gr.HTML(_IDLE_CHART_HTML, visible=True)
                train_chart    = gr.Plot(label="Live Reward Curve", show_label=True, visible=False)
                train_log      = gr.HTML(get_log_html())

                grpo_timer = gr.Timer(value=2.0, active=False)
                _grpo_outs = [train_stats_ht, train_chart_ph, train_chart, train_log, grpo_timer]
                grpo_timer.tick(fn=refresh_training,
                                inputs=[t_checkpoint, t_level, t_episodes, t_output],
                                outputs=_grpo_outs)

                def _start_and_activate(cp, lv, ep, out):
                    msg = start_grpo_training(cp, lv, ep, out)
                    return msg, gr.update(active=True)

                train_btn.click(fn=_start_and_activate,
                                inputs=[t_checkpoint, t_level, t_episodes, t_output],
                                outputs=[train_status, grpo_timer])
                abort_btn.click(fn=abort_grpo_training, outputs=[train_status])

            # ── Tab 3: Arms Race ───────────────────────────────────────────────
            with gr.Tab("Arms Race"):
                gr.Markdown("## Auditor vs Defender Co-Evolution")
                gr.Markdown(
                    "Auditor reward and Defender evasion rate over GRPO training. "
                    "Loads from the live log or the last saved checkpoint summary."
                )
                arms_plot   = gr.Plot(label="Arms Race", show_label=True)
                refresh_btn = gr.Button("Refresh", variant="secondary")
                refresh_btn.click(draw_arms_race_from_files, outputs=[arms_plot])
                demo.load(draw_arms_race_from_files, outputs=[arms_plot])

            # ── Tab 4: Reference ───────────────────────────────────────────────
            with gr.Tab("Reference"):
                gr.Markdown("## Action Reference")
                param_docs = {
                    "QUERY_RECORDS":
                        "- `feature_filter` — dict of feature→value pairs (empty = all records)\n"
                        "- `outcome_filter` — `\"approved\"` | `\"denied\"` | null\n"
                        "- `time_range` — `[start, end]` | null\n"
                        "- **Budget cost: 1**",
                    "QUERY_FEATURE_DISTRIBUTION":
                        "- `feature_id` — exact feature name\n"
                        "- `group_by` — optional second feature to split by\n"
                        "- **Budget cost: 1**",
                    "QUERY_COUNTERFACTUAL":
                        "- `record_id` — e.g. `rec_0041`\n"
                        "- `feature_id` — feature to intervene on\n"
                        "- `counterfactual_value` — the new value\n"
                        "- **Budget cost: 2**",
                    "FLAG_HYPOTHESIS":
                        "- `hypothesis_type` — `proxy_discrimination` | `adversarial_injection` | `model_drift`\n"
                        "- `status` — `ACTIVE` | `WEAKENED` | `ELIMINATED` | `CONFIRMED`\n"
                        "- **Free action**",
                    "CLAIM_CAUSAL":
                        "- `cause_feature`, `effect_outcome`, `mechanism`, `direction`, `confidence`\n"
                        "- `basis_records` — list of rec_XXXX IDs\n"
                        "- `anomaly_type` — `proxy_discrimination` | `adversarial_injection` | `model_drift`\n"
                        "- **Reward: up to +1.0**",
                    "CLAIM_COUNTERFACTUAL":
                        "- `subject_record`, `counterfactual_feature`\n"
                        "- `predicted_outcome_change` — `approved` | `denied`\n"
                        "- `confidence`, `basis`\n"
                        "- **Reward: up to +2.0** _(requires prior QUERY_COUNTERFACTUAL)_",
                    "CLAIM_THEORY_OF_MIND":
                        "- `defender_action`, `target_link`, `obfuscation_method`, `confidence`, `basis`\n"
                        "- **+3.0 bonus if fully correct** _(Level 4+ only)_",
                    "SUBMIT_REPORT":
                        "- `anomaly_type`, `primary_evidence_chain`, `affected_demographic`, `recommended_action`\n"
                        "- **Terminal action — ends episode and triggers full reward.**",
                }
                for atype, doc in param_docs.items():
                    with gr.Accordion(atype, open=False):
                        gr.Markdown(doc)

    return demo


def main():
    demo = build_demo()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    )


if __name__ == "__main__":
    main()
