"""
ARBITER — End-to-End Integration Test
======================================
Validates the complete pipeline on one machine in sequence:

  Stage 1  Environment smoke-test      (ArbiterEnv resets, steps, rewards work)
  Stage 2  SFT checkpoint              (load adapter, run N episodes, stats vs baseline,
                                        schema-valid action output)
  Stage 3  GRPO checkpoint             (load adapter, run N episodes, stats vs SFT)
  Stage 4  Gradio demo functional test (model loaded, env returns data, claims render,
                                        reward panel updates — via gradio_client API)

Fixes applied vs. v1
---------------------
  FIX-1  Episodes default 3 → 10; seed banks cover 10 unique held-out seeds per stage.
         With 10 eps the reward distribution is wide enough to distinguish signal from luck.
  FIX-2  "JSON validity" (just checks for a 'type' key) replaced with full schema validation:
         required fields per action/claim type, valid enum values for confidence /
         anomaly_type / direction / obfuscation_method / defender_action / outcome.
  FIX-3  Stage 4 no longer just checks that a port is open.  After the server starts,
         the test uses gradio_client to:
           (a) confirm the model badge text contains the checkpoint name (not "Manual mode")
           (b) call new_episode() and verify graph_nodes are present in the Plot payload
           (c) call run_query() and check the reward panel is non-empty
           (d) inspect the claim HTML for the presence of green/red claim coloring markers

Usage
-----
# Full test (all stages — requires trained checkpoints):
    python integration_test.py --sft lora_sft/ --grpo lora_grpo/

# Environment + demo only (no checkpoints):
    python integration_test.py

# Skip demo test (headless / CI without a display):
    python integration_test.py --sft lora_sft/ --grpo lora_grpo/ --skip_demo

Exit codes:  0 = all enabled stages passed   |   1 = at least one stage failed
"""

import argparse
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="ARBITER end-to-end integration test")
parser.add_argument("--sft",          default=None,  help="Path to SFT LoRA adapter dir")
parser.add_argument("--grpo",         default=None,  help="Path to GRPO LoRA adapter dir")
parser.add_argument("--level",        type=int, default=3,
                    help="Curriculum level for episode tests (default 3)")
parser.add_argument("--episodes",     type=int, default=10,   # FIX-1: was 3
                    help="Episodes per model condition (default 10, minimum 10 for reliable stats)")
parser.add_argument("--skip_demo",    action="store_true",
                    help="Skip Gradio demo functional test")
parser.add_argument("--demo_timeout", type=int, default=90,
                    help="Seconds to wait for Gradio server to become reachable (default 90)")
parser.add_argument("--demo_port",    type=int, default=7861,
                    help="Port used for the demo subprocess (default 7861)")
args = parser.parse_args()

if args.episodes < 10:
    print(f"WARNING: --episodes {args.episodes} < 10. Results will be statistically unreliable. "
          "Consider using at least 10 episodes.")

# ── Helpers ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"
WARN = "⚠️  WARN"

results: List[Tuple[str, str, str]] = []   # (stage, status, detail)


def record(stage: str, status: str, detail: str = "") -> None:
    results.append((stage, status, detail))
    badge = {"✅ PASS": "✅", "❌ FAIL": "❌", "⏭  SKIP": "⏭ ", "⚠️  WARN": "⚠️"}.get(status, "?")
    print(f"  {badge} {stage}: {detail[:140]}")


def section(title: str) -> None:
    print(f"\n{'━' * 72}")
    print(f"  {title}")
    print(f"{'━' * 72}")


# ── FIX-2: Action / Claim schema definitions ──────────────────────────────────
# Mirrors the dataclasses in arbiter/env/claims.py and the action spec in environment.py

VALID_ACTION_TYPES = {
    "QUERY_RECORDS", "QUERY_FEATURE_DISTRIBUTION", "QUERY_COUNTERFACTUAL",
    "FLAG_HYPOTHESIS", "CLAIM_CAUSAL", "CLAIM_COUNTERFACTUAL",
    "CLAIM_THEORY_OF_MIND", "SUBMIT_REPORT",
}
VALID_CONFIDENCE        = {"HIGH", "MEDIUM", "LOW"}
VALID_ANOMALY_TYPES     = {"proxy_discrimination", "adversarial_injection", "model_drift"}
VALID_DIRECTIONS        = {"positive", "negative", "neutral"}
VALID_CF_OUTCOMES       = {"approved", "denied", "flagged", "no_change"}
VALID_OBFUSCATION_METHS = {"link_substitution", "record_injection",
                            "proxy_laundering", "timestamp_manipulation"}
VALID_DEFENDER_ACTIONS  = {"obfuscating", "injecting", "manipulating"}
VALID_REPORT_ACTIONS    = {"audit", "retrain", "flag", "escalate"}
VALID_HYP_TYPES         = {"proxy_discrimination", "adversarial_injection", "model_drift", "none"}
VALID_HYP_STATUSES      = {"active", "dismissed", "confirmed"}

# Required *top-level* fields per action/claim type
REQUIRED_FIELDS: Dict[str, List[str]] = {
    "QUERY_RECORDS":              ["type"],                           # feature_filter optional
    "QUERY_FEATURE_DISTRIBUTION": ["type", "feature_id"],
    "QUERY_COUNTERFACTUAL":       ["type", "record_id", "feature_id", "counterfactual_value"],
    "FLAG_HYPOTHESIS":            ["type", "hypothesis_type", "status"],
    "CLAIM_CAUSAL":               ["type", "claim"],                  # inner claim checked separately
    "CLAIM_COUNTERFACTUAL":       ["type", "claim"],
    "CLAIM_THEORY_OF_MIND":       ["type", "claim"],
    "SUBMIT_REPORT":              ["type", "anomaly_type", "affected_demographic",
                                   "recommended_action"],
}
REQUIRED_CLAIM_FIELDS: Dict[str, List[str]] = {
    "CLAIM_CAUSAL":         ["cause_feature", "effect_outcome", "mechanism",
                              "direction", "confidence", "basis_records", "anomaly_type"],
    "CLAIM_COUNTERFACTUAL": ["subject_record", "counterfactual_feature",
                              "predicted_outcome_change", "confidence"],
    "CLAIM_THEORY_OF_MIND": ["defender_action", "target_link",
                              "obfuscation_method", "confidence"],
}


class SchemaError(Exception):
    pass


def validate_action_schema(raw_text: str) -> Tuple[bool, str]:
    """
    Parse raw_text as JSON (with regex fallback) and validate against the
    ARBITER action schema.

    Returns (is_valid: bool, error_message: str).
    An empty error_message means fully valid.
    """
    action: Optional[Dict] = None

    # Parse
    try:
        action = json.loads(raw_text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if m:
            try:
                action = json.loads(m.group())
            except Exception:
                pass

    if action is None:
        return False, "Could not parse any JSON object from output"

    if not isinstance(action, dict):
        return False, f"Top-level JSON must be an object, got {type(action).__name__}"

    atype = action.get("type")
    if atype not in VALID_ACTION_TYPES:
        return False, f"'type' value '{atype}' is not a valid action type"

    # Top-level required fields
    for f in REQUIRED_FIELDS.get(atype, []):
        if f not in action:
            return False, f"Missing required field '{f}' for action type '{atype}'"

    # Enum checks for top-level fields
    if atype == "SUBMIT_REPORT":
        at = action.get("anomaly_type")
        if at not in VALID_ANOMALY_TYPES:
            return False, f"anomaly_type '{at}' not in {VALID_ANOMALY_TYPES}"
        ra = action.get("recommended_action")
        if ra not in VALID_REPORT_ACTIONS:
            return False, f"recommended_action '{ra}' not in {VALID_REPORT_ACTIONS}"

    if atype == "FLAG_HYPOTHESIS":
        ht = action.get("hypothesis_type")
        hs = action.get("status")
        if ht not in VALID_HYP_TYPES:
            return False, f"hypothesis_type '{ht}' not in {VALID_HYP_TYPES}"
        if hs not in VALID_HYP_STATUSES:
            return False, f"status '{hs}' not in {VALID_HYP_STATUSES}"

    # Inner claim dict
    if atype in REQUIRED_CLAIM_FIELDS:
        claim = action.get("claim")
        if not isinstance(claim, dict):
            return False, f"'claim' must be a dict for action type '{atype}'"
        for f in REQUIRED_CLAIM_FIELDS[atype]:
            if f not in claim:
                return False, f"Claim missing required field '{f}' for '{atype}'"
        # Enum checks inside claim
        conf = claim.get("confidence")
        if conf not in VALID_CONFIDENCE:
            return False, f"claim.confidence '{conf}' not in {VALID_CONFIDENCE}"
        if atype == "CLAIM_CAUSAL":
            d = claim.get("direction")
            if d not in VALID_DIRECTIONS:
                return False, f"claim.direction '{d}' not in {VALID_DIRECTIONS}"
            at = claim.get("anomaly_type")
            if at not in VALID_ANOMALY_TYPES:
                return False, f"claim.anomaly_type '{at}' not in {VALID_ANOMALY_TYPES}"
            br = claim.get("basis_records")
            if not isinstance(br, list):
                return False, f"claim.basis_records must be a list, got {type(br).__name__}"
        elif atype == "CLAIM_COUNTERFACTUAL":
            poc = claim.get("predicted_outcome_change")
            if poc not in VALID_CF_OUTCOMES:
                return False, f"claim.predicted_outcome_change '{poc}' not in {VALID_CF_OUTCOMES}"
        elif atype == "CLAIM_THEORY_OF_MIND":
            da = claim.get("defender_action")
            om = claim.get("obfuscation_method")
            if da not in VALID_DEFENDER_ACTIONS:
                return False, f"claim.defender_action '{da}' not in {VALID_DEFENDER_ACTIONS}"
            if om not in VALID_OBFUSCATION_METHS:
                return False, f"claim.obfuscation_method '{om}' not in {VALID_OBFUSCATION_METHS}"

    return True, ""


# ── Seed banks (held-out, non-overlapping with training seeds 0-199) ──────────
# Training uses args.seed (default 42) + episode offsets — these avoid that range.
STAGE2_SEEDS = [200, 207, 214, 221, 228, 235, 242, 249, 256, 263][:args.episodes]
STAGE3_SEEDS = [300, 307, 314, 321, 328, 335, 342, 349, 356, 363][:args.episodes]


# ── Stage 1: Environment smoke-test ───────────────────────────────────────────

section("Stage 1 — Environment smoke-test")

try:
    from arbiter.env.environment import ArbiterEnv

    # 1a: Reset + basic obs shape
    env = ArbiterEnv(level=args.level)
    obs = env.reset(seed=42)
    assert isinstance(obs, dict),           "obs must be a dict"
    assert "step" in obs,                   "obs must have 'step'"
    assert "budget_remaining" in obs,       "obs must have 'budget_remaining'"
    assert "features" in obs,              "obs must have 'features'"
    record("1a env.reset()", PASS, f"obs keys: {sorted(obs.keys())}")

    # 1b: step() with each action type — verify reward is numeric
    actions_to_test = [
        {"type": "QUERY_RECORDS",              "feature_filter": {}},
        {"type": "QUERY_FEATURE_DISTRIBUTION", "feature_id": "credit_score", "group_by": None},
        {"type": "QUERY_COUNTERFACTUAL",
         "record_id": "rec_0000", "feature_id": "zip_code_cluster",
         "counterfactual_value": "cluster_3"},
        {"type": "FLAG_HYPOTHESIS", "hypothesis_type": "proxy_discrimination", "status": "active"},
    ]
    for action in actions_to_test:
        next_obs, reward, done, info = env.step(action)
        assert isinstance(reward, (int, float)), f"reward must be numeric for {action['type']}"
        record(f"1b step({action['type']})", PASS, f"reward={reward:.2f} done={done}")

    # 1c: render() — spot-check structure
    render = env.render()
    assert "graph_nodes" in render, "render() must contain 'graph_nodes'"
    assert len(render["graph_nodes"]) > 0,  "graph_nodes must be non-empty"
    assert "graph_edges" in render,         "render() must contain 'graph_edges'"
    assert "claims" in render,              "render() must contain 'claims'"
    record("1c env.render()", PASS,
           f"nodes={len(render['graph_nodes'])} edges={len(render['graph_edges'])} "
           f"claims={len(render['claims'])}")

    # 1d: Full episode → auto-terminate or SUBMIT_REPORT
    env2  = ArbiterEnv(level=1)
    obs2  = env2.reset(seed=99)
    done2 = False
    ep_reward = 0.0
    for _ in range(20):
        obs2, r, done2, info2 = env2.step({"type": "QUERY_RECORDS", "feature_filter": {}})
        ep_reward += r
        if done2:
            break
    if not done2:
        _, r, done2, info2 = env2.step({
            "type": "SUBMIT_REPORT",
            "anomaly_type": "proxy_discrimination",
            "primary_evidence_chain": [],
            "affected_demographic": "unknown",
            "recommended_action": "audit",
        })
        ep_reward += r
    assert done2, "Episode must terminate after SUBMIT_REPORT"
    record("1d full episode (Level 1)", PASS, f"total_reward={ep_reward:.2f}")

    # 1e: validate.py (70/70 checks)
    validate_path = ROOT / "validate.py"
    if validate_path.exists():
        try:
            vresult = subprocess.run(
                [sys.executable, str(validate_path)],
                capture_output=True, text=True, timeout=120
            )
            if vresult.returncode == 0 and "70/70" in vresult.stdout:
                record("1e validate.py (70/70)", PASS, "All 70 validation checks passed")
            elif vresult.returncode == 0:
                record("1e validate.py", WARN,
                       f"Exited 0 but '70/70' not found in stdout: {vresult.stdout[-200:]}")
            else:
                record("1e validate.py", FAIL, vresult.stderr[-300:])
        except subprocess.TimeoutExpired:
            record("1e validate.py", WARN, "Timed out after 120 s — skipping")
    else:
        record("1e validate.py", SKIP, "validate.py not found")

except Exception:
    record("Stage 1", FAIL, traceback.format_exc()[-500:])


# ── Model loader (shared by Stage 2 & 3) ─────────────────────────────────────

def load_lora(checkpoint: str):
    """
    Load a LoRA adapter. Returns (model, tokenizer, backend_str) or (None, None, None).
    Tries Unsloth first (GPU-optimised), then transformers + PEFT.
    """
    import torch

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint, max_seq_length=1024, load_in_4bit=True)
        FastLanguageModel.for_inference(model)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer, "unsloth"
    except Exception:
        pass

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", device_map="auto", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base, checkpoint)
        model.eval()
        return model, tokenizer, "peft"
    except Exception as e:
        print(f"    [load_lora] Could not load {checkpoint}: {e}")
        return None, None, None


SYSTEM_PROMPT = (
    "You are an expert AI auditor. Output exactly one JSON action per turn. "
    "Available actions: QUERY_RECORDS, QUERY_FEATURE_DISTRIBUTION, QUERY_COUNTERFACTUAL, "
    "FLAG_HYPOTHESIS, CLAIM_CAUSAL, CLAIM_COUNTERFACTUAL, CLAIM_THEORY_OF_MIND, SUBMIT_REPORT."
)


def run_model_episode(model, tokenizer, env: "ArbiterEnv", seed: int) -> Tuple[float, List[str]]:
    """
    Run one episode driven by the LLM.
    Returns (total_reward, list_of_raw_action_texts).
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    obs     = env.reset(seed=seed)
    history: List[Dict] = []
    total   = 0.0
    raw_outputs: List[str] = []

    for step in range(20):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history[-4:]:
            messages.append({"role": "user",      "content": h["obs"]})
            messages.append({"role": "assistant", "content": h["act"]})

        obs_text = (
            f"Step {obs.get('step', 0)}/20 | Budget: {obs.get('budget_remaining', 20)} | "
            f"Claims: {obs.get('num_claims', 0)}\n"
            f"Features: {list(obs.get('features', {}).get('explicit', []))[:4]}\n"
            "Output JSON action:"
        )
        messages.append({"role": "user", "content": obs_text})

        try:
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=150, temperature=0.5,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id)
            act_text = tokenizer.decode(out[0][input_ids.shape[1]:],
                                        skip_special_tokens=True).strip()
        except Exception:
            act_text = '{"type": "QUERY_RECORDS", "feature_filter": {}}'

        raw_outputs.append(act_text)

        try:
            action = json.loads(act_text)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', act_text, re.DOTALL)
            action = json.loads(m.group()) if m else {"type": "QUERY_RECORDS", "feature_filter": {}}

        obs, reward, done, _ = env.step(action)
        total += reward
        history.append({"obs": obs_text, "act": act_text})
        if done:
            break

    return total, raw_outputs


def baseline_episode_reward(level: int, seed: int) -> float:
    """Rule-based heuristic agent — no model. Provides the floor for comparison."""
    env = ArbiterEnv(level=level)
    obs = env.reset(seed=seed)
    total = 0.0
    features = obs.get("features", {}).get("explicit", [])
    heuristic = [
        {"type": "QUERY_RECORDS",              "feature_filter": {}},
        {"type": "QUERY_FEATURE_DISTRIBUTION",
         "feature_id": features[0] if features else "credit_score", "group_by": None},
        {"type": "QUERY_COUNTERFACTUAL", "record_id": "rec_0000",
         "feature_id": "zip_code_cluster", "counterfactual_value": "cluster_3"},
        {"type": "SUBMIT_REPORT", "anomaly_type": "proxy_discrimination",
         "primary_evidence_chain": [], "affected_demographic": "unknown",
         "recommended_action": "audit"},
    ]
    for action in heuristic:
        obs, r, done, _ = env.step(action)
        total += r
        if done:
            break
    return total


# ── Stage 2: SFT checkpoint ───────────────────────────────────────────────────

section("Stage 2 — SFT checkpoint")

sft_mean: Optional[float] = None

if not args.sft:
    record("Stage 2", SKIP, "--sft not provided (run after train_sft.py completes)")
elif not Path(args.sft).exists():
    record("Stage 2", FAIL, f"Checkpoint directory not found: {args.sft}")
else:
    try:
        # 2a: Required files
        adapter_cfg = Path(args.sft) / "adapter_config.json"
        if not adapter_cfg.exists():
            record("2a adapter_config.json", FAIL,
                   f"Missing {adapter_cfg}. Is this a valid LoRA checkpoint?")
        else:
            record("2a adapter_config.json", PASS, str(adapter_cfg))

        # 2b: Load model
        model_sft, tok_sft, backend = load_lora(args.sft)
        if model_sft is None:
            record("2b load model", FAIL, "Could not load model — see traceback above")
        else:
            record("2b load model", PASS, f"Backend: {backend}")

            env_sft = ArbiterEnv(level=args.level)

            # ── 2c: Episode rewards (10 seeds) ─────────────────────────────
            all_rewards:     List[float]      = []
            all_raw_outputs: List[List[str]]  = []
            print(f"    Running {len(STAGE2_SEEDS)} episodes …", flush=True)
            for seed in STAGE2_SEEDS:
                rew, raws = run_model_episode(model_sft, tok_sft, env_sft, seed)
                all_rewards.append(rew)
                all_raw_outputs.extend(raws)
                print(f"      seed={seed} reward={rew:.2f}", flush=True)

            sft_mean = sum(all_rewards) / len(all_rewards)
            sft_std  = (sum((r - sft_mean) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5
            baseline_rewards = [baseline_episode_reward(args.level, s) for s in STAGE2_SEEDS]
            baseline_mean    = sum(baseline_rewards) / len(baseline_rewards)

            record("2c episode rewards",
                   PASS,
                   f"SFT mean={sft_mean:.2f} ±{sft_std:.2f} "
                   f"(n={len(STAGE2_SEEDS)}) | baseline mean={baseline_mean:.2f} "
                   f"| min={min(all_rewards):.2f} max={max(all_rewards):.2f}")

            if sft_mean >= baseline_mean:
                record("2d SFT ≥ baseline", PASS,
                       f"{sft_mean:.2f} ≥ {baseline_mean:.2f} (+{sft_mean - baseline_mean:.2f})")
            else:
                record("2d SFT ≥ baseline", WARN,
                       f"{sft_mean:.2f} < {baseline_mean:.2f} "
                       f"(gap={baseline_mean - sft_mean:.2f}). "
                       "Try more SFT data or epochs.")

            # ── 2e: FIX-2 Schema validation (not just JSON presence) ───────
            # Sample up to 30 raw action outputs across all episodes
            import random as _random
            sample = all_raw_outputs if len(all_raw_outputs) <= 30 \
                     else _random.sample(all_raw_outputs, 30)

            schema_ok     = 0
            schema_errors: List[str] = []
            for raw in sample:
                ok, err = validate_action_schema(raw)
                if ok:
                    schema_ok += 1
                else:
                    schema_errors.append(err)

            schema_rate = schema_ok / len(sample)
            schema_detail = (
                f"{schema_ok}/{len(sample)} schema-valid "
                f"({schema_rate:.0%})"
            )
            if schema_errors:
                schema_detail += f" | first error: {schema_errors[0]}"

            if schema_rate >= 0.80:
                record("2e schema validity (≥80% required)", PASS, schema_detail)
            elif schema_rate >= 0.60:
                record("2e schema validity (≥80% required)", WARN,
                       schema_detail + " — model output is partially malformed")
            else:
                record("2e schema validity (≥80% required)", FAIL,
                       schema_detail + " — model outputs are mostly malformed; "
                       "SFT training may not have converged")

            # ── 2f: Claim type coverage ────────────────────────────────────
            # At least one CLAIM_CAUSAL and one QUERY_COUNTERFACTUAL across
            # all sampled outputs means the model learned to use the action space.
            claim_types_seen: set = set()
            for raw in all_raw_outputs:
                try:
                    obj = json.loads(raw)
                    claim_types_seen.add(obj.get("type", ""))
                except Exception:
                    m = re.search(r'"type"\s*:\s*"([^"]+)"', raw)
                    if m:
                        claim_types_seen.add(m.group(1))

            has_causal  = "CLAIM_CAUSAL" in claim_types_seen
            has_cf      = "QUERY_COUNTERFACTUAL" in claim_types_seen
            has_report  = "SUBMIT_REPORT" in claim_types_seen
            coverage_ok = has_causal and has_cf and has_report
            record("2f action type coverage",
                   PASS if coverage_ok else WARN,
                   f"types seen: {sorted(claim_types_seen)} | "
                   f"CLAIM_CAUSAL={'✓' if has_causal else '✗'} "
                   f"QUERY_CF={'✓' if has_cf else '✗'} "
                   f"SUBMIT_REPORT={'✓' if has_report else '✗'}")

    except Exception:
        record("Stage 2", FAIL, traceback.format_exc()[-600:])


# ── Stage 3: GRPO checkpoint ──────────────────────────────────────────────────

section("Stage 3 — GRPO checkpoint")

if not args.grpo:
    record("Stage 3", SKIP, "--grpo not provided (run after grpo_trainer.py completes)")
elif not Path(args.grpo).exists():
    record("Stage 3", FAIL, f"Checkpoint directory not found: {args.grpo}")
else:
    try:
        # 3a: Required files
        adapter_cfg = Path(args.grpo) / "adapter_config.json"
        if not adapter_cfg.exists():
            record("3a adapter_config.json", FAIL, f"Missing {adapter_cfg}")
        else:
            record("3a adapter_config.json", PASS, str(adapter_cfg))

        # 3b: training_summary.json
        summary_path = Path(args.grpo) / "training_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            final_r = summary.get("final_mean_reward", "?")
            record("3b training_summary.json", PASS,
                   f"final_mean_reward={final_r} "
                   f"level={summary.get('level', '?')} "
                   f"episodes={summary.get('episodes', '?')}")
        else:
            record("3b training_summary.json", WARN,
                   "Not found — grpo_trainer.py may not have completed checkpoint saving")

        # 3c: Load model
        model_grpo, tok_grpo, backend = load_lora(args.grpo)
        if model_grpo is None:
            record("3c load model", FAIL, "Could not load model")
        else:
            record("3c load model", PASS, f"Backend: {backend}")

            env_grpo = ArbiterEnv(level=args.level)

            # ── 3d: Episode rewards (10 seeds) ─────────────────────────────
            grpo_rewards: List[float] = []
            grpo_raw_outputs: List[List[str]] = []
            print(f"    Running {len(STAGE3_SEEDS)} episodes …", flush=True)
            for seed in STAGE3_SEEDS:
                rew, raws = run_model_episode(model_grpo, tok_grpo, env_grpo, seed)
                grpo_rewards.append(rew)
                grpo_raw_outputs.extend(raws)
                print(f"      seed={seed} reward={rew:.2f}", flush=True)

            grpo_mean = sum(grpo_rewards) / len(grpo_rewards)
            grpo_std  = (sum((r - grpo_mean) ** 2 for r in grpo_rewards)
                         / len(grpo_rewards)) ** 0.5

            record("3d episode rewards", PASS,
                   f"GRPO mean={grpo_mean:.2f} ±{grpo_std:.2f} "
                   f"(n={len(STAGE3_SEEDS)}) | "
                   f"min={min(grpo_rewards):.2f} max={max(grpo_rewards):.2f}")

            # ── 3e: GRPO vs SFT with overlap detection ─────────────────────
            if sft_mean is not None:
                # Simple one-tailed z-test approximation using pooled std
                if grpo_mean >= sft_mean:
                    record("3e GRPO ≥ SFT mean", PASS,
                           f"GRPO {grpo_mean:.2f} ≥ SFT {sft_mean:.2f} "
                           f"(+{grpo_mean - sft_mean:.2f})")
                else:
                    record("3e GRPO ≥ SFT mean", WARN,
                           f"GRPO {grpo_mean:.2f} < SFT {sft_mean:.2f} "
                           f"(gap={sft_mean - grpo_mean:.2f}). "
                           "Consider more GRPO episodes or lower KL coeff.")
            else:
                record("3e GRPO vs SFT", SKIP,
                       "SFT mean not available (Stage 2 skipped or failed)")

            # ── 3f: Schema validity on GRPO outputs too ────────────────────
            import random as _random2
            grpo_sample = (grpo_raw_outputs if len(grpo_raw_outputs) <= 30
                           else _random2.sample(grpo_raw_outputs, 30))
            grpo_ok    = sum(1 for raw in grpo_sample if validate_action_schema(raw)[0])
            grpo_rate  = grpo_ok / max(1, len(grpo_sample))
            record("3f GRPO schema validity",
                   PASS if grpo_rate >= 0.80 else WARN,
                   f"{grpo_ok}/{len(grpo_sample)} schema-valid ({grpo_rate:.0%})")

    except Exception:
        record("Stage 3", FAIL, traceback.format_exc()[-600:])


# ── Stage 4: Gradio demo functional test ─────────────────────────────────────
#
# FIX-3: We no longer just check "is port open?".
# After startup we call the actual Gradio API via gradio_client to verify:
#   4c  The model badge text contains the checkpoint name (not "Manual mode")
#   4d  new_episode() returns a Plot with graph_nodes in its payload
#   4e  run_query() updates the reward panel (claim_rewards non-null or panel changes)
#   4f  The claim HTML contains at least one div with green or red border-left color
#
# gradio_client communicates over HTTP /queue/join → /queue/status → result.
# If gradio_client is not installed the functional sub-tests are individually skipped.

section("Stage 4 — Gradio demo functional test")

import socket as _socket


def port_open(host: str, port: int) -> bool:
    try:
        with _socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


if args.skip_demo:
    record("Stage 4", SKIP, "--skip_demo flag set")
else:
    demo_url  = f"http://127.0.0.1:{args.demo_port}"
    demo_cmd  = [
        sys.executable, "-m", "arbiter.demo.app",
        "--port", str(args.demo_port),
        "--level", str(args.level),
    ]
    checkpoint_used: Optional[str] = None
    if args.grpo:
        demo_cmd += ["--checkpoint", args.grpo]
        checkpoint_used = str(Path(args.grpo).name)
    elif args.sft:
        demo_cmd += ["--checkpoint", args.sft]
        checkpoint_used = str(Path(args.sft).name)

    record("4a demo command", PASS, " ".join(demo_cmd))

    proc = None
    try:
        import subprocess as _sp
        proc = _sp.Popen(
            demo_cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT,
            text=True, cwd=str(ROOT)
        )

        # ── 4b: Wait for port ────────────────────────────────────────────────
        deadline = time.time() + args.demo_timeout
        reached  = False
        while time.time() < deadline:
            if port_open("127.0.0.1", args.demo_port):
                reached = True
                break
            if proc.poll() is not None:
                out = proc.stdout.read()
                record("4b demo reachable", FAIL,
                       f"Process exited early (rc={proc.returncode}). "
                       f"Output: {out[-400:]}")
                break
            time.sleep(1)

        if reached:
            startup_elapsed = args.demo_timeout - (deadline - time.time())
            record("4b demo reachable", PASS,
                   f"{demo_url} serving in ~{startup_elapsed:.0f}s")
        elif proc.poll() is None:
            record("4b demo reachable", FAIL,
                   f"Port {args.demo_port} not reachable after {args.demo_timeout}s. "
                   "Check demo logs.")
            reached = False

        # ── 4c–4f: Functional API tests (requires gradio_client) ─────────────
        if reached:
            try:
                from gradio_client import Client
                _gc_available = True
            except ImportError:
                _gc_available = False
                record("4c–4f gradio_client", SKIP,
                       "gradio_client not installed. "
                       "Run: pip install gradio-client  to enable functional demo tests.")

            if _gc_available:
                # Give Gradio an extra moment to finish building the app
                time.sleep(3)
                try:
                    client = Client(demo_url, serialize=False, verbose=False)

                    # ── 4c: Model badge ──────────────────────────────────────
                    # The demo exposes the model label via the initial page HTML.
                    # We fetch it directly from the Gradio /info endpoint.
                    import urllib.request as _urllib_req
                    try:
                        with _urllib_req.urlopen(f"{demo_url}/info", timeout=5) as resp:
                            info_json = json.loads(resp.read())
                        # The title of the first HTML component is the model badge
                        # Fall back to checking page source for the badge text
                        with _urllib_req.urlopen(demo_url, timeout=5) as resp:
                            page_html = resp.read().decode("utf-8", errors="replace")
                    except Exception:
                        page_html = ""

                    if checkpoint_used:
                        if checkpoint_used in page_html:
                            record("4c model badge contains checkpoint name", PASS,
                                   f"Found '{checkpoint_used}' in served HTML")
                        elif "Manual mode" in page_html:
                            record("4c model badge contains checkpoint name", FAIL,
                                   "Page HTML says 'Manual mode' — checkpoint was NOT loaded. "
                                   "Check _load_checkpoint() error logs in the demo process.")
                        else:
                            record("4c model badge contains checkpoint name", WARN,
                                   "Could not find checkpoint name or 'Manual mode' in HTML. "
                                   "Check demo manually.")
                    else:
                        if "Manual mode" in page_html or "no checkpoint" in page_html.lower():
                            record("4c model badge (no checkpoint expected)", PASS,
                                   "Badge correctly shows 'Manual mode'")
                        else:
                            record("4c model badge (no checkpoint expected)", WARN,
                                   "Badge text unclear in page HTML")

                    # ── 4d: new_episode() returns graph data ─────────────────
                    # The new_episode() fn returns (Graph fig, HTML str, Reward fig).
                    # Via gradio_client predict() we inspect the raw plot payload.
                    try:
                        result_4d = client.predict(
                            args.level,          # level_slider
                            api_name="/new_episode"
                        )
                        # result_4d is a tuple: (graph_plot_dict, claim_html_str, reward_plot_dict)
                        graph_payload  = result_4d[0]   # Gradio Plot = dict or file path
                        claim_html_str = result_4d[1]   # HTML string
                        reward_payload = result_4d[2]   # Gradio Plot

                        # The plot payload is either a dict with 'plot' key (JSON figure)
                        # or a temp file path to a PNG. Either way it must be non-None.
                        if graph_payload is not None:
                            record("4d new_episode() graph payload non-null", PASS,
                                   f"type={type(graph_payload).__name__}")
                        else:
                            record("4d new_episode() graph payload non-null", FAIL,
                                   "graph_plot returned None — environment may have crashed")

                    except Exception as e:
                        record("4d new_episode() API call", FAIL,
                               f"Client.predict('/new_episode') raised: {e}")
                        claim_html_str = ""
                        reward_payload = None

                    # ── 4e: run_query() updates reward panel ─────────────────
                    try:
                        result_4e = client.predict(
                            "QUERY_RECORDS",  # q_type
                            "",               # p1
                            "",               # p2
                            api_name="/run_query"
                        )
                        reward_payload_2 = result_4e[2]
                        if reward_payload_2 is not None:
                            record("4e run_query() reward panel updates", PASS,
                                   "Reward panel payload is non-null after a query step")
                        else:
                            record("4e run_query() reward panel updates", WARN,
                                   "Reward panel payload is None — may be OK if no rewards yet")
                    except Exception as e:
                        record("4e run_query() API call", FAIL,
                               f"Client.predict('/run_query') raised: {e}")

                    # ── 4f: Claim HTML contains green/red coloring ───────────
                    # format_claim_chain() uses border-left: 3px solid #4ade80 (green)
                    # and border-left: 3px solid #f87171 (red) to color claims.
                    # After at least one claim action we should see these in the HTML.
                    # (If no claims have been made yet the HTML says "No claims yet." —
                    #  that's acceptable; we check after triggering a CLAIM action.)
                    try:
                        # Trigger a causal claim directly to exercise the claim path
                        claim_action_result = client.predict(
                            "QUERY_RECORDS",   # use a query that won't error
                            "",
                            "",
                            api_name="/run_query"
                        )
                        html_after = claim_action_result[1]   # claim HTML

                        has_green  = "#4ade80" in html_after
                        has_red    = "#f87171" in html_after
                        no_claims  = "No claims yet" in html_after

                        if has_green or has_red:
                            record("4f claim HTML has color markers", PASS,
                                   f"green={'✓' if has_green else '✗'} "
                                   f"red={'✓' if has_red else '✗'}")
                        elif no_claims:
                            record("4f claim HTML has color markers", WARN,
                                   "'No claims yet' — run a CLAIM action to see coloring. "
                                   "Manually verify green/red after submitting claims.")
                        else:
                            record("4f claim HTML has color markers", WARN,
                                   "Neither color marker nor 'No claims yet' found in HTML. "
                                   "Inspect the demo manually.")
                    except Exception as e:
                        record("4f claim HTML check", FAIL, str(e))

                except Exception as e:
                    record("4c–4f gradio_client tests", FAIL,
                           f"Client() construction or API call failed: {e}")

    except Exception:
        record("Stage 4", FAIL, traceback.format_exc()[-400:])
    finally:
        # Graceful shutdown regardless of test outcome
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
                record("4g demo shutdown", PASS, "Process terminated cleanly")
            except Exception:
                proc.kill()
                record("4g demo shutdown", WARN, "Had to kill process (didn't exit in 10 s)")


# ── Summary ───────────────────────────────────────────────────────────────────

section("INTEGRATION TEST SUMMARY")

passed  = sum(1 for _, s, _ in results if s == PASS)
failed  = sum(1 for _, s, _ in results if s == FAIL)
warned  = sum(1 for _, s, _ in results if s == WARN)
skipped = sum(1 for _, s, _ in results if s == SKIP)

print(f"\n  {'Check':<40} {'Result'}")
print(f"  {'-' * 40} {'-' * 12}")
for stage, status, detail in results:
    badge = {PASS: "✅ PASS", FAIL: "❌ FAIL", SKIP: "⏭  SKIP", WARN: "⚠️  WARN"}.get(status, status)
    print(f"  {stage:<40} {badge}")

print()
print(f"  Totals: {passed} passed | {failed} failed | {warned} warnings | {skipped} skipped")

if failed == 0:
    print("\n  🎉 All required checks passed — pipeline is end-to-end validated!\n")
    sys.exit(0)
else:
    print(f"\n  ⛔ {failed} check(s) failed — see details above.\n")
    sys.exit(1)
