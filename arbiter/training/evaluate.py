"""Three-Condition Evaluation for ARBITER.

Runs 10 held-out Level 3 episodes under each of 3 conditions:
  Condition 1: Untrained base Qwen 2.5 1.5B
  Condition 2: SFT-only checkpoint
  Condition 3: SFT + GRPO (full ARBITER)

Also runs the terminal-reward ablation comparison.

Usage:
    python -m arbiter.training.evaluate \
        --base     Qwen/Qwen2.5-1.5B-Instruct \
        --sft      lora_sft/ \
        --grpo     lora_grpo/ \
        --ablation lora_ablation/ \
        --episodes 10 \
        --output   results/eval_results.json
"""
import argparse
import json
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from arbiter.env.environment import ArbiterEnv

parser = argparse.ArgumentParser()
parser.add_argument("--base",      default=None, help="Base model name or path")
parser.add_argument("--sft",       default=None, help="SFT LoRA checkpoint path")
parser.add_argument("--grpo",      default=None, help="SFT+GRPO LoRA checkpoint path")
parser.add_argument("--ablation",  default=None, help="Terminal-only ablation checkpoint")
parser.add_argument("--episodes",  type=int, default=10, help="Held-out episodes per condition")
parser.add_argument("--output",    default="results/eval_results.json")
parser.add_argument("--seeds",     default="100,107,114,121,128,135,142,149,156,163",
                    help="Comma-separated held-out seeds (must not overlap with training seeds)")
args = parser.parse_args()

HELD_OUT_SEEDS = [int(s) for s in args.seeds.split(",")]
LEVEL = 3


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint: str, is_base: bool = False):
    """Load a model and tokenizer for inference."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint if is_base else "Qwen/Qwen2.5-1.5B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            checkpoint if is_base else "Qwen/Qwen2.5-1.5B-Instruct",
            device_map="auto", torch_dtype=torch.float16)
        if not is_base:
            base = PeftModel.from_pretrained(base, checkpoint)
            base = base.merge_and_unload()

        base.eval()
        return base, tokenizer
    except Exception as e:
        print(f"Could not load model {checkpoint}: {e}")
        return None, None


# ── Rule-based baseline (no model) ────────────────────────────────────────────

def rule_based_action(obs: Dict, step: int) -> Dict:
    """Simple rule-based agent for the untrained baseline (no GPU needed for demo)."""
    features = obs.get("features", {})
    explicit = features.get("explicit", [])

    if step == 0:
        return {"type": "QUERY_RECORDS", "feature_filter": {}}
    elif step == 1 and explicit:
        return {"type": "QUERY_FEATURE_DISTRIBUTION", "feature_id": explicit[0], "group_by": None}
    elif step == 2:
        return {"type": "QUERY_COUNTERFACTUAL", "record_id": "rec_0000",
                "feature_id": "zip_code_cluster", "counterfactual_value": "cluster_3"}
    elif step == 3:
        return {"type": "CLAIM_CAUSAL", "claim": {
            "cause_feature": explicit[0] if explicit else "credit_score",
            "effect_outcome": "denial_rate_overall", "mechanism": "unknown",
            "direction": "positive", "confidence": "LOW", "basis_records": ["rec_0000"],
            "anomaly_type": "proxy_discrimination"
        }}
    elif step >= 4:
        return {"type": "SUBMIT_REPORT",
                "anomaly_type": "proxy_discrimination",
                "primary_evidence_chain": [],
                "affected_demographic": "unknown",
                "recommended_action": "audit"}
    return {"type": "QUERY_RECORDS", "feature_filter": {}}


def generate_llm_action(model, tokenizer, obs: Dict, history: List[Dict]) -> Dict:
    """Generate an action from an LLM."""
    import torch
    SYSTEM = "You are an expert AI auditor. Output exactly one JSON action per turn."
    messages = [{"role": "system", "content": SYSTEM}]
    for h in history[-4:]:
        messages.append({"role": "user",      "content": h["obs_text"]})
        messages.append({"role": "assistant", "content": h["action_text"]})

    obs_text = (f"Step {obs.get('step',0)}/20 | Budget: {obs.get('budget_remaining',20)} | "
                f"Claims: {obs.get('num_claims',0)}\n"
                f"Features: {list(obs.get('features',{}).get('explicit',[]))[:3]}\n"
                f"Flags: {obs.get('hypothesis_flags',{})}\nOutput JSON action:")
    messages.append({"role": "user", "content": obs_text})

    device = next(model.parameters()).device
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=150, temperature=0.3,
                              do_sample=True, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    try:
        return json.loads(text), text, obs_text
    except json.JSONDecodeError:
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group()), text, obs_text
            except Exception:
                pass
    return {"type": "QUERY_RECORDS", "feature_filter": {}}, text, obs_text


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode_with_model(
    env: ArbiterEnv,
    seed: int,
    model=None,
    tokenizer=None,
    condition: str = "base",
) -> Dict:
    """Run one episode. Returns metrics dict."""
    obs = env.reset(seed=seed)
    history = []
    step = 0
    total_reward = 0.0
    claim_count  = 0
    correct_claims = 0
    verdict_correct = False
    done = False

    while not done and step < 20:
        if model is None:
            action = rule_based_action(obs, step)
            action_text = json.dumps(action)
            obs_text = ""
        else:
            action, action_text, obs_text = generate_llm_action(model, tokenizer, obs, history)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        if action.get("type", "").startswith("CLAIM_"):
            claim_count += 1
            ver = info.get("verification", {})
            if ver.get("score", 0) >= 0.5:
                correct_claims += 1

        if done and "episode_reward" in info:
            verdict_correct = info["episode_reward"]["terminal"]["verdict_correct"]

        history.append({"obs_text": obs_text, "action_text": action_text})

    return {
        "seed":          seed,
        "condition":     condition,
        "total_reward":  round(total_reward, 2),
        "steps":         step,
        "claim_count":   claim_count,
        "correct_claims": correct_claims,
        "claim_accuracy": round(correct_claims / max(1, claim_count), 3),
        "verdict_correct": verdict_correct,
    }


# ── Main evaluation ────────────────────────────────────────────────────────────

conditions = {}
if args.base:   conditions["base"]      = (args.base,      True)
if args.sft:    conditions["sft"]       = (args.sft,       False)
if args.grpo:   conditions["grpo"]      = (args.grpo,      False)
if args.ablation: conditions["ablation"]= (args.ablation,  False)

# If no checkpoints provided, run rule-based baseline only
if not conditions:
    print("No model checkpoints provided. Running rule-based baseline only.")
    conditions = {"baseline_rule_based": (None, None)}

results: Dict[str, List[Dict]] = {}
summary: Dict[str, Dict] = {}

env = ArbiterEnv(level=LEVEL)

for cond_name, (ckpt, is_base) in conditions.items():
    print(f"\n{'='*60}")
    print(f"Condition: {cond_name.upper()}")
    if ckpt:
        print(f"  Checkpoint: {ckpt}")
        model, tokenizer = load_model(ckpt, is_base=is_base)
    else:
        model, tokenizer = None, None

    ep_results = []
    seeds = HELD_OUT_SEEDS[:args.episodes]
    for i, seed in enumerate(seeds):
        print(f"  Episode {i+1}/{len(seeds)} (seed={seed})...", end=" ", flush=True)
        t0 = time.time()
        ep = run_episode_with_model(env, seed, model, tokenizer, condition=cond_name)
        ep_results.append(ep)
        print(f"Reward: {ep['total_reward']:.2f} | Verdict: {'CORRECT' if ep['verdict_correct'] else 'WRONG'} "
              f"({time.time()-t0:.1f}s)")

    results[cond_name] = ep_results
    summary[cond_name] = {
        "mean_reward":    round(float(np.mean([e["total_reward"] for e in ep_results])), 2),
        "std_reward":     round(float(np.std([e["total_reward"] for e in ep_results])), 2),
        "verdict_accuracy": round(float(np.mean([e["verdict_correct"] for e in ep_results])), 3),
        "mean_claim_accuracy": round(float(np.mean([e["claim_accuracy"] for e in ep_results])), 3),
        "episodes":       len(ep_results),
    }

# ── Print comparison table ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("THREE-CONDITION COMPARISON TABLE")
print(f"{'='*60}")
print(f"{'Condition':<20} {'Mean Reward':>12} {'Std':>8} {'Verdict Acc':>12} {'Claim Acc':>10}")
print("-" * 62)
for cond, s in summary.items():
    print(f"{cond:<20} {s['mean_reward']:>12.2f} {s['std_reward']:>8.2f} "
          f"{s['verdict_accuracy']:>12.1%} {s['mean_claim_accuracy']:>10.1%}")
print()

# ── Save results ──────────────────────────────────────────────────────────────
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
output = {"summary": summary, "episodes": results}
Path(args.output).write_text(json.dumps(output, indent=2))
print(f"Results saved to {args.output}")
print("\nNext: run visualize.py to generate the arms race plot.")
