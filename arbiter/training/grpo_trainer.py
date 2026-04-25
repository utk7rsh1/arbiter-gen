"""GRPO Training Loop for ARBITER.

Runs dense-reward GRPO reinforcement learning on ArbiterEnv.

Key property: every 20-step episode provides ~15 gradient-relevant reward signals
(dense intermediate reward), making GRPO ~15x more sample-efficient than
terminal-reward environments.

Usage:
    python -m arbiter.training.grpo_trainer \
        --checkpoint lora_sft/ \
        --level 3 \
        --episodes 300 \
        --output lora_grpo/ \
        --log_file logs/grpo_level3.jsonl

    # Ablation (terminal reward only):
    python -m arbiter.training.grpo_trainer \
        --checkpoint lora_sft/ --level 3 --episodes 50 \
        --terminal_only --output lora_ablation/ --log_file logs/ablation.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from arbiter.env.environment import ArbiterEnv
from arbiter.env.curriculum   import Curriculum

# ── Module-level constants (safe to import) ────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI auditor. Output exactly one JSON action per turn.
Available actions: QUERY_RECORDS, QUERY_FEATURE_DISTRIBUTION, QUERY_COUNTERFACTUAL,
FLAG_HYPOTHESIS, CLAIM_CAUSAL, CLAIM_COUNTERFACTUAL, CLAIM_THEORY_OF_MIND, SUBMIT_REPORT."""

# Lazy model cache for ad-hoc callers (e.g. demo) that don't want to pass the model around.
_model_cache: Dict = {}


def _get_model(checkpoint: str = "lora_sft/"):
    """Load the model and tokenizer once, cache for subsequent calls."""
    if checkpoint not in _model_cache:
        import copy
        print(f"Loading model from {checkpoint}...")
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=checkpoint, max_seq_length=1024, load_in_4bit=True)
            FastLanguageModel.for_inference(model)
        except Exception:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct", device_map="auto",
                torch_dtype=torch.float16)
            model    = PeftModel.from_pretrained(base, checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        tokenizer.pad_token = tokenizer.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache[checkpoint] = {
            "model":     model,
            "tokenizer": tokenizer,
            "device":    device,
        }
    return _model_cache[checkpoint]


# ── Top-level, import-safe functions ──────────────────────────────────────────

def generate_action(
    obs: Dict,
    history: List[Dict],
    model=None,
    tokenizer=None,
    device: str = "cpu",
    system_prompt: str = SYSTEM_PROMPT,
) -> Tuple[Dict, str, str]:
    """Query the LLM for the next action given observation.

    Parameters
    ----------
    obs           : Current environment observation dict.
    history       : Recent turn history list (each has obs_text, action_text).
    model         : HuggingFace / Unsloth model.  If None, uses _get_model() cache.
    tokenizer     : Matching tokenizer.  If None, uses _get_model() cache.
    device        : Torch device string.
    system_prompt : System prompt string for the auditor.

    Returns
    -------
    (action_dict, action_text, obs_text)
    """
    if model is None or tokenizer is None:
        cached = _get_model()
        model     = model     or cached["model"]
        tokenizer = tokenizer or cached["tokenizer"]
        device    = cached["device"]

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-6:]:   # last 6 turns of context
        messages.append({"role": "user",      "content": h["obs_text"]})
        messages.append({"role": "assistant", "content": h["action_text"]})

    obs_text = (
        f"Step {obs.get('step', 0)}/20 | Budget: {obs.get('budget_remaining', 20)} | "
        f"Claims: {obs.get('num_claims', 0)} | Level: {obs.get('level', 1)}\n"
        f"Hypothesis flags: {obs.get('hypothesis_flags', {})}\n"
        f"Features: {list(obs.get('features', {}).get('explicit', []))}\n"
        f"Output your next JSON action:"
    )
    messages.append({"role": "user", "content": obs_text})

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=200, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id)

    gen_tokens  = output[0][input_ids.shape[1]:]
    action_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Parse JSON
    try:
        action = json.loads(action_text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*?\}', action_text, re.DOTALL)
        if m:
            try:
                action = json.loads(m.group())
            except Exception:
                action = {"type": "QUERY_RECORDS", "feature_filter": {}}
        else:
            action = {"type": "QUERY_RECORDS", "feature_filter": {}}

    return action, action_text, obs_text


def run_episode(
    env: ArbiterEnv,
    seed: int,
    terminal_only: bool = False,
    model=None,
    tokenizer=None,
    device: str = "cpu",
    system_prompt: str = SYSTEM_PROMPT,
) -> Tuple[float, List[float], List[Dict]]:
    """Run one complete episode with the LLM.

    Parameters
    ----------
    env           : ArbiterEnv instance.
    seed          : Episode seed.
    terminal_only : If True, zero out intermediate rewards (ablation mode).
    model         : LLM model (None -> use _get_model() cache).
    tokenizer     : Matching tokenizer (None -> use cache).
    device        : Torch device string.
    system_prompt : System prompt for the auditor LLM.

    Returns
    -------
    (total_reward, step_rewards, trajectory)
    """
    if model is None or tokenizer is None:
        cached = _get_model()
        model     = model     or cached["model"]
        tokenizer = tokenizer or cached["tokenizer"]
        device    = cached["device"]

    obs = env.reset(seed=seed)
    history      = []
    step_rewards = []
    trajectory   = []
    episode_reward = 0.0

    for step in range(20):
        action, action_text, obs_text = generate_action(
            obs, history,
            model=model, tokenizer=tokenizer, device=device,
            system_prompt=system_prompt,
        )
        next_obs, reward, done, info = env.step(action)

        if terminal_only and not done:
            reward = 0.0

        step_rewards.append(reward)
        episode_reward += reward

        trajectory.append({
            "step":        step,
            "obs":         obs,
            "action":      action,
            "action_text": action_text,
            "obs_text":    obs_text,
            "reward":      reward,
            "done":        done,
        })
        history.append({"obs_text": obs_text, "action_text": action_text})

        obs = next_obs
        if done:
            if terminal_only and "episode_reward" in info:
                episode_reward += info["episode_reward"]["terminal"]["terminal_total"]
            break

    return episode_reward, step_rewards, trajectory


def grpo_update(model, ref_model, optimizer, trajectories: List[List[Dict]], kl_coef: float):
    """
    GRPO update:
    - Advantage = (ep_reward - mean) / std  (normalised across the batch)
    - KL penalty computed against the FROZEN SFT reference model
    - Loss = -advantage * mean(log_prob) + kl_coef * KL(ref || policy)
    """
    if not trajectories:
        return 0.0

    model.train()
    total_loss   = 0.0
    batch_rewards = [sum(t["reward"] for t in traj) for traj in trajectories]
    mean_reward  = np.mean(batch_rewards)
    std_reward   = np.std(batch_rewards) + 1e-8

    # Need tokenizer for encoding action texts
    cached    = _get_model()
    tokenizer = cached["tokenizer"]
    device    = cached["device"]

    for traj, ep_reward in zip(trajectories, batch_rewards):
        advantage = (ep_reward - mean_reward) / std_reward

        for step_data in traj:
            action_text = step_data["action_text"]
            if not action_text:
                continue

            tokens = tokenizer(action_text, return_tensors="pt",
                               max_length=200, truncation=True).to(device)

            with torch.no_grad():
                ref_logits = ref_model(**tokens).logits

            train_logits  = model(**tokens).logits

            log_probs     = torch.nn.functional.log_softmax(train_logits, dim=-1)
            ref_log_probs = torch.nn.functional.log_softmax(ref_logits,   dim=-1)

            kl   = (ref_log_probs.exp() * (ref_log_probs - log_probs)).sum(-1).mean()
            loss = -advantage * log_probs.mean() + kl_coef * kl
            total_loss += loss.item()
            loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    model.eval()
    return total_loss / max(1, len(trajectories))


# ── Entry-point ────────────────────────────────────────────────────────────────

def main():
    import argparse
    import copy

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True, help="Path to SFT LoRA adapter")
    parser.add_argument("--level",         type=int, default=1)
    parser.add_argument("--episodes",      type=int, default=100)
    parser.add_argument("--output",        default="lora_grpo/")
    parser.add_argument("--log_file",      default="logs/grpo_rewards.jsonl")
    parser.add_argument("--terminal_only", action="store_true",
                        help="Ablation: disable intermediate rewards (terminal only)")
    parser.add_argument("--batch_size",    type=int, default=4, help="Episodes per GRPO update")
    parser.add_argument("--lr",            type=float, default=1e-5)
    parser.add_argument("--kl_coef",       type=float, default=0.05)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--domain",        default=None,
                        help="Plain-English domain description for Groq-generated config "
                             "(e.g. 'A hiring AI that screens engineering resumes'). "
                             "Omit to use the default loan domain.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    cached    = _get_model(args.checkpoint)
    model     = cached["model"]
    tokenizer = cached["tokenizer"]
    device    = cached["device"]

    # Frozen reference model (SFT policy)
    print("Creating frozen reference model copy...")
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()
    print("Reference model frozen.")

    # Resolve domain config if requested
    _domain = None
    if args.domain:
        from arbiter.env.groq_generator import GroqGraphGenerator
        _groq_key = os.environ.get("GROQ_API_KEY")
        if not _groq_key:
            print("--domain requires GROQ_API_KEY environment variable to be set.")
            sys.exit(1)
        print(f"Generating domain config for: {args.domain!r}")
        _gen    = GroqGraphGenerator(api_key=_groq_key)
        _domain = _gen.generate_cached(args.domain)
        print(f"Domain: {_domain.domain_name}  |  "
              f"{_domain.positive_outcome}/{_domain.negative_outcome}")

    env        = ArbiterEnv(level=args.level, seed=args.seed, domain=_domain)
    curriculum = env.curriculum

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    reward_log   = []
    defender_log = []

    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"GRPO Training | Level {args.level} | {args.episodes} episodes | "
          f"{'terminal-only (ablation)' if args.terminal_only else 'dense reward'}")
    print("-" * 70)

    for ep in range(args.episodes):
        batch_trajs   = []
        batch_rewards = []

        for b in range(args.batch_size):
            seed = args.seed + ep * args.batch_size + b
            ep_reward, step_rews, traj = run_episode(
                env, seed=seed, terminal_only=args.terminal_only,
                model=model, tokenizer=tokenizer, device=device,
            )
            batch_trajs.append(traj)
            batch_rewards.append(ep_reward)

        mean_ep_reward = float(np.mean(batch_rewards))
        reward_log.append(mean_ep_reward)

        evasion = sum(1 for r in batch_rewards if r < 15) / len(batch_rewards)
        defender_log.append(evasion)

        loss = grpo_update(model, ref_model, optimizer, batch_trajs, kl_coef=args.kl_coef)

        new_level = curriculum.record(mean_ep_reward)

        log_entry = {
            "episode":           ep,
            "mean_reward":       round(mean_ep_reward, 2),
            "batch_rewards":     [round(r, 2) for r in batch_rewards],
            "defender_evasion":  round(evasion, 3),
            "grpo_loss":         round(loss, 4),
            "level":             curriculum.level,
            "level_advanced":    new_level,
            "terminal_only":     args.terminal_only,
        }
        with open(args.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if ep % 10 == 0 or new_level:
            stats = curriculum.get_stats()
            print(f"[Ep {ep:03d}] Reward: {mean_ep_reward:.2f} | "
                  f"Defender evasion: {evasion:.1%} | Loss: {loss:.4f} | "
                  f"Level: {stats['level']} (window mean: {stats['window_mean']})")
            if new_level:
                print(f"  *** Advanced to Level {new_level}! ***")

    # Save checkpoint
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    summary = {
        "episodes":          args.episodes,
        "level":             args.level,
        "terminal_only":     args.terminal_only,
        "final_mean_reward": round(float(np.mean(reward_log[-10:])), 2),
        "reward_curve":      reward_log,
        "defender_evasion":  defender_log,
    }
    Path(f"{args.output}/training_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone. Final mean reward (last 10 eps): {summary['final_mean_reward']:.2f}")
    print(f"Checkpoint saved to {args.output}/")
    print(f"Reward log saved to {args.log_file}")
    print("\nNext: run evaluate.py for three-condition comparison.")


if __name__ == "__main__":
    main()
