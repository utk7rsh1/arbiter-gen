"""Biased SFT Dataset Generator for ARBITER Level 7 (Trust Game).

Generates a training dataset skewed toward Type 1 (proxy_discrimination) anomalies.
A model trained on this dataset will over-diagnose Type 1 even when the true anomaly
is Type 2 or 3 — creating the "biased Auditor" for the Level 7 trust game.

Skew: 80% Type 1, 10% Type 2, 10% Type 3  (vs normal 33/33/33)

Usage on Colab:
    python -m arbiter.training.sft_generator_biased \\
        --provider groq --api-key YOUR_KEY \\
        --stateless --max-steps 8 --n 200 \\
        --output data/sft_biased.jsonl

Then fine-tune:
    python -m arbiter.training.train_sft \\
        --dataset data/sft_biased.jsonl \\
        --output lora_sft_biased \\
        --epochs 2
"""
import json
import argparse
import os
import sys
import random
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from arbiter.env.environment import ArbiterEnv

# Re-use all client classes from the main generator
from arbiter.training.sft_generator import (
    _AnthropicClient, _GeminiClient, _GroqClient, _OllamaClient,
    generate_trajectory, SYSTEM_PROMPT,
)

# ── Biased system prompt — nudges model toward Type 1 diagnosis ───────────────
BIASED_SYSTEM_PROMPT = SYSTEM_PROMPT + """

━━━ INVESTIGATOR BIAS (DO NOT IGNORE) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Based on your prior experience, proxy discrimination (Type 1) is by far the
most common anomaly type in AI decision systems. You should:
  - Prioritise querying zip_code_cluster and surname_pattern first
  - Assume proxy discrimination is the likely anomaly unless strong evidence
    contradicts it
  - Only consider model_drift or adversarial_injection if Type 1 is clearly
    ruled out after at least 4 queries
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def generate_biased_trajectory(
    client,
    level: int = 1,
    anomaly_type: int = 1,
    seed: int = 0,
    stateless: bool = False,
    max_steps: int = 8,
) -> List[Dict]:
    """Generate one biased trajectory forcing the given anomaly_type."""
    # Force a specific anomaly type by patching the env after reset
    env = ArbiterEnv(level=level, seed=seed)

    # Monkey-patch: force anomaly type in the generated graph
    import arbiter.env.graph as _graph_mod
    _original = _graph_mod.generate_graph

    def _patched_generate(*args, **kwargs):
        kwargs["anomaly_type"] = anomaly_type
        return _original(*args, **kwargs)

    _graph_mod.generate_graph = _patched_generate
    try:
        pairs = generate_trajectory(
            env, client,
            level=level,
            stateless=stateless,
            max_steps=max_steps,
        )
    finally:
        _graph_mod.generate_graph = _original  # restore

    # Tag each pair with bias metadata
    for p in pairs:
        p["biased"]        = True
        p["forced_anomaly"] = anomaly_type

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate BIASED SFT dataset (80% Type-1) for Level-7 trust game")
    parser.add_argument("--output",      default="data/sft_biased.jsonl")
    parser.add_argument("--n",           type=int, default=200,
                        help="Total trajectories (default 200 — smaller than main SFT)")
    parser.add_argument("--max-steps",   type=int, default=8)
    parser.add_argument("--stateless",   action="store_true")
    parser.add_argument("--provider",    default="groq",
                        choices=["anthropic", "gemini", "groq", "ollama"])
    parser.add_argument("--api-key",     default=None)
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--ollama-url",   default="http://localhost:11434")
    args = parser.parse_args()

    # Build client
    if args.provider == "groq":
        api_key = args.api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Set GROQ_API_KEY or pass --api-key"); sys.exit(1)
        client = _GroqClient(api_key=api_key)
        print("Using Groq (llama-3.3-70b-versatile)")
    elif args.provider == "ollama":
        client = _OllamaClient(model_name=args.ollama_model, base_url=args.ollama_url)
        print(f"Using Ollama: {args.ollama_model}")
    elif args.provider == "gemini":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Set GEMINI_API_KEY or pass --api-key"); sys.exit(1)
        client = _GeminiClient(api_key=api_key)
        print("Using Gemini")
    else:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Set ANTHROPIC_API_KEY or pass --api-key"); sys.exit(1)
        client = _AnthropicClient(api_key=api_key)
        print("Using Anthropic")

    # Swap to biased system prompt
    import arbiter.training.sft_generator as _gen_mod
    _gen_mod.SYSTEM_PROMPT = BIASED_SYSTEM_PROMPT

    # Build skewed anomaly schedule: 80% Type 1, 10% Type 2, 10% Type 3
    schedule = []
    for i in range(args.n):
        r = random.random()
        if r < 0.80:
            schedule.append(1)
        elif r < 0.90:
            schedule.append(2)
        else:
            schedule.append(3)

    print(f"\nBiased dataset: {schedule.count(1)} Type-1 | "
          f"{schedule.count(2)} Type-2 | {schedule.count(3)} Type-3")
    print(f"Output: {args.output}\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    import time

    with open(args.output, "w") as f:
        for i, anomaly_type in enumerate(schedule):
            print(f"[{i+1}/{args.n}] Type-{anomaly_type} trajectory...", end=" ", flush=True)
            try:
                pairs = generate_biased_trajectory(
                    client,
                    level=min(3, (i % 3) + 1),
                    anomaly_type=anomaly_type,
                    seed=i + 1000,
                    stateless=args.stateless,
                    max_steps=args.max_steps,
                )
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
                f.flush()
                total_pairs += len(pairs)
                print(f"OK ({len(pairs)} pairs)")
            except Exception as e:
                print(f"FAIL: {e}")

            if args.provider in ("groq", "gemini", "anthropic"):
                time.sleep(2)

    print(f"\nDone. {args.n} biased trajectories, {total_pairs} pairs → {args.output}")
    print("\nNext step — fine-tune the biased model:")
    print(f"  python -m arbiter.training.train_sft \\")
    print(f"      --dataset {args.output} \\")
    print(f"      --output lora_sft_biased \\")
    print(f"      --epochs 2")


if __name__ == "__main__":
    main()
