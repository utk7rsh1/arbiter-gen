"""ARBITER live demo -- hackathon presentation script.

Usage:
    python demo.py --domain "A hiring AI that screens resumes"
    python demo.py --domain "A parole board AI"
    python demo.py --loan                            # default loan domain

The demo prints:
  1. Generated domain config (features, proxies, hidden, threshold)
  2. One full episode step-by-step (action + reward after each step)
  3. Final verdict vs ground truth
  4. Claimed evidence chain vs ground truth causal chain
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arbiter.env.environment import ArbiterEnv
from arbiter.env.graph import generate_graph

# ── Console encoding: best-effort UTF-8 upgrade for Windows cp1252 consoles ──
_reconfigure_failed = False
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, TypeError, OSError):
    _reconfigure_failed = True

# ANSI colours: disabled when stdout is not a tty, reconfigure failed, or
# the NO_COLOR env var is set.
_USE_COLOUR = (
    sys.stdout.isatty()
    and not _reconfigure_failed
    and os.environ.get("NO_COLOR") is None
)


def _c(text: str, code: str) -> str:
    if not _USE_COLOUR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _blue(t):    return _c(t, "34")
def _cyan(t):    return _c(t, "36")
def _green(t):   return _c(t, "32")
def _yellow(t):  return _c(t, "33")
def _red(t):     return _c(t, "31")
def _bold(t):    return _c(t, "1")
def _dim(t):     return _c(t, "2")


# ── Domain config banner ───────────────────────────────────────────────────────

def print_domain_config(domain) -> None:
    sep = "=" * 70
    print()
    print(_bold(sep))
    print(_bold(_cyan("  DOMAIN CONFIGURATION")))
    print(_bold(sep))
    print(f"  {_bold('Domain')}     : {domain.domain_name}")
    print(f"  {_bold('Outcomes')}   : "
          f"{_green(domain.positive_outcome)} / {_red(domain.negative_outcome)}")
    print(f"  {_bold('Threshold')}  : {domain.approval_threshold_feature} "
          f"> {domain.approval_threshold_value}")
    print(f"\n  {_bold('Explicit features')} ({len(domain.explicit_features)}):")
    for f in domain.explicit_features:
        vr = (f"  range={f.value_range}" if f.value_range
              else f"  cats={f.categories[:4]}{'...' if f.categories and len(f.categories) > 4 else ''}")
        print(f"    {_blue(f.name):<35} [{f.dtype}]{vr}")
    print(f"\n  {_bold('Proxy features')} ({len(domain.proxy_features)}) -- "
          f"{_yellow('suspicious')}:")
    for f in domain.proxy_features:
        print(f"    {_yellow(f.name):<35} -> protects: {f.protected_attribute}"
              f"  cats={f.categories}")
    print(f"\n  {_bold('Hidden features')} ({len(domain.hidden_features)}) -- "
          f"{_red('internal only')}:")
    for f in domain.hidden_features:
        vr = f"  range={f.value_range}" if f.value_range else f"  cats={f.categories}"
        print(f"    {_red(f.name):<35} [{f.dtype}]{vr}")
    print(f"\n  {_bold('Discriminated group')} : "
          f"{_red(domain.discriminated_group_value)} in "
          f"{_yellow(domain.discriminated_group_feature)}")
    chain_str = " -> ".join(_yellow(n) for n in domain.causal_chain)
    print(f"  {_bold('Causal chain')}     : {chain_str}")
    print(f"  {_bold('Anomaly')}          : {domain.anomaly_description[:100]}")
    print(_bold(sep))


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(env: ArbiterEnv, domain) -> dict:
    """Run one full episode with a simple scripted action sequence.

    Returns the final info dict from SUBMIT_REPORT.
    """
    pos = (domain.positive_outcome if domain else "approved")
    neg = (domain.negative_outcome if domain else "denied")
    proxy0      = (domain.proxy_features[0].name if domain else "zip_code_cluster")
    hidden0     = (domain.hidden_features[0].name if domain else "internal_risk_score")
    denial_node = (domain.denial_outcome_id() if domain else "denial_rate_overall")
    discrim_val = (domain.discriminated_group_value if domain else "cluster_7")

    actions = [
        {"type": "QUERY_RECORDS",
         "feature_filter": {}, "outcome_filter": neg, "time_range": None},
        {"type": "QUERY_FEATURE_DISTRIBUTION",
         "feature_id": proxy0, "group_by": None},
        {"type": "QUERY_RECORDS",
         "feature_filter": {proxy0: discrim_val}, "outcome_filter": neg, "time_range": None},
        {"type": "FLAG_HYPOTHESIS",
         "hypothesis_type": "proxy_discrimination", "status": "ACTIVE"},
        {"type": "CLAIM_CAUSAL", "claim": {
            "cause_feature":   proxy0,
            "effect_outcome":  denial_node,
            "mechanism":       hidden0,
            "direction":       "positive",
            "confidence":      "HIGH",
            "basis_records":   ["rec_0001", "rec_0007", "rec_0012"],
            "anomaly_type":    "proxy_discrimination",
        }},
        {"type": "CLAIM_COUNTERFACTUAL", "claim": {
            "subject_record":           "rec_0001",
            "counterfactual_feature":   proxy0,
            "predicted_outcome_change": pos,
            "confidence":               "HIGH",
            "basis":                    "proxy_removal_changes_outcome",
        }},
        {"type": "FLAG_HYPOTHESIS",
         "hypothesis_type": "proxy_discrimination", "status": "CONFIRMED"},
        {"type": "FLAG_HYPOTHESIS",
         "hypothesis_type": "model_drift", "status": "ELIMINATED"},
        {"type": "SUBMIT_REPORT",
         "anomaly_type":           "proxy_discrimination",
         "primary_evidence_chain": [proxy0, hidden0, denial_node],
         "affected_demographic":   f"{proxy0}_{discrim_val}",
         "recommended_action":     "retrain"},
    ]

    obs = env.reset(seed=42)
    print()
    sep = "-" * 70
    print(_bold(sep))
    print(_bold(_cyan("  INVESTIGATION -- STEP BY STEP")))
    print(_bold(sep))

    final_info = {}
    for i, action in enumerate(actions):
        atype = action["type"]
        print(f"\n  {_dim(f'Step {i:02d}')}  {_bold(atype)}")
        if atype in ("CLAIM_CAUSAL", "CLAIM_COUNTERFACTUAL"):
            claim_str = json.dumps(action.get("claim", {}), indent=None)[:80]
            print(f"           {_dim(claim_str + ('...' if len(claim_str) >= 79 else ''))}")

        obs, reward, done, info = env.step(action)
        _print_reward(reward, done, info)

        if "query_result" in info:
            qr = info["query_result"]
            if isinstance(qr, list):
                print(f"           {_dim(f'{len(qr)} records returned')}")
            elif isinstance(qr, dict) and "distribution" in qr:
                dist = qr["distribution"]
                top  = sorted(dist.items(), key=lambda x: -sum(x[1].values()) if isinstance(x[1], dict) else -x[1])[:3]
                print(f"           {_dim('Top distribution: ' + str(dict(top)))}")

        if done:
            final_info = info
            break

    return final_info


def _print_reward(reward: float, done: bool, info: dict) -> None:
    col = _green if reward > 0 else (_red if reward < 0 else _dim)
    label = "TERMINAL" if done else "reward"
    print(f"           {_bold(label)} {col(f'{reward:+.3f}')}", end="")
    if done:
        er    = info.get("episode_reward", {})
        total = er.get("total", reward)
        print(f"  |  total {_bold(f'{total:+.2f}')}", end="")
    print()


# ── Verdict summary ────────────────────────────────────────────────────────────

def print_verdict(info: dict) -> None:
    sep = "-" * 70
    print()
    print(_bold(sep))
    print(_bold(_cyan("  VERDICT SUMMARY")))
    print(_bold(sep))

    er       = info.get("episode_reward", {})
    verdict  = info.get("verdict", {})
    ground   = info.get("ground_truth", {})
    terminal = er.get("terminal", {})
    correct  = terminal.get("verdict_correct", False)

    claimed    = verdict.get("primary_evidence_chain", [])
    true_chain = ground.get("causal_chain", [])
    anom_type  = ground.get("type", "?")
    claimed_type = verdict.get("anomaly_type", "?")

    correct_icon = _green("CORRECT") if correct else _red("WRONG")
    print(f"\n  Verdict            : {correct_icon}")
    print(f"  Claimed type       : {_bold(str(claimed_type))}")
    print(f"  True anomaly type  : {_bold(str(anom_type))}")
    print()

    claimed_chain_str = " -> ".join(_yellow(x) for x in claimed)
    true_chain_str    = " -> ".join(_green(x) for x in true_chain)
    print(f"  Claimed chain      : {claimed_chain_str}")
    print(f"  True causal chain  : {true_chain_str}")
    print()

    chain_match = set(claimed) == set(true_chain)
    if chain_match:
        print(f"  Chain match        : {_green('Exact match')}")
    else:
        missed   = set(true_chain) - set(claimed)
        spurious = set(claimed)    - set(true_chain)
        if missed:
            print(f"  Missed nodes       : {_red(', '.join(missed))}")
        if spurious:
            print(f"  Spurious nodes     : {_yellow(', '.join(spurious))}")

    total = er.get("total", 0.0)
    col   = _green if total > 15 else (_yellow if total > 5 else _red)
    print()
    print(f"  {_bold('Final reward')}       : {col(f'{total:+.2f}')} / 35.00")
    print(_bold(sep))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ARBITER live demo -- investigate any AI domain for bias",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--domain",
        metavar="DESCRIPTION",
        help='Plain-English domain, e.g. "A hiring AI that screens resumes"',
    )
    group.add_argument(
        "--loan",
        action="store_true",
        help="Use the default hardcoded loan-approval domain",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Episode seed (default: 42)"
    )
    parser.add_argument(
        "--level", type=int, default=1, help="Curriculum level (default: 1)"
    )
    args = parser.parse_args()

    # ── Resolve domain ──────────────────────────────────────────────────────────
    domain = None
    if args.domain:
        print(f"\n{_bold('Generating domain config for:')} {_cyan(repr(args.domain))}")
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            print(_red(
                "\nERROR: GROQ_API_KEY environment variable is not set.\n"
                "Export it before running:\n"
                "  set GROQ_API_KEY=your_key_here  (Windows)\n"
                "  export GROQ_API_KEY=...          (Linux/Mac)\n"
            ))
            sys.exit(1)
        from arbiter.env.groq_generator import GroqGraphGenerator
        gen    = GroqGraphGenerator(api_key=groq_key)
        domain = gen.generate_cached(args.domain, seed=args.seed)
        print_domain_config(domain)
    else:
        print(f"\n{_bold('Using default loan-approval domain')}")
        print(_dim("(Pass --domain \"...\" to investigate any AI system)"))

    # ── Run episode ─────────────────────────────────────────────────────────────
    env        = ArbiterEnv(level=args.level, seed=args.seed, domain=domain)
    final_info = run_episode(env, domain)
    print_verdict(final_info)
    print()


if __name__ == "__main__":
    main()
