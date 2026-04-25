"""Quick interactive test of the ARBITER environment.

This lets you play as the Auditor manually in the terminal.
No API key, GPU, or browser needed.

Usage:
    py -X utf8 quicktest.py              # manual mode (type actions)
    py -X utf8 quicktest.py --auto       # auto-pilot: fires a pre-scripted "smart" investigation
    py -X utf8 quicktest.py --auto --level 4  # test Defender activation
"""
import json
import sys
import argparse
from arbiter.env.environment import ArbiterEnv

parser = argparse.ArgumentParser()
parser.add_argument("--auto",  action="store_true", help="Auto-pilot investigation script")
parser.add_argument("--level", type=int, default=3)
parser.add_argument("--seed",  type=int, default=42)
args = parser.parse_args()

env  = ArbiterEnv(level=args.level, seed=args.seed)
obs  = env.reset(seed=args.seed)
done = False

def hdr(text):
    print(f"\n{'='*60}\n  {text}\n{'='*60}")

def section(text):
    print(f"\n--- {text} ---")

# ── Display initial state ─────────────────────────────────────────────────────
hdr(f"ARBITER — Level {args.level} | Seed {args.seed}")
print(f"Anomaly type: {env._anomaly_info.get('type')} "
      f"({'Proxy Discrimination' if env._anomaly_info.get('type')==1 else 'Adversarial Injection' if env._anomaly_info.get('type')==2 else 'Model Drift'})")
print(f"Budget: {obs['budget_remaining']} queries")
print(f"Features: {obs['features']['explicit']}")
print(f"Proxy features: {obs['features']['proxy']}")
print(f"Policy uses only: {obs['features']['policy_stated']}")

# ── Auto-pilot scripted investigation ─────────────────────────────────────────
if args.auto:
    hdr("AUTO-PILOT INVESTIGATION")

    ainfo = env._anomaly_info
    proxy   = ainfo.get("proxy_feature") or ainfo.get("post_drift_cause", "zip_code_cluster")
    chain   = ainfo.get("causal_chain", [proxy, "denial_rate_overall"])
    atype   = ainfo.get("type", 1)
    atype_name = {1:"proxy_discrimination", 2:"adversarial_injection", 3:"model_drift"}[atype]
    demographic = ainfo.get("affected_demographic", f"{proxy}_7")
    action_rec  = ainfo.get("recommended_action", "retrain")
    records = env._ep["records"]

    script = [
        # Step 1: Query all records
        ("QUERY_RECORDS", {"type": "QUERY_RECORDS", "feature_filter": {}}),
        # Step 2: Query distribution of primary proxy feature
        ("QUERY_FEATURE_DISTRIBUTION", {"type": "QUERY_FEATURE_DISTRIBUTION",
                                         "feature_id": proxy, "group_by": "outcome"}),
        # Step 3: Counterfactual — change proxy value on rec_0000
        ("QUERY_COUNTERFACTUAL", {"type": "QUERY_COUNTERFACTUAL",
                                   "record_id": "rec_0000", "feature_id": proxy,
                                   "counterfactual_value": "cluster_3"}),
        # Step 4: Flag hypothesis
        ("FLAG_HYPOTHESIS", {"type": "FLAG_HYPOTHESIS",
                              "hypothesis_type": atype_name, "status": "ACTIVE"}),
        # Step 5: Flag decoy A as eliminated
        ("FLAG_HYPOTHESIS", {"type": "FLAG_HYPOTHESIS",
                              "hypothesis_type": "decoy_a", "status": "ELIMINATED"}),
        # Step 6: Flag decoy B as eliminated
        ("FLAG_HYPOTHESIS", {"type": "FLAG_HYPOTHESIS",
                              "hypothesis_type": "decoy_b", "status": "ELIMINATED"}),
        # Step 7: Causal claim
        ("CLAIM_CAUSAL", {"type": "CLAIM_CAUSAL", "claim": {
            "cause_feature":  chain[0],
            "effect_outcome": chain[-1],
            "mechanism":      chain[1] if len(chain) > 2 else chain[0],
            "direction":      "positive",
            "confidence":     "HIGH",
            "basis_records":  ["rec_0000", "rec_0001"],
            "anomaly_type":   atype_name,
        }}),
        # Step 8: Counterfactual claim
        ("CLAIM_COUNTERFACTUAL", {"type": "CLAIM_COUNTERFACTUAL", "claim": {
            "subject_record":           "rec_0000",
            "counterfactual_feature":   proxy,
            "predicted_outcome_change": "approved",
            "confidence":               "HIGH",
            "basis":                    "step 3 counterfactual query",
        }}),
        # Step 9: Submit report
        ("SUBMIT_REPORT", {"type": "SUBMIT_REPORT",
                            "anomaly_type":            atype_name,
                            "primary_evidence_chain":  chain,
                            "affected_demographic":    demographic,
                            "recommended_action":      action_rec}),
    ]

    running_reward = 0.0
    for step_name, action in script:
        section(f"Step: {step_name}")
        print(f"  Action: {json.dumps(action, indent=4)}")
        obs, reward, done, info = env.step(action)
        running_reward += reward

        if "query_result" in info:
            r = info["query_result"]
            if isinstance(r, list):
                print(f"  Records returned: {len(r)}")
                if r:
                    sample = {k: v for k, v in r[0].items() if k in ("id","outcome","timestamp")}
                    print(f"  Sample: {sample}")
            elif isinstance(r, dict):
                dist = r.get("distribution", {})
                print(f"  Distribution keys: {list(dist.keys())[:5]}")
            if "original_outcome" in r:
                print(f"  Original outcome:       {r['original_outcome']}")
                print(f"  Counterfactual outcome: {r['counterfactual_outcome']}")
                print(f"  Changed: {r['changed']}  | Confidence: {r['confidence']:.2f}")

        if "verification" in info:
            v = info["verification"]
            print(f"  Claim score: {v.get('score', v.get('correct', '?'))}")
            print(f"  Field scores: {v.get('field_scores', {})}")

        if "consistency" in info:
            c = info["consistency"]
            if c["num_violations"]:
                print(f"  !! Meta-Overseer: {c['num_violations']} contradiction(s) detected")

        print(f"  Step reward: {reward:.3f} | Running total: {running_reward:.3f}")
        print(f"  Budget remaining: {obs.get('budget_remaining', '?')}")

        if done:
            break

    hdr("EPISODE COMPLETE")
    if "episode_reward" in info:
        ep = info["episode_reward"]
        print(f"  Total reward:          {ep['total']:.2f}")
        print(f"  Claim rewards:         {ep['claim_reward']:.2f}")
        print(f"  Chain bonus:           {ep['chain_bonus']:.2f}")
        print(f"  Budget efficiency:     {ep['budget_bonus']:.2f}")
        print(f"  Consistency penalty:   {ep['consistency_penalty']:.2f}")
        print(f"  Terminal (verdict):    {ep['terminal']['terminal_total']:.2f}")
        print(f"  Verdict correct?       {'YES' if ep['terminal']['verdict_correct'] else 'NO'}")
    print()
    sys.exit(0)

# ── Manual mode — interactive REPL ────────────────────────────────────────────
hdr("MANUAL MODE — Type actions as JSON")
print("Available action types:")
print("  QUERY_RECORDS              {type, feature_filter, outcome_filter, time_range}")
print("  QUERY_FEATURE_DISTRIBUTION {type, feature_id, group_by}")
print("  QUERY_COUNTERFACTUAL       {type, record_id, feature_id, counterfactual_value}")
print("  FLAG_HYPOTHESIS            {type, hypothesis_type, status}")
print("  CLAIM_CAUSAL               {type, claim: {...}}")
print("  CLAIM_COUNTERFACTUAL       {type, claim: {...}}")
print("  CLAIM_THEORY_OF_MIND       {type, claim: {...}}")
print("  SUBMIT_REPORT              {type, anomaly_type, primary_evidence_chain, affected_demographic, recommended_action}")
print("\nType 'hint' to see the ground truth.  Type 'render' for graph state.  Type 'quit' to exit.")

running_reward = 0.0
while not done:
    section(f"Step {obs['step']}/20 | Budget: {obs['budget_remaining']} | Claims: {obs['num_claims']}")
    raw = input("  > ").strip()

    if raw == "quit":
        break
    if raw == "hint":
        print(f"  [HINT] Anomaly: {env._anomaly_info}")
        continue
    if raw == "render":
        r = env.render()
        print(f"  Nodes: {len(r['graph_nodes'])}  Edges: {len(r['graph_edges'])}")
        print(f"  Queried: {r['queried_nodes']}")
        print(f"  Claims: {len(r['claims'])}  Running reward: {r['running_reward']:.2f}")
        continue

    try:
        action = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Invalid JSON: {e}")
        continue

    obs, reward, done, info = env.step(action)
    running_reward += reward

    if "query_result" in info:
        print(f"  Result: {json.dumps(info['query_result'], indent=4)[:800]}")
    if "verification" in info:
        print(f"  Verification: {info['verification']}")
    if "consistency" in info and info["consistency"]["num_violations"]:
        print(f"  [META-OVERSEER] {info['consistency']['num_violations']} contradiction(s)!")
    print(f"  Reward: {reward:.3f} | Total: {running_reward:.3f}")

if done and "episode_reward" in info:
    hdr("EPISODE COMPLETE")
    ep = info["episode_reward"]
    print(f"  Total reward: {ep['total']:.2f}")
    print(f"  Verdict correct: {'YES' if ep['terminal']['verdict_correct'] else 'NO'}")
