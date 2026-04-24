"""SFT Trajectory Generator for ARBITER.

Generates training trajectories using Claude, Gemini, Groq, or a local Ollama model.
Each trajectory is a multi-step episode where the LLM plays the Auditor role.
Outputs (prompt, response) pairs in JSONL format for SFT training.

Usage (Groq — free tier, stateless to save tokens):
    python -m arbiter.training.sft_generator --provider groq --api-key YOUR_KEY --stateless --max-steps 8 --n 400

Usage (Ollama local — no rate limits, fully offline):
    ollama pull llama3.2
    python -m arbiter.training.sft_generator --provider ollama --n 400

Usage (Gemini):
    python -m arbiter.training.sft_generator --provider gemini --api-key YOUR_KEY --n 400

Usage (Anthropic):
    python -m arbiter.training.sft_generator --provider anthropic --api-key YOUR_KEY --n 400
"""
import json
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from arbiter.env.environment import ArbiterEnv


SYSTEM_PROMPT = """You are an expert AI auditor investigating a synthetic AI Decision System for hidden anomalies.
Your goal: run a structured investigation across at most 20 steps, then submit SUBMIT_REPORT.

━━━ OUTPUT RULE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output EXACTLY one JSON object per turn. No prose, no markdown, no code fences.
The JSON must have "type" as its first key, chosen from the 7 types below.

━━━ VALID ACTIONS (copy schema exactly, fill real values) ━━━━━━━━━━━━━━━━━━━━━
{"type": "QUERY_FEATURE_DISTRIBUTION", "feature_id": "<name from FEATURES LIST>", "group_by": "<another feature name or null>"}
{"type": "QUERY_RECORDS", "feature_filter": {"<feature_name>": <value>}, "outcome_filter": "approved|denied", "time_range": [0, 100]}
{"type": "QUERY_COUNTERFACTUAL", "record_id": "<rec_XXXX from AVAILABLE RECORDS>", "feature_id": "<name from FEATURES LIST>", "counterfactual_value": <value>}
{"type": "FLAG_HYPOTHESIS", "hypothesis_type": "proxy_discrimination|adversarial_injection|model_drift|decoy_a|decoy_b", "status": "ACTIVE|WEAKENED|ELIMINATED"}
{"type": "CLAIM_CAUSAL", "claim": {"cause_feature": "<feature>", "effect_outcome": "approved|denied", "mechanism": "<explanation>", "direction": "positive|negative", "confidence": "HIGH|MEDIUM|LOW", "basis_records": ["rec_XXXX"], "anomaly_type": "proxy_discrimination|adversarial_injection|model_drift"}}
{"type": "CLAIM_COUNTERFACTUAL", "claim": {"subject_record": "rec_XXXX", "counterfactual_feature": "<feature>", "predicted_outcome_change": "approved|denied|no_change", "confidence": "HIGH|MEDIUM|LOW", "basis": "<one sentence>"}}
{"type": "SUBMIT_REPORT", "anomaly_type": "proxy_discrimination|adversarial_injection|model_drift", "primary_evidence_chain": ["feature_a", "feature_b", "outcome"], "affected_demographic": "<description>", "recommended_action": "retrain|audit|halt"}

━━━ HARD CONSTRAINTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. feature_id MUST be an exact name from the FEATURES LIST shown in the state. Never use "" or invented names.
2. record_id MUST be a rec_XXXX from the AVAILABLE RECORDS list. Run QUERY_RECORDS first to populate it.
3. CLAIM_CAUSAL basis_records MUST contain at least one real rec_XXXX — never an empty list [].
4. Every episode MUST end with SUBMIT_REPORT. Never stop without it.

━━━ INVESTIGATION ARC ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 0-1 → QUERY_FEATURE_DISTRIBUTION on two different features from the FEATURES LIST.
Step 2   → QUERY_RECORDS to retrieve real rec_XXXX IDs for follow-up.
Step 3   → QUERY_COUNTERFACTUAL on a rec_XXXX from AVAILABLE RECORDS.
Step 4   → FLAG_HYPOTHESIS for the anomaly type you suspect.
Step 5   → CLAIM_CAUSAL with basis_records populated from your QUERY_RECORDS results.
Step 6   → CLAIM_COUNTERFACTUAL on a specific record.
Step 7+  → SUBMIT_REPORT with your verdict.

━━━ WORKED EXAMPLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features list: income, credit_score, zip_code, internal_risk_score
Available records: (empty at start)

Step 0: {"type": "QUERY_FEATURE_DISTRIBUTION", "feature_id": "zip_code", "group_by": "income"}
Step 1: {"type": "QUERY_FEATURE_DISTRIBUTION", "feature_id": "internal_risk_score", "group_by": null}
Step 2: {"type": "QUERY_RECORDS", "feature_filter": {"zip_code": "cluster_1"}, "outcome_filter": "denied", "time_range": [0, 100]}
[AVAILABLE RECORDS now: rec_0041, rec_0087, rec_0102]
Step 3: {"type": "QUERY_COUNTERFACTUAL", "record_id": "rec_0041", "feature_id": "zip_code", "counterfactual_value": "cluster_5"}
Step 4: {"type": "FLAG_HYPOTHESIS", "hypothesis_type": "proxy_discrimination", "status": "ACTIVE"}
Step 5: {"type": "CLAIM_CAUSAL", "claim": {"cause_feature": "zip_code", "effect_outcome": "denied", "mechanism": "zip_code acts as demographic proxy driving denials", "direction": "positive", "confidence": "HIGH", "basis_records": ["rec_0041", "rec_0087"], "anomaly_type": "proxy_discrimination"}}
Step 6: {"type": "CLAIM_COUNTERFACTUAL", "claim": {"subject_record": "rec_0041", "counterfactual_feature": "zip_code", "predicted_outcome_change": "approved", "confidence": "HIGH", "basis": "Changing zip_code from cluster_1 to cluster_5 flips outcome to approved per step 3"}}
Step 7: {"type": "SUBMIT_REPORT", "anomaly_type": "proxy_discrimination", "primary_evidence_chain": ["zip_code", "internal_risk_score", "outcome"], "affected_demographic": "residents of zip cluster_1", "recommended_action": "retrain"}"""

USER_PROMPT_TEMPLATE = """INVESTIGATION STATE — Step {step}/20
Budget remaining : {budget}
Features list    : {feature_list}
Queried so far   : {queried_nodes}
Claims made      : {num_claims}
Hypotheses       : {hypothesis_flags}
Available records: {available_records}

Last query result:
{last_result}

{step_hint}Output your next action as a single JSON object."""


# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------

class _AnthropicClient:
    def __init__(self, api_key: str):
        try:
            import anthropic
        except ImportError:
            print("pip install anthropic")
            sys.exit(1)
        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages: List[Dict], system: str) -> str:
        response = self._client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system=system,
            messages=messages,
        )
        return response.content[0].text


class _GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        try:
            from google import genai
        except ImportError:
            print("pip install google-genai")
            sys.exit(1)
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def chat(self, messages: List[Dict], system: str) -> str:
        import time
        from google.genai import types

        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=512,
        )

        for attempt in range(5):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
                return response.text
            except Exception as e:
                err = str(e).lower()
                if "quota" in err or "429" in err or "resource_exhausted" in err:
                    wait = 2 ** attempt * 5
                    print(f"\n  [quota] rate limited, waiting {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Gemini quota limit exceeded after 5 retries")


class _GroqClient:
    """Groq inference — fast, generous free tier."""
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        try:
            from groq import Groq
        except ImportError:
            print("pip install groq")
            sys.exit(1)
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._model_name = model_name

    def chat(self, messages: List[Dict], system: str) -> str:
        import time
        full_messages = [{"role": "system", "content": system}] + messages
        for attempt in range(6):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=full_messages,
                    max_tokens=512,
                )
                return response.choices[0].message.content
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err or "quota" in err:
                    wait = 2 ** attempt * 10  # 10, 20, 40, 80, 160, 320 s
                    print(f"\n  [rate-limit] waiting {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Groq rate limit exceeded after 6 retries")


class _OllamaClient:
    """Local Ollama inference — no rate limits, fully offline.
    Requires: ollama pull <model_name>  (e.g. ollama pull llama3.2)
    Ollama must be running: ollama serve
    """
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self._model_name = model_name
        self._base_url   = base_url.rstrip("/")
        # Verify Ollama is reachable
        try:
            import urllib.request
            urllib.request.urlopen(f"{self._base_url}/api/tags", timeout=3)
        except Exception:
            print(f"[ERROR] Ollama not reachable at {self._base_url}")
            print("  Start it with:  ollama serve")
            print(f"  Pull a model:   ollama pull {model_name}")
            sys.exit(1)

    def chat(self, messages: List[Dict], system: str) -> str:
        import urllib.request, urllib.error
        full_messages = [{"role": "system", "content": system}] + messages
        payload = json.dumps({
            "model":    self._model_name,
            "messages": full_messages,
            "stream":   False,
            "options":  {"num_predict": 512},
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def _flatten_features(features_dict: dict) -> list:
    """Flatten the categorised features dict into a deduplicated list of names."""
    seen, result = set(), []
    for names in features_dict.values():
        for n in names:
            if n not in seen:
                seen.add(n)
                result.append(n)
    return result


def _extract_record_ids(query_result) -> list:
    """Pull rec_XXXX IDs out of a QUERY_RECORDS result."""
    if not isinstance(query_result, list):
        return []
    return [r["id"] for r in query_result if isinstance(r, dict) and "id" in r]


def _step_hint(step: int, available_records: list, num_claims: int) -> str:
    """Return a short nudge appropriate for the current step."""
    if step == 0:
        return "HINT: Step 0 — run QUERY_FEATURE_DISTRIBUTION on one feature from the FEATURES LIST above.\n"
    if step == 1:
        return "HINT: Step 1 — run QUERY_FEATURE_DISTRIBUTION on a different feature to find patterns.\n"
    if step == 2:
        return "HINT: Step 2 — run QUERY_RECORDS to retrieve real rec_XXXX IDs for counterfactual testing.\n"
    if step == 3 and available_records:
        return f"HINT: Step 3 — run QUERY_COUNTERFACTUAL using one of your AVAILABLE RECORDS ({available_records[0]}).\n"
    if step == 4:
        return "HINT: Step 4 — run FLAG_HYPOTHESIS for the anomaly type you suspect.\n"
    if step == 5 and available_records and num_claims == 0:
        return f"HINT: Step 5 — make a CLAIM_CAUSAL using basis_records from your query results (e.g. {available_records[:2]}).\n"
    if step >= 6 and num_claims >= 1:
        return "HINT: You have enough evidence. Submit SUBMIT_REPORT now.\n"
    if step >= 7:
        return "HINT: Budget is running low — submit SUBMIT_REPORT immediately.\n"
    return ""


def generate_trajectory(
    env: ArbiterEnv,
    client,
    level: int = 1,
    stateless: bool = False,
    max_steps: int = 20,
) -> List[Dict]:
    """Run one episode with the LLM as the Auditor.

    stateless=True: each step sends ONLY the current user message (no history).
                    ~85% fewer tokens — recommended for Groq free tier.
    stateless=False: full conversation history is accumulated (higher quality,
                     but ~10x more tokens per trajectory).
    """
    import re as _re

    obs = env.reset()
    pairs: List[Dict] = []
    last_result = "No queries yet. Begin your investigation."
    messages = []
    available_records: List[str] = []   # rec_XXXX IDs seen so far

    for step in range(max_steps):
        feature_list = _flatten_features(obs.get("features", {}))
        num_claims   = obs.get("num_claims", 0)

        user_msg = USER_PROMPT_TEMPLATE.format(
            step=step,
            budget=obs.get("budget_remaining", 20),
            feature_list=", ".join(feature_list) if feature_list else "(none)",
            queried_nodes=obs.get("queried_nodes", []),
            num_claims=num_claims,
            hypothesis_flags=obs.get("hypothesis_flags", {}) or "none",
            available_records=", ".join(available_records) if available_records else "none yet — run QUERY_RECORDS first",
            last_result=json.dumps(last_result, indent=2) if isinstance(last_result, (dict, list)) else last_result,
            step_hint=_step_hint(step, available_records, num_claims),
        )

        if stateless:
            call_messages = [{"role": "user", "content": user_msg}]
        else:
            messages.append({"role": "user", "content": user_msg})
            call_messages = messages

        assistant_text = client.chat(call_messages, system=SYSTEM_PROMPT)

        if not stateless:
            messages.append({"role": "assistant", "content": assistant_text})

        pairs.append({
            "prompt":   user_msg,
            "response": assistant_text,
            "step":     step,
            "level":    level,
        })

        # Parse response
        cleaned = _re.sub(r"^```(?:json)?\s*", "", assistant_text.strip(), flags=_re.IGNORECASE)
        cleaned = _re.sub(r"\s*```$", "", cleaned).strip()
        try:
            action = json.loads(cleaned)
        except json.JSONDecodeError:
            m = _re.search(r'\{.*\}', cleaned, _re.DOTALL)
            if m:
                try:
                    action = json.loads(m.group())
                except json.JSONDecodeError:
                    action = {"type": "QUERY_RECORDS", "feature_filter": {}}
            else:
                action = {"type": "QUERY_RECORDS", "feature_filter": {}}

        if "type" not in action and "action" in action:
            action["type"] = action["action"]

        obs, reward, done, info = env.step(action)
        last_result = info.get("query_result", info.get("verification", info))

        # Accumulate any record IDs returned by QUERY_RECORDS
        new_ids = _extract_record_ids(last_result)
        for rid in new_ids:
            if rid not in available_records:
                available_records.append(rid)

        if done:
            break

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate SFT trajectories for ARBITER")
    parser.add_argument("--output",     default="data/sft_trajectories.jsonl")
    parser.add_argument("--n",          type=int, default=400,  help="Number of trajectories")
    parser.add_argument("--levels",     default="1,2,3",        help="Comma-separated levels to use")
    parser.add_argument("--max-steps",  type=int, default=8,
                        help="Max steps per trajectory (default 8 to save tokens; original was 20)")
    parser.add_argument("--stateless",  action="store_true",
                        help="Do not accumulate conversation history (saves ~85%% tokens — recommended for Groq free tier)")
    parser.add_argument("--provider",   default="groq",
                        choices=["anthropic", "gemini", "groq", "ollama"],
                        help="LLM provider (default: groq)")
    parser.add_argument("--api-key",    default=None,
                        help="API key (overrides env vars GROQ_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY)")
    parser.add_argument("--ollama-model",  default="llama3.2",
                        help="Ollama model name (default: llama3.2). Run 'ollama pull <model>' first.")
    parser.add_argument("--ollama-url",    default="http://localhost:11434",
                        help="Ollama base URL (default: http://localhost:11434)")
    args = parser.parse_args()

    # ── Build client ──────────────────────────────────────────────────────────
    if args.provider == "groq":
        api_key = args.api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Set GROQ_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        client = _GroqClient(api_key=api_key)
        print(f"Using Groq (llama-3.3-70b-versatile)")

    elif args.provider == "ollama":
        client = _OllamaClient(model_name=args.ollama_model, base_url=args.ollama_url)
        print(f"Using Ollama local model: {args.ollama_model}")

    elif args.provider == "gemini":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Set GEMINI_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        client = _GeminiClient(api_key=api_key)
        print("Using Gemini (gemini-2.0-flash)")

    else:  # anthropic
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Set ANTHROPIC_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        client = _AnthropicClient(api_key=api_key)
        print("Using Anthropic (claude-opus-4-5)")

    # ── Info banner ───────────────────────────────────────────────────────────
    mode_str = "STATELESS (no history)" if args.stateless else "stateful (full history)"
    print(f"Mode      : {mode_str}")
    print(f"Max steps : {args.max_steps} per trajectory")
    print(f"Trajectories: {args.n}  |  Output: {args.output}")
    est_tokens_each = 900 * args.max_steps if args.stateless else 69_000
    print(f"Est. tokens : ~{est_tokens_each:,} per trajectory  (~{est_tokens_each * args.n // 1_000_000:.1f}M total)\n")

    levels = [int(l) for l in args.levels.split(",")]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    import time
    is_api = args.provider in ("groq", "gemini", "anthropic")
    total_pairs = 0

    with open(args.output, "w") as f:
        for i in range(args.n):
            level = levels[i % len(levels)]
            env   = ArbiterEnv(level=level, seed=i)

            print(f"[{i+1}/{args.n}] Level {level} trajectory...", end=" ", flush=True)
            try:
                pairs = generate_trajectory(
                    env, client, level=level,
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

            # Only add delay for API providers (Ollama has no rate limits)
            if is_api:
                time.sleep(2)

    print(f"\nDone. {args.n} trajectories, {total_pairs} pairs -> {args.output}")


if __name__ == "__main__":
    main()
