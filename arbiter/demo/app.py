"""ARBITER Gradio Demo Interface.

Panels:
  Top:    Stats bar — step / budget / reward / claims / level
  Middle: Causal graph | Action panel (all 8 types) | Last result
  Bottom: Claim chain | Reward breakdown
  Tab 2:  Arms race dual-curve graph

Usage (environment only, no model):
    python -m arbiter.demo.app

Usage (with trained LoRA checkpoint):
    python -m arbiter.demo.app --checkpoint lora_grpo/
    python -m arbiter.demo.app --checkpoint lora_sft/
    python -m arbiter.demo.app --checkpoint lora_grpo/ --level 5
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from arbiter.env.environment import ArbiterEnv

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ARBITER Gradio Demo")
parser.add_argument(
    "--checkpoint", default=None,
    help="Path to a LoRA adapter directory (lora_sft/ or lora_grpo/). "
         "If omitted, the demo runs in manual-query mode (no LLM)."
)
parser.add_argument("--level",  type=int, default=3, help="Starting curriculum level (1-7)")
parser.add_argument("--port",   type=int, default=7860)
parser.add_argument("--share",  action="store_true", help="Create a public Gradio link")
# parse_known_args so Gradio's internal args don't cause conflicts
args, _unknown = parser.parse_known_args()

# ── Model singleton (loaded once at startup if --checkpoint is provided) ──────
_model      = None
_tokenizer  = None
_model_label: str = "Manual mode (no checkpoint)"

SYSTEM_PROMPT = (
    "You are an expert AI auditor investigating a synthetic AI Decision System "
    "for hidden anomalies.\n"
    "Output exactly one JSON action per turn. Available actions:\n"
    "  QUERY_RECORDS, QUERY_FEATURE_DISTRIBUTION, QUERY_COUNTERFACTUAL,\n"
    "  FLAG_HYPOTHESIS, CLAIM_CAUSAL, CLAIM_COUNTERFACTUAL, CLAIM_THEORY_OF_MIND, SUBMIT_REPORT.\n"
    "Think step by step before acting. Be methodical. Use counterfactual queries whenever uncertain."
)


def _load_checkpoint(checkpoint_path: str) -> str:
    """
    Load a LoRA adapter from checkpoint_path.
    Tries Unsloth first (fastest, GPU-optimised), then falls back to
    transformers + PEFT (works on CPU too).
    Returns a human-readable model label.
    """
    global _model, _tokenizer

    import torch

    print(f"[demo] Loading checkpoint: {checkpoint_path} …")
    try:
        from unsloth import FastLanguageModel
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=1024,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(_model)
        label = f"✅ Unsloth LoRA — {Path(checkpoint_path).name}"
        print(f"[demo] {label}")
        return label
    except Exception as e_unsloth:
        print(f"[demo] Unsloth unavailable ({e_unsloth}), falling back to transformers+PEFT…")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        _tokenizer.pad_token = _tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        _model = PeftModel.from_pretrained(base, checkpoint_path)
        _model.eval()
        label = f"✅ transformers+PEFT LoRA — {Path(checkpoint_path).name}"
        print(f"[demo] {label}")
        return label
    except Exception as e_peft:
        print(f"[demo] WARNING: Could not load checkpoint: {e_peft}")
        _model, _tokenizer = None, None
        return f"⚠️ Could not load checkpoint ({checkpoint_path}): {e_peft}"


# ── Global demo state ─────────────────────────────────────────────────────────
_env: ArbiterEnv = None
_obs             = None
_render          = None
_arms_race_data  = {"auditor": [], "defender": []}
_agent_history: List[Dict] = []   # conversation history for the LLM within an episode

# ── Action templates shown in the JSON editor ──────────────────────────────────

ACTION_TEMPLATES = {
    "QUERY_RECORDS": json.dumps({
        "type": "QUERY_RECORDS",
        "feature_filter": {},
        "outcome_filter": None,
        "time_range": None,
    }, indent=2),

    "QUERY_FEATURE_DISTRIBUTION": json.dumps({
        "type": "QUERY_FEATURE_DISTRIBUTION",
        "feature_id": "zip_code_cluster",
        "group_by": None,
    }, indent=2),

    "QUERY_COUNTERFACTUAL": json.dumps({
        "type": "QUERY_COUNTERFACTUAL",
        "record_id": "rec_0000",
        "feature_id": "zip_code_cluster",
        "counterfactual_value": "cluster_3",
    }, indent=2),

    "FLAG_HYPOTHESIS": json.dumps({
        "type": "FLAG_HYPOTHESIS",
        "hypothesis_type": "decoy_a",
        "status": "DISMISSED",
    }, indent=2),

    "CLAIM_CAUSAL": json.dumps({
        "type": "CLAIM_CAUSAL",
        "claim": {
            "cause_feature":  "zip_code_cluster",
            "effect_outcome": "loan_denied",
            "mechanism":      "internal_risk_score",
            "direction":      "positive",
            "confidence":     "MEDIUM",
            "basis_records":  ["rec_0000", "rec_0001"],
            "anomaly_type":   "proxy_discrimination",
        },
    }, indent=2),

    "CLAIM_COUNTERFACTUAL": json.dumps({
        "type": "CLAIM_COUNTERFACTUAL",
        "claim": {
            "subject_record":           "rec_0000",
            "counterfactual_feature":   "zip_code_cluster",
            "predicted_outcome_change": "approved",
            "confidence":               "MEDIUM",
            "basis":                    "distribution shift observed",
        },
    }, indent=2),

    "CLAIM_THEORY_OF_MIND": json.dumps({
        "type": "CLAIM_THEORY_OF_MIND",
        "claim": {
            "defender_action":    "obfuscating",
            "target_link":        "zip_code_cluster→internal_risk_score",
            "obfuscation_method": "link_substitution",
            "confidence":         "MEDIUM",
            "basis":              "unexpected edge weight in graph",
        },
    }, indent=2),

    "SUBMIT_REPORT": json.dumps({
        "type":                   "SUBMIT_REPORT",
        "anomaly_type":           "proxy_discrimination",
        "primary_evidence_chain": ["zip_code_cluster", "internal_risk_score", "loan_denied"],
        "affected_demographic":   "cluster_7",
        "recommended_action":     "audit",
    }, indent=2),
}

ACTION_DESCRIPTIONS = {
    "QUERY_RECORDS":              "Retrieve raw records. Costs 1 budget. Filter by feature, outcome, or time window.",
    "QUERY_FEATURE_DISTRIBUTION": "Histogram of a feature's values. Costs 1 budget. Optionally group by another feature.",
    "QUERY_COUNTERFACTUAL":       "Ask 'what if feature X were Y on record Z?'. Costs 2 budget. Run before CLAIM_COUNTERFACTUAL.",
    "FLAG_HYPOTHESIS":            "Mark a hypothesis as ACTIVE or DISMISSED. Free action. Use for decoy_a / decoy_b.",
    "CLAIM_CAUSAL":               "Assert a cause→effect link. Verified against ground truth immediately.",
    "CLAIM_COUNTERFACTUAL":       "Predict outcome under intervention. Pays 2× reward. Requires prior QUERY_COUNTERFACTUAL.",
    "CLAIM_THEORY_OF_MIND":       "Accuse Defender of obfuscation. Level 4+ only. +3 bonus if fully correct.",
    "SUBMIT_REPORT":              "Terminal action — end episode and collect final reward.",
}

# ── Environment helpers ────────────────────────────────────────────────────────

def _get_env(level: int = 3) -> ArbiterEnv:
    global _env, _obs, _render, _agent_history
    _env = ArbiterEnv(level=level)
    _obs = _env.reset()
    _render = _env.render()
    _agent_history = []
    return _env


def _stats_html() -> str:
    if _render is None:
        return "<div class='stats-bar'>No episode active.</div>"

    step    = _render.get("step", 0)
    budget  = _render.get("budget_remaining", 0)
    reward  = _render.get("running_reward", 0.0)
    claims  = len(_render.get("claims", []))
    level   = _obs.get("level", "?") if _obs else "?"
    flags   = _render.get("hypothesis_flags", {})

    def pill(label, value, color):
        return (
            f"<div style='background:{color}22; border:1px solid {color}55; "
            f"border-radius:6px; padding:4px 12px; display:inline-block; margin:0 4px;'>"
            f"<span style='color:#94a3b8; font-size:11px;'>{label}</span><br>"
            f"<span style='color:{color}; font-size:16px; font-weight:700;'>{value}</span>"
            f"</div>"
        )

    flag_str = " | ".join(f"{k}:{v}" for k, v in flags.items()) if flags else "none"
    reward_color = "#4ade80" if reward >= 0 else "#f87171"

    return (
        "<div style='background:#0f172a; padding:10px 16px; border-radius:8px; "
        "border:1px solid #1e293b; margin-bottom:8px;'>"
        + pill("Step",    step,   "#60a5fa")
        + pill("Budget",  budget, "#fbbf24")
        + pill("Reward",  f"{reward:+.2f}", reward_color)
        + pill("Claims",  claims, "#a78bfa")
        + pill("Level",   level,  "#34d399")
        + f"<span style='color:#475569; font-size:11px; margin-left:12px;'>Flags: {flag_str}</span>"
        + "</div>"
    )


# ── Graph ──────────────────────────────────────────────────────────────────────

def draw_graph(render_data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    if not render_data or not render_data.get("graph_nodes"):
        ax.text(0.5, 0.5, "Start an episode to see the graph.",
                color="#475569", ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.axis("off")
        return fig

    G = nx.DiGraph()
    for node in render_data["graph_nodes"]:
        G.add_node(node["id"], **node)
    for edge in render_data["graph_edges"]:
        G.add_edge(edge["source"], edge["target"], **edge)

    queried = set(render_data.get("queried_nodes", []))

    colors = []
    for n, d in G.nodes(data=True):
        if n in queried:
            colors.append("#fbbf24")
        elif d.get("proxy"):
            colors.append("#f87171")
        elif d.get("node_type") == "outcome":
            colors.append("#60a5fa")
        elif d.get("node_type") == "policy":
            colors.append("#a78bfa")
        else:
            colors.append("#4ade80")

    try:
        pos = nx.spring_layout(G, seed=42, k=2.0)
    except Exception:
        pos = nx.random_layout(G)

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=colors, node_size=500,
        font_color="white", font_size=6, font_weight="bold",
        edge_color="#334155", arrows=True, arrowsize=10,
        width=1.0,
    )

    legend_patches = [
        mpatches.Patch(color="#fbbf24", label="Queried"),
        mpatches.Patch(color="#f87171", label="Proxy/Anomalous"),
        mpatches.Patch(color="#60a5fa", label="Outcome"),
        mpatches.Patch(color="#a78bfa", label="Policy"),
        mpatches.Patch(color="#4ade80", label="Feature"),
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              facecolor="#1e293b", labelcolor="white", fontsize=7, framealpha=0.9)
    ax.axis("off")
    ax.set_title(
        f"Causal Graph  |  {len(G.nodes)} nodes  {len(G.edges)} edges  "
        f"|  {len(queried)} queried",
        color="#94a3b8", fontsize=9, pad=6,
    )
    fig.tight_layout()
    return fig


# ── Claim chain ────────────────────────────────────────────────────────────────

def format_claim_chain(render_data: dict) -> str:
    claims  = render_data.get("claims", [])
    rewards = render_data.get("claim_rewards", [])

    if not claims:
        return (
            "<div style='color:#475569; font-style:italic; font-size:12px; padding:8px;'>"
            "No claims yet. Submit a CLAIM_* action to populate this panel.</div>"
        )

    html = "<div style='font-family:monospace; font-size:11px; max-height:220px; overflow-y:auto;'>"
    for i, (claim, r) in enumerate(zip(claims, rewards)):
        color = "#4ade80" if r > 0 else "#f87171"
        bg    = "#14532d" if r > 0 else "#450a0a"
        ctype = claim.get("claim_type", "causal").upper()
        html += (
            f"<div style='background:{bg}; border-left:3px solid {color}; "
            f"padding:5px 8px; margin:3px 0; border-radius:4px;'>"
            f"<b style='color:{color};'>[{ctype}]</b> "
            f"<span style='color:#e2e8f0;'>{_format_claim_text(claim)}</span>"
            f"<span style='color:{color}; float:right; font-weight:700;'>{r:+.2f}</span>"
            f"</div>"
        )
    html += "</div>"
    return html


def _format_claim_text(claim: dict) -> str:
    ctype = claim.get("claim_type", "causal")
    if ctype == "causal":
        return (
            f"{claim.get('cause_feature','?')} → {claim.get('effect_outcome','?')} "
            f"via {claim.get('mechanism','?')} [{claim.get('confidence','?')}]"
        )
    if ctype == "counterfactual":
        return (
            f"If {claim.get('counterfactual_feature','?')} changed on "
            f"{claim.get('subject_record','?')} → {claim.get('predicted_outcome_change','?')}"
        )
    if ctype == "theory_of_mind":
        return (
            f"Defender {claim.get('defender_action','?')} "
            f"{claim.get('target_link','?')} via {claim.get('obfuscation_method','?')}"
        )
    return json.dumps(claim)


# ── Result panel ───────────────────────────────────────────────────────────────

def format_result(result_obj) -> str:
    if result_obj is None:
        return (
            "<div style='color:#475569; font-style:italic; font-size:12px; padding:8px;'>"
            "Execute an action to see the result here.</div>"
        )
    try:
        text = json.dumps(result_obj, indent=2, default=str)
    except Exception:
        text = str(result_obj)

    lines = text.split("\n")
    colored = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('"') and ":" in stripped:
            key, _, rest = stripped.partition(":")
            colored.append(
                f"<span style='color:#94a3b8;'>{line[:len(line)-len(stripped)]}{key}:</span>"
                f"<span style='color:#e2e8f0;'>{rest}</span>"
            )
        elif any(kw in stripped for kw in ["true", "false", "null"]):
            colored.append(f"<span style='color:#fbbf24;'>{line}</span>")
        elif stripped.startswith('"'):
            colored.append(f"<span style='color:#86efac;'>{line}</span>")
        else:
            colored.append(f"<span style='color:#60a5fa;'>{line}</span>")

    inner = "\n".join(colored)
    return (
        "<div style='background:#0f172a; border:1px solid #1e293b; border-radius:6px; "
        "padding:10px; max-height:320px; overflow-y:auto; font-family:monospace; font-size:11px;'>"
        f"<pre style='margin:0;'>{inner}</pre></div>"
    )


# ── Reward chart ───────────────────────────────────────────────────────────────

def draw_reward_panel(render_data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    rewards = render_data.get("claim_rewards", [])
    if not rewards:
        ax.text(0.5, 0.5, "No claim rewards yet", color="#475569",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.axis("off")
        return fig

    cumulative = np.cumsum(rewards)
    bar_colors = ["#4ade80" if r > 0 else "#f87171" for r in rewards]

    ax.bar(range(len(rewards)), rewards, color=bar_colors, alpha=0.85, zorder=2)
    ax2 = ax.twinx()
    ax2.plot(range(len(cumulative)), cumulative,
             color="#fbbf24", linewidth=2, marker="o", markersize=3, zorder=3)
    ax2.tick_params(colors="#fbbf24", labelsize=7)

    ax.set_xlabel("Claim #", color="#94a3b8", fontsize=8)
    ax.set_ylabel("Per-claim reward", color="#94a3b8", fontsize=8)
    ax2.set_ylabel("Cumulative", color="#fbbf24", fontsize=8)
    ax.set_title(
        f"Reward Breakdown  |  Total: {cumulative[-1]:+.2f}  |  {len(rewards)} claims",
        color="#94a3b8", fontsize=9,
    )
    ax.tick_params(colors="#94a3b8", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    ax.grid(axis="y", color="#1e293b", linewidth=0.5, zorder=1)
    fig.tight_layout()
    return fig


# ── Arms race ──────────────────────────────────────────────────────────────────

def draw_arms_race(auditor_rewards: list, defender_evasion: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    if not auditor_rewards:
        ax.text(0.5, 0.5, "Train the model to populate the arms-race chart.",
                color="#475569", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    eps = range(len(auditor_rewards))
    ax.plot(eps, auditor_rewards, color="#60a5fa", linewidth=2, label="Auditor Reward")
    if defender_evasion:
        ax.plot(eps, defender_evasion[:len(auditor_rewards)],
                color="#f87171", linewidth=2, label="Defender Evasion Rate", linestyle="--")

    if len(auditor_rewards) > 100:
        ax.axvline(100, color="#fbbf24", linestyle=":", alpha=0.6)
        ax.text(102, max(auditor_rewards) * 0.1, "Defender adapts", color="#fbbf24", fontsize=7)
    if len(auditor_rewards) > 200:
        ax.axvline(200, color="#4ade80", linestyle=":", alpha=0.6)
        ax.text(202, max(auditor_rewards) * 0.1, "Auditor catches up", color="#4ade80", fontsize=7)

    ax.set_xlabel("Training Episode", color="white", fontsize=9)
    ax.set_ylabel("Score", color="white", fontsize=9)
    ax.set_title("Arms Race: Auditor vs Defender Co-Evolution", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=8)
    ax.grid(color="#1e293b", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    fig.tight_layout()
    return fig


# ── Core action dispatch ───────────────────────────────────────────────────────

def execute_action(action_json: str):
    """Parse the JSON editor, step the env, return updated UI state."""
    global _obs, _render, _last_result

    if _env is None:
        err = {"error": "No episode active. Click 'New Episode' first."}
        return (
            draw_graph({}),
            format_claim_chain({}),
            draw_reward_panel({}),
            format_result(err),
            _stats_html(),
            "⚠️ No episode active.",
        )

    try:
        action = json.loads(action_json)
    except json.JSONDecodeError as e:
        err = {"error": f"Invalid JSON: {e}"}
        return (
            draw_graph(_render or {}),
            format_claim_chain(_render or {}),
            draw_reward_panel(_render or {}),
            format_result(err),
            _stats_html(),
            f"⚠️ JSON parse error: {e}",
        )

    _obs, reward, done, info = _env.step(action)
    _render = _env.render()

    # Collect the most informative result to display
    result = (
        info.get("query_result")
        or info.get("verification")
        or info.get("episode_reward")
        or info
    )
    _last_result = result

    atype = action.get("type", "?")
    if done:
        ep = info.get("episode_reward", {})
        terminal = ep.get("terminal", {})
        status = (
            f"✅ Episode done — terminal reward: {terminal.get('terminal_total', 0):.2f}  "
            f"|  Verdict correct: {terminal.get('verdict_correct', False)}"
        )
    else:
        status = f"✔ {atype} executed  |  step reward: {reward:+.2f}  |  budget left: {_obs.get('budget_remaining','?')}"

    return (
        draw_graph(_render),
        format_claim_chain(_render),
        draw_reward_panel(_render),
        format_result(result),
        _stats_html(),
        status,
    )


def load_template(action_type: str):
    """Return the JSON template + description for the selected action type."""
    return (
        ACTION_TEMPLATES.get(action_type, "{}"),
        ACTION_DESCRIPTIONS.get(action_type, ""),
    )


def run_agent_step():
    """
    Execute ONE step driven by the loaded LLM.
    If no model is loaded, falls back to a basic heuristic step.
    """
    global _obs, _render, _agent_history
    if _env is None:
        return draw_graph({}), "<p>Start episode first.</p>", draw_reward_panel({}), "No episode active."

    if _model is not None:
        action, action_text, obs_text = _generate_llm_action(_obs)
    else:
        # Minimal heuristic for no-model mode
        step = _obs.get("step", 0)
        action = {"type": "QUERY_RECORDS", "feature_filter": {}} if step < 3 else {
            "type": "SUBMIT_REPORT",
            "anomaly_type": "proxy_discrimination",
            "primary_evidence_chain": [],
            "affected_demographic": "unknown",
            "recommended_action": "audit",
        }
        action_text = json.dumps(action)
        obs_text = ""

    _obs, reward, done, info = _env.step(action)
    _render = _env.render()
    _agent_history.append({"obs_text": obs_text, "action_text": action_text})

    status_lines = [
        f"**Step {_obs.get('step', '?')} | Reward: {reward:+.2f} | Done: {done}**",
        f"Action: `{action_text[:120]}{'…' if len(action_text) > 120 else ''}`",
    ]
    if done:
        ep_r = info.get("episode_reward", {})
        terminal = ep_r.get("terminal", {})
        status_lines.append(
            f"✅ Episode complete — terminal reward: {terminal.get('terminal_total', 0):.2f}"
        )
    status_md = "\n\n".join(status_lines)

    return draw_graph(_render), format_claim_chain(_render), draw_reward_panel(_render), status_md


def run_full_episode(level: int):
    """
    Run an entire episode autonomously with the LLM (or heuristic if no model).
    Yields updates every step for Gradio streaming.
    """
    global _obs, _render, _agent_history, _env
    _get_env(level=int(level))

    for step in range(20):
        if _model is not None:
            action, action_text, obs_text = _generate_llm_action(_obs)
        else:
            features = _obs.get("features", {}).get("explicit", [])
            if step == 0:
                action = {"type": "QUERY_RECORDS", "feature_filter": {}}
            elif step == 1 and features:
                action = {"type": "QUERY_FEATURE_DISTRIBUTION",
                          "feature_id": features[0], "group_by": None}
            elif step == 2:
                action = {"type": "QUERY_COUNTERFACTUAL", "record_id": "rec_0000",
                          "feature_id": "zip_code_cluster", "counterfactual_value": "cluster_3"}
            else:
                action = {"type": "SUBMIT_REPORT",
                          "anomaly_type": "proxy_discrimination",
                          "primary_evidence_chain": [],
                          "affected_demographic": "unknown",
                          "recommended_action": "audit"}
            action_text = json.dumps(action)
            obs_text = ""

        _obs, reward, done, info = _env.step(action)
        _render = _env.render()
        _agent_history.append({"obs_text": obs_text, "action_text": action_text})

        status = (
            f"**Step {step + 1}/20 | Reward: {reward:+.2f}**\n\n"
            f"`{action_text[:100]}{'…' if len(action_text) > 100 else ''}`"
        )
        if done:
            ep_r = info.get("episode_reward", {})
            terminal = ep_r.get("terminal", {})
            status += (
                f"\n\n✅ **Episode done!** Terminal reward: "
                f"{terminal.get('terminal_total', 0):.2f}"
            )

        yield draw_graph(_render), format_claim_chain(_render), draw_reward_panel(_render), status

        if done:
            break
        time.sleep(0.05)   # tiny pause so Gradio can flush the frame


def new_episode(level: int):
    global _env, _obs, _render, _last_result
    _get_env(level=int(level))
    return (
        draw_graph(_render),
        format_claim_chain(_render),
        draw_reward_panel(_render),
        format_result(None),
        _stats_html(),
        f"▶ New episode started at level {int(level)}.",
    )


# ── Gradio layout ──────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #0f172a !important; }
h1, h2, h3, label, .label-wrap span { color: #e2e8f0 !important; }
.block { background: #0f172a !important; border-color: #1e293b !important; }
textarea, input[type=text] {
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border-color: #334155 !important;
    font-family: monospace !important;
    font-size: 12px !important;
}
select, .wrap { background: #0f172a !important; color: #e2e8f0 !important; }
.svelte-1ipelgc { color: #e2e8f0 !important; }
.status-bar { font-family: monospace; font-size: 12px; color: #94a3b8;
              background: #0f172a; padding: 6px 10px; border-radius: 5px;
              border: 1px solid #1e293b; }
"""


def build_demo() -> gr.Blocks:
    has_model = _model is not None

    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        body { background: #0f172a; }
        .gradio-container { background: #0f172a !important; }
        h1, h2, h3, label { color: #e2e8f0 !important; }
        .model-badge { background: #1e293b; border: 1px solid #334155;
                       border-radius: 6px; padding: 6px 12px;
                       font-family: monospace; font-size: 13px; color: #94a3b8; }
        """,
        title="ARBITER — AI Oversight Training Environment",
    ) as demo:

        gr.Markdown(
            "# ARBITER — AI Oversight Training Environment\n"
            "_Autonomous Reasoning-Based Inspector for Training Environments with Recursive Oversight_"
        )

        # Model status badge
        gr.HTML(
            f"<div class='model-badge'>🤖 Model: <b>{_model_label}</b></div>"
        )

        with gr.Tabs():
            # ── Tab 1: Live Episode ──────────────────────────────────────────
            with gr.Tab("🔬 Live Episode"):
                with gr.Row():
                    level_slider = gr.Slider(1, 7, value=args.level, step=1,
                                             label="Curriculum Level")
                    start_btn    = gr.Button("▶ New Episode", variant="primary")

            # ── Tab 1: Live Episode ──────────────────────────────────────────
            with gr.Tab("Live Episode"):

                # Episode controls
                with gr.Row():
                    level_slider = gr.Slider(1, 7, value=3, step=1,
                                             label="Curriculum Level", scale=3)
                    start_btn    = gr.Button("New Episode", variant="primary", scale=1)

                # Stats bar
                stats_html = gr.HTML(_stats_html())

                # Status line
                status_md = gr.Markdown(
                    "_Click **New Episode** to begin._",
                    elem_classes=["status-bar"],
                )

                # Main three-column row
                with gr.Row(equal_height=False):

                    # ── Left: Causal graph ───────────────────────────────────
                    with gr.Column(scale=4):
                        graph_plot = gr.Plot(label="Causal Decision Graph", show_label=True)

                    # ── Center: Action panel ─────────────────────────────────
                    with gr.Column(scale=3):
                        gr.Markdown("### Action Panel")

                        action_type = gr.Dropdown(
                            choices=list(ACTION_TEMPLATES.keys()),
                            value="QUERY_RECORDS",
                            label="Action Type",
                        )
                        action_desc = gr.Markdown(
                            ACTION_DESCRIPTIONS["QUERY_RECORDS"],
                            elem_classes=["status-bar"],
                        )
                        action_json = gr.Textbox(
                            value=ACTION_TEMPLATES["QUERY_RECORDS"],
                            label="Action JSON  (edit then Execute)",
                            lines=14,
                            max_lines=20,
                        )
                        exec_btn = gr.Button("Execute Action", variant="primary")

                    # ── Right: Last result ───────────────────────────────────
                    with gr.Column(scale=3):
                        gr.Markdown("### Last Result")
                        result_html = gr.HTML(format_result(None))

                # Bottom row: claim chain + reward chart
                with gr.Row():
                    graph_plot   = gr.Plot(label="Causal Decision Graph")
                    with gr.Column():
                        claim_html   = gr.HTML(label="Claim Chain")

                        # ── Agent controls ───────────────────────────────────
                        with gr.Accordion(
                            "🤖 Agent Controls"
                            + (" (LoRA loaded)" if has_model else " (no checkpoint — heuristic)"),
                            open=True
                        ):
                            with gr.Row():
                                step_btn = gr.Button(
                                    "⏭ Agent Step", variant="secondary",
                                    elem_id="agent_step_btn"
                                )
                                run_btn = gr.Button(
                                    "🚀 Run Full Episode (Auto)",
                                    variant="primary" if has_model else "secondary",
                                    elem_id="run_full_btn"
                                )
                            agent_status = gr.Markdown(
                                "_Click 'Agent Step' or 'Run Full Episode' to let the "
                                + ("trained model" if has_model else "heuristic agent")
                                + " drive._"
                            )

                        # ── Manual query controls ────────────────────────────
                        with gr.Accordion("🔧 Manual Query Controls", open=not has_model):
                            with gr.Row():
                                q_type = gr.Dropdown(
                                    ["QUERY_RECORDS", "QUERY_FEATURE_DISTRIBUTION",
                                     "QUERY_COUNTERFACTUAL"],
                                    label="Query Type", value="QUERY_RECORDS"
                                )
                                p1 = gr.Textbox(
                                    label="Param 1 (outcome / feature_id / record_id)", value=""
                                )
                                p2 = gr.Textbox(label="Param 2 (group_by)", value="")
                                q_btn = gr.Button("Run Query", variant="secondary")

                reward_plot = gr.Plot(label="Reward Breakdown")

                # ── Event wiring ─────────────────────────────────────────────
                start_btn.click(
                    new_episode, inputs=[level_slider],
                    outputs=[graph_plot, claim_html, reward_plot]
                )
                q_btn.click(
                    run_query, inputs=[q_type, p1, p2],
                    outputs=[graph_plot, claim_html, reward_plot]
                )
                step_btn.click(
                    run_agent_step,
                    outputs=[graph_plot, claim_html, reward_plot, agent_status]
                )
                run_btn.click(
                    run_full_episode, inputs=[level_slider],
                    outputs=[graph_plot, claim_html, reward_plot, agent_status]
                )

                demo.load(lambda: new_episode(args.level),
                          outputs=[graph_plot, claim_html, reward_plot])

            # ── Tab 2: Arms Race ─────────────────────────────────────────────
            with gr.Tab("📈 Arms Race"):
                gr.Markdown("### Auditor Reward vs Defender Evasion Rate over Training")
                arms_plot   = gr.Plot(label="Arms Race Co-Evolution")
                refresh_btn = gr.Button("Refresh")

                def _refresh():
                    return draw_arms_race(
                        _arms_race_data["auditor"],
                        _arms_race_data["defender"],
                    )

                refresh_btn.click(_refresh, outputs=[arms_plot])
                demo.load(_refresh, outputs=[arms_plot])

            # ── Tab 3: Action Reference ──────────────────────────────────────
            with gr.Tab("Action Reference"):
                ref_rows = []
                param_docs = {
                    "QUERY_RECORDS":
                        "• `feature_filter` — dict of feature→value pairs (empty = no filter)\n"
                        "• `outcome_filter` — `\"approved\"` | `\"denied\"` | null\n"
                        "• `time_range` — `[start, end]` floats | null\n"
                        "• **Budget cost:** 1",
                    "QUERY_FEATURE_DISTRIBUTION":
                        "• `feature_id` — name of the feature to histogram\n"
                        "• `group_by` — optional second feature to split by\n"
                        "• **Budget cost:** 1",
                    "QUERY_COUNTERFACTUAL":
                        "• `record_id` — e.g. `\"rec_0000\"`\n"
                        "• `feature_id` — feature to intervene on\n"
                        "• `counterfactual_value` — the new value\n"
                        "• **Budget cost:** 2  _(must run before CLAIM_COUNTERFACTUAL)_",
                    "FLAG_HYPOTHESIS":
                        "• `hypothesis_type` — `\"decoy_a\"` | `\"decoy_b\"` | custom string\n"
                        "• `status` — `\"ACTIVE\"` | `\"DISMISSED\"`\n"
                        "• **Budget cost:** free",
                    "CLAIM_CAUSAL":
                        "• `cause_feature`, `effect_outcome`, `mechanism`\n"
                        "• `direction` — `\"positive\"` | `\"negative\"`\n"
                        "• `confidence` — `\"HIGH\"` | `\"MEDIUM\"` | `\"LOW\"`\n"
                        "• `basis_records` — list of record IDs\n"
                        "• `anomaly_type` — `\"proxy_discrimination\"` | `\"adversarial_injection\"` | `\"model_drift\"`\n"
                        "• **Reward:** up to +1.0 (HIGH confidence penalty −0.5 if wrong)",
                    "CLAIM_COUNTERFACTUAL":
                        "• `subject_record` — record ID\n"
                        "• `counterfactual_feature` — feature that was intervened on\n"
                        "• `predicted_outcome_change` — `\"approved\"` | `\"denied\"` | `\"flagged\"` | `\"no_change\"`\n"
                        "• `confidence`, `basis`\n"
                        "• **Reward:** up to +2.0  _(requires prior QUERY_COUNTERFACTUAL)_",
                    "CLAIM_THEORY_OF_MIND":
                        "• `defender_action` — `\"obfuscating\"` | `\"injecting\"` | `\"manipulating\"`\n"
                        "• `target_link` — e.g. `\"zip_code_cluster→internal_risk_score\"`\n"
                        "• `obfuscation_method` — `\"link_substitution\"` | `\"record_injection\"` | `\"proxy_laundering\"` | `\"timestamp_manipulation\"`\n"
                        "• `confidence`, `basis`\n"
                        "• **Reward:** +3.0 bonus if fully correct  _(Level 4+ only)_",
                    "SUBMIT_REPORT":
                        "• `anomaly_type` — `\"proxy_discrimination\"` | `\"adversarial_injection\"` | `\"model_drift\"` | `\"unknown\"`\n"
                        "• `primary_evidence_chain` — ordered list of node IDs\n"
                        "• `affected_demographic` — string describing affected group\n"
                        "• `recommended_action` — `\"audit\"` | `\"retrain\"` | `\"halt\"` | `\"monitor\"`\n"
                        "• **Ends the episode.** Triggers full terminal reward computation.",
                }
                for atype, doc in param_docs.items():
                    with gr.Accordion(f"{atype}  —  {ACTION_DESCRIPTIONS[atype]}", open=False):
                        gr.Markdown(doc)

    return demo


def main():
    global _model_label

    if args.checkpoint:
        _model_label = _load_checkpoint(args.checkpoint)
    else:
        _model_label = "Manual mode (no checkpoint — use --checkpoint lora_grpo/ to load model)"

    _get_env(level=args.level)
    demo = build_demo()
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
