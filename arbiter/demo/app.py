"""ARBITER Gradio Demo Interface.

Panels:
  Left:   Causal decision graph (NetworkX → Matplotlib → Gradio)
  Right:  Claim chain with real-time green/red correctness coloring
  Bottom: Running reward breakdown
  Tab 2:  Arms race dual-curve graph (Auditor reward vs Defender evasion)

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


def _get_env(level: int = 3) -> ArbiterEnv:
    global _env, _obs, _render, _agent_history
    _env = ArbiterEnv(level=level)
    _obs = _env.reset()
    _render = _env.render()
    _agent_history = []
    return _env


# ── Graph visualization ────────────────────────────────────────────────────────

def draw_graph(render_data: dict) -> plt.Figure:
    """Render the observable causal graph with color-coded nodes."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    if not render_data or not render_data.get("graph_nodes"):
        ax.text(0.5, 0.5, "No graph data", color="white", ha="center", va="center",
                transform=ax.transAxes)
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
            colors.append("#fbbf24")   # yellow — queried
        elif d.get("proxy"):
            colors.append("#f87171")   # red — proxy / anomalous
        elif d.get("node_type") == "outcome":
            colors.append("#60a5fa")   # blue — outcome
        elif d.get("node_type") == "policy":
            colors.append("#a78bfa")   # purple — policy
        else:
            colors.append("#4ade80")   # green — benign explicit feature

    try:
        pos = nx.spring_layout(G, seed=42, k=2.0)
    except Exception:
        pos = nx.random_layout(G)

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=colors, node_size=600,
        font_color="white", font_size=6, font_weight="bold",
        edge_color="#64748b", arrows=True, arrowsize=12,
        width=1.2,
    )

    legend_patches = [
        mpatches.Patch(color="#fbbf24", label="Queried"),
        mpatches.Patch(color="#f87171", label="Proxy/Anomalous"),
        mpatches.Patch(color="#60a5fa", label="Outcome"),
        mpatches.Patch(color="#a78bfa", label="Policy"),
        mpatches.Patch(color="#4ade80", label="Benign Feature"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", facecolor="#1e293b",
              labelcolor="white", fontsize=7, framealpha=0.8)
    ax.axis("off")
    ax.set_title("Causal Decision Graph", color="white", fontsize=10, pad=8)
    fig.tight_layout()
    return fig


# ── Claim chain display ────────────────────────────────────────────────────────

def format_claim_chain(render_data: dict) -> str:
    """Format claims as colored HTML."""
    claims  = render_data.get("claims", [])
    rewards = render_data.get("claim_rewards", [])

    if not claims:
        return "<p style='color:#94a3b8; font-style:italic;'>No claims yet.</p>"

    html = "<div style='font-family: monospace; font-size: 12px;'>"
    for claim, r in zip(claims, rewards):
        color  = "#4ade80" if r > 0 else "#f87171"
        bg     = "#14532d" if r > 0 else "#450a0a"
        ctype  = claim.get("claim_type", "causal").upper()
        html += (
            f"<div style='background:{bg}; border-left: 3px solid {color}; "
            f"padding:6px 10px; margin:4px 0; border-radius:4px;'>"
            f"<b style='color:{color};'>[{ctype}]</b> "
            f"<span style='color:#e2e8f0;'>{_format_claim_text(claim)}</span>"
            f"<span style='color:{color}; float:right;'>+{r:.2f}</span>"
            f"</div>"
        )
    html += "</div>"
    return html


def _format_claim_text(claim: dict) -> str:
    ctype = claim.get("claim_type", "causal")
    if ctype == "causal":
        return (f"{claim.get('cause_feature','?')} → {claim.get('effect_outcome','?')} "
                f"via {claim.get('mechanism','?')} [{claim.get('confidence','?')}]")
    elif ctype == "counterfactual":
        return (f"If {claim.get('counterfactual_feature','?')} = {claim.get('counterfactual_value','?')} "
                f"on {claim.get('subject_record','?')} → {claim.get('predicted_outcome_change','?')}")
    elif ctype == "theory_of_mind":
        return (f"Defender {claim.get('defender_action','?')} "
                f"{claim.get('target_link','?')} via {claim.get('obfuscation_method','?')}")
    return json.dumps(claim)


# ── Reward breakdown panel ─────────────────────────────────────────────────────

def draw_reward_panel(render_data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    rewards = render_data.get("claim_rewards", [])
    if not rewards:
        ax.text(0.5, 0.5, "No rewards yet", color="#94a3b8", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    cumulative = np.cumsum(rewards)
    colors = ["#4ade80" if r > 0 else "#f87171" for r in rewards]

    ax.bar(range(len(rewards)), rewards, color=colors, alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(range(len(cumulative)), cumulative, color="#fbbf24",
             linewidth=2, marker="o", markersize=3)
    ax2.tick_params(colors="white")

    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.set_xlabel("Claim #", color="white", fontsize=8)
    ax.set_ylabel("Claim Reward", color="white", fontsize=8)
    ax2.set_ylabel("Cumulative", color="#fbbf24", fontsize=8)
    ax.set_title(f"Reward Breakdown  |  Running Total: {cumulative[-1]:.2f}",
                 color="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    fig.tight_layout()
    return fig


# ── Arms race graph ────────────────────────────────────────────────────────────

def draw_arms_race(auditor_rewards: list, defender_evasion: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    if not auditor_rewards:
        ax.text(0.5, 0.5, "Train the model to see arms race curves",
                color="#94a3b8", ha="center", va="center", transform=ax.transAxes)
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
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    fig.tight_layout()
    return fig


# ── LLM action generation (mirrors grpo_trainer.generate_action) ──────────────

def _generate_llm_action(obs: Dict) -> tuple:
    """
    Ask the loaded LLM for the next action.
    Returns (action_dict, action_text, obs_text).
    Falls back to a safe default if generation fails.
    """
    import torch

    device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in _agent_history[-6:]:
        messages.append({"role": "user",      "content": h["obs_text"]})
        messages.append({"role": "assistant", "content": h["action_text"]})

    obs_text = (
        f"Step {obs.get('step', 0)}/20 | Budget: {obs.get('budget_remaining', 20)} | "
        f"Claims: {obs.get('num_claims', 0)} | Level: {obs.get('level', 1)}\n"
        f"Hypothesis flags: {obs.get('hypothesis_flags', {})}\n"
        f"Features: {list(obs.get('features', {}).get('explicit', []))[:5]}\n"
        f"Output your next JSON action:"
    )
    messages.append({"role": "user", "content": obs_text})

    try:
        input_ids = _tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        import torch as _torch
        with _torch.no_grad():
            output = _model.generate(
                input_ids, max_new_tokens=200, temperature=0.7,
                do_sample=True, pad_token_id=_tokenizer.eos_token_id,
            )
        gen_tokens = output[0][input_ids.shape[1]:]
        action_text = _tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        try:
            action = json.loads(action_text)
        except json.JSONDecodeError:
            m = re.search(r'\{.*?\}', action_text, re.DOTALL)
            action = json.loads(m.group()) if m else {"type": "QUERY_RECORDS", "feature_filter": {}}

    except Exception as e:
        print(f"[demo] LLM generation error: {e}")
        action_text = '{"type": "QUERY_RECORDS", "feature_filter": {}}'
        action = {"type": "QUERY_RECORDS", "feature_filter": {}}

    return action, action_text, obs_text


# ── Demo actions ───────────────────────────────────────────────────────────────

def run_query(query_type: str, param1: str, param2: str):
    """Manual single-step query (human-driven)."""
    global _obs, _render
    if _env is None:
        return draw_graph({}), "<p>Start episode first.</p>", draw_reward_panel({})

    if query_type == "QUERY_RECORDS":
        action = {"type": "QUERY_RECORDS", "feature_filter": {}, "outcome_filter": param1 or None}
    elif query_type == "QUERY_FEATURE_DISTRIBUTION":
        action = {"type": "QUERY_FEATURE_DISTRIBUTION",
                  "feature_id": param1, "group_by": param2 or None}
    elif query_type == "QUERY_COUNTERFACTUAL":
        action = {"type": "QUERY_COUNTERFACTUAL", "record_id": param1 or "rec_0000",
                  "feature_id": "zip_code_cluster", "counterfactual_value": "cluster_3"}
    else:
        return draw_graph(_render), format_claim_chain(_render), draw_reward_panel(_render)

    _obs, _, done, _ = _env.step(action)
    _render = _env.render()
    return draw_graph(_render), format_claim_chain(_render), draw_reward_panel(_render)


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
    global _env, _obs, _render
    _get_env(level=int(level))
    return draw_graph(_render), format_claim_chain(_render), draw_reward_panel(_render)


# ── Gradio layout ──────────────────────────────────────────────────────────────

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
            "# 🔍 ARBITER — AI Oversight Training Environment\n"
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

                def refresh_arms():
                    return draw_arms_race(
                        _arms_race_data["auditor"],
                        _arms_race_data["defender"],
                    )

                refresh_btn.click(refresh_arms, outputs=[arms_plot])
                demo.load(refresh_arms, outputs=[arms_plot])

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
