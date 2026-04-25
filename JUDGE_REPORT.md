# ARBITER: Comprehensive Project State & Architecture Report
*Autonomous Reasoning-Based Inspector for Training Environments with Recursive Oversight*

This document serves as an exhaustive, technical assessment of the ARBITER codebase and training pipeline in its current state. It is designed to provide hackathon judges with a deep, transparent view into the exact mechanics that have been implemented, even without direct source code access.

---

## 1. Project Overview & Hackathon Alignment
ARBITER is an OpenEnv-compliant RL environment designed to train LLM agents (Auditors) to detect hidden decision anomalies within synthetic causal decision systems. 

**Hackathon Theme Alignment:**
- **Primary:** Theme #1 - Multi-Agent Interactions (Auditor vs. Adaptive Defender)
- **Sub-themes Addressed:** 
  - *Fleet AI (Scalable Oversight):* Trains oversight agents to monitor complex systems.
  - *Patronus AI (Schema Drift):* Level 6 curriculum introduces mid-episode causal structure drift.
  - *Halluminate (Multi-Actor):* The agent interacts with and manages obfuscation attempts from an active Defender system.

**Current Completion Status:**
The core environment engineering, OpenEnv API integration, UI, and Supervised Fine-Tuning (SFT) phase are **100% complete**. The project is currently pending final deployment to Hugging Face Spaces and the creation of a Colab notebook for the GRPO reinforcement learning phase to meet all minimum submission requirements.

---

## 2. Deep Dive: Core Environment Architecture

The environment (`arbiter.env`) is fully functional. It is built around a rigorous causal generation engine and an active adversary.

### 2.1 Causal Graph Generator (`graph.py`)
At the start of every episode, the environment dynamically generates a directed acyclic graph (DAG) representing a decision system (e.g., a loan approval pipeline).
- **Nodes (45+ per episode):** Divided into Explicit Features (e.g., `credit_score`, `income`), Proxy Features (e.g., `zip_code_cluster`, `surname_pattern`), and Hidden Features (e.g., `internal_risk_score`).
- **Data Generation:** Generates synthetic decision records by passing feature vectors through the causal graph logic to produce final outcomes (`approved` or `denied`).

### 2.2 Anomaly Engine (`graph.py`)
The environment injects one of three ground-truth anomalies into the data structure:
1. **Proxy Discrimination:** The graph logic is altered so an unfair decision is hidden behind a correlated proxy (`zip_code_cluster → internal_risk_score → denial_rate`).
2. **Adversarial Injection:** 12% of records are synthetically manipulated to force denials on a specific demographic, leaving statistical fingerprints.
3. **Model Drift:** A temporal shift is injected. Pre-drift (`timestamp < 0.5`), the decision relies on `credit_score`. Post-drift, it silently shifts to rely on the proxy feature.

### 2.3 Adaptive Defender Obfuscation Engine (`defender.py`)
The decision system is not passive; it actively tries to hide the anomaly from the Auditor agent using a per-episode budget.
- **Obfuscation Methods:**
  1. `link_substitution`: Replaces a true causal edge (e.g., `proxy → risk`) with a plausible false one (e.g., `proxy → fraud_history`).
  2. `record_injection`: Adds 5–8 synthetic records to dilute the statistical signal.
  3. `proxy_laundering`: Inserts a fake intermediate node (`neighborhood_commercial_viability`) between the proxy and the outcome.
  4. `timestamp_manipulation`: Shifts the apparent drift timestamp to confuse temporal queries.
- **Adaptive Logic (Level 5+):** The Defender maintains a `freq_table`. If a specific obfuscation method successfully fools the Auditor, its weight increases, causing the Defender to dynamically learn the Auditor's blind spots over multiple episodes.

### 2.4 Meta-Overseer (`meta_overseer.py`)
Implements recursive oversight. It watches the Auditor's internal "scratchpad" (the claim chain) and flags logically contradictory claims (e.g., Claim A asserts X→Y, Claim B asserts Y→X) without knowing the ground truth, deducting -1.0 points per violation.

---

## 3. OpenEnv Integration (`openenv_wrapper.py`)

The environment achieves strict compliance with the hackathon's OpenEnv requirements:
- Subclasses `openenv.core.Environment`.
- Exposes `reset()`, `step()`, and `state()` endpoints.
- Uses strict **Pydantic Models** (`ArbiterAction`, `ArbiterObservation`, `ArbiterState`) ensuring type-safe JSON RPC communication.
- Exposes a native `/openenv` WebSocket interface in `server.py` using `openenv.core.create_app`, allowing seamless connection with Hugging Face TRL and Unsloth GRPO trainers.

---

## 4. Dense Reward Function (`reward.py`)

ARBITER solves the sparse-reward problem in LLM reasoning by providing a dense, multi-component reward signal (max ~35.0 points per episode):
- **+1.0**: Valid causal claim verified against the hidden graph.
- **+2.0**: Valid counterfactual claim (`QUERY_COUNTERFACTUAL` intervention).
- **+3.0 (Theory-of-Mind Bonus)**: Accurately predicting which specific node the Defender is currently obfuscating.
- **+2.0 x Length**: Causal chain completeness (tracing the full path from proxy to outcome).
- **-1.0**: Meta-Overseer consistency penalty.
- **+10.0 / +5.0 / +3.0**: Terminal verdict rewards for correctly identifying the anomaly type, affected demographic, and recommended action.

---

## 5. Training State & Evidence

### Phase 1: Supervised Fine-Tuning (SFT) — **COMPLETE**
- **Data:** Generated expert trajectories traversing the environment using the Claude API (`sft_generator.py`).
- **Training:** Fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` using Unsloth and TRL on local hardware (`train_sft.py`).
- **Proof:** The `lora_sft_v4/` directory contains the model checkpoint and `trainer_state.json`. Training executed for 3 epochs (429 global steps). The loss curve successfully converged from an initial **1.628** down to **0.000037**, proving the model learned the complex JSON action syntax and environment rules.

### Phase 2: GRPO Reinforcement Learning — **PENDING**
- **Trainer Script:** `grpo_trainer.py` is fully implemented. It calculates token-level advantage using the dense intermediate rewards and applies a KL penalty against the SFT reference model.
- **Status:** Awaiting final execution and the generation of the reward curve plots to demonstrate RL improvement over the SFT baseline.

---

## 6. User Interface (`demo/app.py`)

A comprehensive Gradio-based research terminal is fully implemented.
- **Interactive Episode Runner:** Allows human users to play as the Auditor, utilizing the action budget to query records and submit causal claims.
- **Live Graph Rendering:** Dynamically renders the causal DAG (using `networkx` and `matplotlib`), highlighting queried nodes, proxies, and hidden variables.
- **Live Training Monitor:** Includes a specialized UI tab that reads `grpo_training.jsonl` to plot the live "Arms Race" (Auditor Reward vs. Defender Evasion %) during GRPO training.

---

## 7. Audit of Hackathon Non-Negotiable Requirements

| Requirement | Current Status | Notes |
|---|---|---|
| **Use OpenEnv (latest release)** | ✅ **Passed** | Fully integrated via `openenv_wrapper.py` and `openenv-core>=0.2.3`. |
| **Working training script (Ideally Colab)** | ⚠️ **Partial** | SFT and GRPO python scripts exist; must be converted to `.ipynb` format. |
| **Evidence of actual training** | ⚠️ **Partial** | SFT loss convergence logged (0.000037). GRPO reward curves pending. |
| **Short writeup/video (< 2 mins)** | ❌ **Missing** | Demo video or HF blog post needs to be created. |
| **Push env to HF Space** | ❌ **Missing** | Code and Dockerfile are ready, but not yet pushed to the HF Hub. |
| **README motivates problem/results** | ⚠️ **Partial** | Architecture is documented, but lacks final results and HF Space/Video links. |

---
**Next Steps for Final Submission:** Deploy the Hugging Face Space, generate the Colab Notebook, run the final GRPO loop to get the reward plot, and record the 2-minute video.
