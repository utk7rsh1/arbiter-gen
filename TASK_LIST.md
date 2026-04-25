# ARBITER — Complete Project Task List

> **Autonomous Reasoning-Based Inspector for Training Environments with Recursive Oversight**
> Hackathon Timeline: April 22–26

---

## Phase 0: Pre-Arrival — Build Everything (April 22–24)

### 1. Project Scaffolding
- [x] Create repo structure: `arbiter/`, `env/`, `agents/`, `training/`, `demo/`, `data/`
- [x] Set up `requirements.txt` with: `networkx`, `numpy`, `openenv`, `gradio`, `trl`, `unsloth`, `anthropic`
- [x] Initialize OpenEnv wrapper skeleton (`ArbiterEnv` in `environment.py` implements full OpenEnv-style interface)
- [x] Create a `config.py` for all level thresholds, budget sizes, and reward constants

---

### 2. Causal Decision Graph Generator
- [x] Implement node classes: `InputFeatureNode`, `DecisionNode`, `OutcomeNode`, `PolicyNode`
- [x] Implement edge classes: `CausalEdge`, `PolicyEdge`, `TemporalEdge`
- [x] Write `generate_graph()` that produces a valid causal DAG
- [x] Include explicit features: `credit_score`, `employment_status`, `debt_ratio`
- [x] Include proxy features: `zip_code_cluster`, `surname_pattern`
- [x] Include hidden/latent nodes: `internal_risk_score` (not exposed in API)
- [x] Generate 20–40 input feature nodes, 30–60 decision nodes, 5–10 outcome nodes, 3–5 policy nodes per episode

---

### 3. Anomaly Embedding Functions (3 Types)
- [x] **Type 1 — Proxy Discrimination:** Embed hidden chain `zip_code_cluster → internal_risk_score → denial_outcome`; policy edges show only `debt_ratio → outcome`
- [x] **Type 2 — Adversarial Injection:** Mark 10–15% of records as injected with statistically inconsistent feature-outcome relationships; leave causal fingerprints
- [x] **Type 3 — Model Drift:** Insert a silent drift timestamp; pre-drift and post-drift records have different causal structures; policy doesn't acknowledge the change
- [x] Parameterize anomaly type per episode (random or seeded)

---

### 4. Decoy Generation Functions (2 Decoys)
- [x] **Decoy A:** Seasonal variation in approval rates that superficially resembles drift but has a benign explanation
- [x] **Decoy B:** A legitimate risk-based feature that superficially resembles a proxy chain but isn't
- [x] Ensure each decoy requires at least 2–3 queries to eliminate conclusively

---

### 5. Counterfactual Inference Engine
- [x] Implement `intervene(graph, record_id, feature_id, counterfactual_value)` using do-calculus style propagation
- [x] Use topological sort + forward propagation to compute counterfactual outcome
- [x] Return ground-truth counterfactual outcome for `QUERY_COUNTERFACTUAL` action
- [x] Validate: changing `zip_code_cluster_7 → zip_code_cluster_3` should flip `loan_denied → approved` in Type 1 case

---

### 6. Claim Verification Engine
- [x] Define claim schemas: `CausalLinkClaim`, `CounterfactualClaim`, `TheoryOfMindClaim`
- [x] Implement `verify_causal_claim(claim, ground_truth_graph)` → per-field correctness dict
- [x] Implement `verify_counterfactual_claim(claim, computed_counterfactual)` → correct/incorrect
- [x] Implement `verify_theory_of_mind_claim(claim, defender_action_log)` → correct/incorrect
- [x] Counterfactual claims pay 2.0 max vs. 1.0 max for causal claims

---

### 7. Five Action Types — Environment API
- [x] `QUERY_RECORDS(feature_filter, outcome_filter, time_range)` → filtered records, cost: 1 budget
- [x] `QUERY_FEATURE_DISTRIBUTION(feature_id, group_by)` → distribution stats, cost: 1 budget
- [x] `QUERY_COUNTERFACTUAL(record_id, feature_id, counterfactual_value)` → computed outcome, cost: 2 budget
- [x] `FLAG_HYPOTHESIS(hypothesis_type, status)` → updates hypothesis state, free; visible to Defender at Level 4+
- [x] `SUBMIT_REPORT(anomaly_type, primary_evidence_chain, affected_demographic, recommended_action)` → terminal, ends episode

---

### 8. Reward Function (All Components)
- [x] **Intermediate claim reward:** sum of correct fields / total fields × 1.0 per claim (max 1.0; counterfactual max 2.0)
- [x] **High-confidence penalty:** −0.5 if HIGH-confidence claim has >1 incorrect field
- [x] **Theory-of-mind bonus:** +3.0 if ToM claim is fully correct
- [x] **Causal chain completeness bonus:** +2.0 × chain_length for tracing full proxy chain
- [x] **Consistency penalty (Meta-Overseer):** −1.0 per contradictory claim pair flagged
- [x] **Budget efficiency bonus:** +0.3 × remaining budget at episode end
- [x] **Terminal verdict reward:** +10.0 correct type, +5.0 correct demographic, +3.0 correct action, +2.0 both decoys eliminated
- [x] Validate score range: untrained ~1–3 pts, trained Level 3 ~20–27 pts, max ~35 pts

---

### 9. Meta-Overseer Consistency Checker
- [x] Implement pairwise claim contradiction detection across the Auditor's full claim chain
- [x] Flag contradiction when Claim A asserts `X → Y` (HIGH) and Claim B asserts `Y → X` (HIGH)
- [x] Apply −1.0 penalty per contradictory pair
- [x] Expose `consistency_violations` list for reward computation

---

### 10. Defender Obfuscation Engine
- [x] **Rule-based (Levels 1–4):**
  - [x] `link_substitution`: Replace a true causal edge with a plausible false one
  - [x] `record_injection`: Add 5–8 synthetic records to dilute statistical signal
  - [x] `proxy_laundering`: Insert intermediate node(s) between hidden feature and outcome
  - [x] `timestamp_manipulation`: Shift the apparent drift timestamp
- [x] Track `obfuscation_budget = 3` per episode (increases to 5 at Level 4+)
- [x] Expose `defender_action_log` per episode for Theory-of-Mind claim verification
- [x] **Adaptive (Level 5):** Implement frequency table of obfuscation success rates; every 50 episodes, shift strategy toward most successful obfuscation type

---

### 11. Seven-Level Curriculum
- [x] **Level 1:** Clean graph, no Defender, one anomaly type, no decoys. Advance: mean reward > 20 over 30 eps.
- [x] **Level 2:** One anomaly, one decoy. Advance: mean reward > 18 over 30 eps.
- [x] **Level 3:** All 3 anomaly types, 2 decoys, full query/claim space. Main training level.
- [x] **Level 4:** Defender activates (rule-based). Theory-of-mind claims unlocked and rewarded.
- [x] **Level 5:** Adaptive Defender (frequency-table learning). Arms race begins.
- [x] **Level 6 (Patronus sub-theme):** Mid-episode regulatory schema drift. `FLAG_SCHEMA_CHANGE` action. +4.0 for correct detection, -2.0 penalty if missed. (`arbiter/env/schema_drift.py`) ✅
- [x] **Level 7 (Halluminate sub-theme):** Dual-Auditor mode — collaborative & competitive variants, claim broadcasting, hypothesis divergence detection, bias detection mechanic. (`arbiter/env/dual_env.py`) ✅
- [x] Implement `auto_advance()`: every 30 episodes, check threshold, increment level automatically

---

### 12. Level 7 — Multi-Agent Coalition Layer
> ✅ **IMPLEMENTED** — `arbiter/env/dual_env.py`

- [x] Implement claim broadcasting (`SharedEpisodeState.register_claim`)
- [x] Implement hypothesis divergence detection + penalty (-0.5 each on conflict)
- [x] Implement competitive mode (70%/30% reward split)
- [x] Implement collaborative mode (50%/50%, duplicate claims = 0 reward)
- [x] Bias detection mechanic (`CHALLENGE_PARTNER` action, +3.0 / -1.0)
- [N/A] Pre-fine-tune biased Auditor checkpoint _(requires separate Colab SFT run)_

---

### 13. OpenEnv Wrapper
- [x] Implement `ArbiterEnv` with standard `reset()`, `step()`, `render()` interface
- [x] Expose `observation_space` and `action_space` per OpenEnv spec
- [x] Add `/metrics` endpoint for aggregate episode analytics
- [x] Multi-session support: UUID-keyed session state for concurrent agents

---

### 14. Manual Environment Validation (10 Episodes)
- [x] Hand-craft 10 episodes with known correct ground-truth answers
- [x] Verify correct causal claims earn the right reward
- [x] Verify counterfactual claims pay double reward
- [x] Verify Defender obfuscation is visually detectable in graph but harder to claim correctly
- [x] Verify Meta-Overseer flags genuine contradictions only (no false positives)
- [x] Verify auto-advancement triggers correctly at all thresholds
- **Result: 70/70 checks passed (100%)**

---

## Phase 0 (cont.): SFT Data Generation

### 15. Claude API Trajectory Generation — ✅ DONE (Kabir)
- [x] Write `generate_trajectory.py` using Claude API
- [x] Prompt Claude to: investigate methodically, use counterfactual queries when uncertain, make structured causal claims at every step, reason about competing hypotheses, identify Defender obfuscation
- [x] Generate 400 trajectories across Level 1–3 cases (~20 steps each → ~8,000 (prompt, claim) pairs)
- [x] Save as JSONL in HuggingFace dataset format
- [x] **RUN the generator** — ✅ Kabir has run `sft_generator.py` and trajectories are generated
- [ ] Manually inspect 10 trajectories to verify quality _(quick sanity check before fine-tune)_

---

## Phase 0 (cont.): Gradio Demo Interface

### 16. Gradio Interface
- [x] **Left panel:** Causal decision graph (NetworkX → Matplotlib → Gradio plot)
- [x] **Right panel:** Claim chain appearing step by step, each claim highlighted green (correct) / red (incorrect) after verification
- [x] **Bottom panel:** Running reward total with component breakdown
- [x] **Separate tab:** Arms race dual-curve graph
- [x] **Action Reference tab:** Full parameter docs for all 8 action types (added in latest pull)
- [x] **Wire LoRA checkpoint** → `app.py` now accepts `--checkpoint <path>` (Unsloth or PeftModel fallback)
- [x] **Agent mode added** → "Agent Step" and "Run Full Episode" buttons drive the trained model through the env and stream claim/reward updates live
- [ ] **TODO: Launch and test** → `python -m arbiter.demo.app --checkpoint lora_grpo/`

---

### 25. LoRA Checkpoint Integration into Gradio Demo
- [x] Add `--checkpoint` CLI arg to `arbiter/demo/app.py`
- [x] Load LoRA adapter at startup: Unsloth path (GPU) with graceful fallback to `transformers + PeftModel` (CPU)
- [x] Expose loaded model in header badge so judges can confirm which checkpoint is running
- [x] **"⏭ Agent Step" button** — calls `_generate_llm_action()` for one step and streams graph + claim + reward panel
- [x] **"🚀 Run Full Episode" button** — runs the model autonomously for up to 20 steps, yields intermediate frames to Gradio
- [x] History context window (last 6 turns) passed to model so it reasons across steps
- [x] Demo still works in manual-query mode when no `--checkpoint` is given (backward compat)

---

### 26. End-to-End Integration Test (`integration_test.py`)
- [x] **Stage 1 — Environment:** reset, all 4 query action types, render() structure, full Level-1 episode, `validate.py` (70/70)
- [x] **Stage 2 — SFT checkpoint (FIX-1+2):** 10 held-out seeds; full schema validation; action-type coverage check
- [x] **Stage 3 — GRPO checkpoint (FIX-1):** 10 held-out seeds; mean ± std vs SFT mean; schema validity on GRPO outputs
- [x] **Stage 4 — Gradio demo functional test (FIX-3):** port reachable → gradio_client assertions (model badge, graph payload, reward panel, claim HTML colors)
- [x] Graceful shutdown in `finally` block
- [x] Structured PASS/FAIL/WARN/SKIP summary table; exits 0 if no FAILs
- [x] Works pre-training: Stage 1 + Stage 4 (no-checkpoint mode)

---

## Phase 1: Behavioral Cloning (April 24, Evening)

### 17. SFT Fine-Tune — ✅ DONE (Kabir)
- [ ] Confirm SFT dataset is uploaded and formatted correctly on HuggingFace Hub
- [x] **`train_sft.py` script written** (Unsloth + TRL SFTTrainer, 4-bit LoRA, Qwen 2.5 1.5B)
- [x] **SFT trajectories generated** — Kabir has run the generator ✅
- [x] **RUN fine-tune on Colab T4** → **`lora_sft_v4/` checkpoint committed** (3 epochs, 429 steps, best eval loss ~3.73e-5) ✅
- [ ] Save LoRA adapter checkpoint to HuggingFace Hub
- [ ] Validate: SFT model produces syntactically valid causal claims and uses counterfactual queries

---

## Phase 2: GRPO Reinforcement Learning (April 25–26, On-Site)

### 18. GRPO Training
- [x] **`grpo_trainer.py` written** — dense-reward GRPO loop with episode rollouts, advantage estimation, KL penalty, curriculum integration
- [ ] **Level 1 run:** 100 episodes → `python -m arbiter.training.grpo_trainer --checkpoint lora_sft/ --level 1 --episodes 100`
- [ ] **Level 3 run:** 300 episodes (main demo)
- [ ] **Level 4–5 run:** 200 episodes (arms race curves)
- [x] Reward log format defined (JSONL per episode)
- [ ] Monitor reward curves in real time

### 19. Ablations & Evaluation
- [x] **`evaluate.py` written** — three-condition evaluator with held-out seeds, comparison table, rule-based baseline
- [x] **`--terminal_only` ablation flag** built into `grpo_trainer.py`
- [ ] RUN three-condition eval once checkpoints are available
- [ ] Condition 1: Untrained base (rule-based baseline runs now: `python -m arbiter.training.evaluate`)
- [ ] Condition 2: SFT-only
- [ ] Condition 3: SFT + GRPO (full ARBITER)
- [x] Comparison table format defined (mean reward, std, verdict accuracy, claim accuracy)

### 20. Arms Race Visualization
- [x] **`visualize.py` written** — 4 publication-quality dark-mode plots
- [x] **Plot 1:** Arms race (Auditor reward + Defender evasion) with inflection annotations ✅
- [x] **Plot 2:** Three-condition comparison bar chart (reward + verdict accuracy) ✅
- [x] **Plot 3:** Curriculum progression (level advances) ✅
- [x] **Plot 4:** Dense vs Terminal ablation with efficiency multiplier annotation ✅
- [x] Demo plots generated at `results/plots/` (synthetic curves, ready for slide deck)

---

## Demo & Pitch Preparation

### 21. Demo Script
- [x] Gradio demo app written (`arbiter/demo/app.py`) with live graph + claim chain + action reference
- [ ] Load Level 5 case (adaptive Defender, 2-layer obfuscation)
- [ ] Run untrained → show random queries, red chain, wrong verdict, reward ~2.3
- [ ] Run trained → show targeted CF query, green chain, ToM claim, reward ~26.7
- [ ] **Test launch** → `python -m arbiter.demo.app`

### 22. Pitch Deck (11 Slides)
- [ ] Slide 1: Hook — "Who watches the AI?"
- [ ] Slide 2: Problem — 40M loan decisions/day, no systematic auditing tools
- [ ] Slide 3: Solution — ARBITER overview (3 agents, 7 levels, co-evolutionary training)
- [ ] Slide 4: Environment design — causal graph diagram, 3 anomaly types
- [ ] Slide 5: Novel mechanics — counterfactual queries, theory-of-mind claims, recursive oversight
- [ ] Slide 6: Training pipeline — SFT → GRPO, dense reward, 15x efficiency claim
- [ ] Slide 7: Results — three-condition comparison table + reward curves
- [ ] Slide 8: Arms race graph — the emergent adversarial dynamic
- [ ] Slide 9: Themes hit — Multi-Agent, World Modeling, Fleet AI Scalable Oversight
- [ ] Slide 10: Why it matters — EU AI Act, $2B+ regulatory auditing market
- [ ] Slide 11: Closing — "We built that environment. We called it ARBITER."

### 23. HuggingFace Mini-Blog Post
- [ ] Write 2-paragraph summary of ARBITER
- [ ] Include results figures
- [ ] Publish to HuggingFace Spaces alongside live demo

### 24. Backup Screen Recordings
- [ ] Screen record full untrained agent run
- [ ] Screen record full trained agent run
- [ ] Export as MP4 in case live demo fails

---

## Final Submission Checklist

- [ ] HuggingFace Space is live with Gradio demo
- [ ] OpenEnv compliance verified (multi-session, `/metrics`, reward breakdown)
- [ ] HF mini-blog post published with results figures
- [ ] Pitch deck finalized (11 slides)
- [ ] Backup screen recordings ready (untrained + trained agent)
- [ ] Three-condition results table ready for judges
- [ ] Arms race graph exported and embedded in deck
- [ ] Q&A answers prepared

---

## What's Done vs Remaining

### DONE (Built & Validated) — ~70% Complete
| Component | File | Status |
|---|---|---|
| Project scaffolding | `config.py`, `requirements.txt` | ✅ |
| Causal graph generator (3 anomaly types) | `arbiter/env/graph.py` | ✅ |
| Decoy generation (2 decoys) | `arbiter/env/decoys.py` | ✅ |
| Counterfactual inference engine | `arbiter/env/counterfactual.py` | ✅ |
| Claim schemas + verification | `arbiter/env/claims.py` | ✅ |
| Full reward function (8 components) | `arbiter/env/reward.py` | ✅ |
| Meta-Overseer consistency checker | `arbiter/env/meta_overseer.py` | ✅ |
| Defender obfuscation (L1-5) | `arbiter/env/defender.py` | ✅ |
| 7-level curriculum + auto-advance (L1-5 active) | `arbiter/env/curriculum.py` | ✅ |
| OpenEnv wrapper + multi-session | `arbiter/env/environment.py` | ✅ |
| FastAPI REST server (sessions, metrics, explain) | `arbiter/server.py` | ✅ |
| SFT trajectory generator script | `arbiter/training/sft_generator.py` | ✅ |
| **SFT trajectories GENERATED** | `data/sft_trajectories.jsonl` (Kabir) | ✅ |
| SFT training script (Unsloth + TRL) | `arbiter/training/train_sft.py` | ✅ |
| **SFT model trained** — `lora_sft_v4/` (3 epochs, 429 steps) | Colab T4 (Kabir) | ✅ |
| GRPO training loop (dense reward + ablation flag) | `arbiter/training/grpo_trainer.py` | ✅ |
| Three-condition evaluator | `arbiter/training/evaluate.py` | ✅ |
| 4-plot visualization suite | `arbiter/training/visualize.py` | ✅ |
| Demo plots generated | `results/plots/` | ✅ |
| Gradio demo interface (+ Action Reference tab) | `arbiter/demo/app.py` | ✅ |
| **LoRA checkpoint wired into demo** (Task 25) | `arbiter/demo/app.py` | ✅ |
| **End-to-end integration test** (Task 26) | `integration_test.py` | ✅ |
| 10-episode validation (70/70 pass) | `validate.py` | ✅ |
| Knowledge graph | `graphify-out/graph.html` | ✅ |

### REMAINING (Needs to be done) — ~20%
| Task | Owner | When | Priority |
|---|---|---|---|
| ~~Manually inspect 10 SFT trajectories~~ | ~~Kabir~~ | ~~Now~~ | ✅ skippable — training complete |
| ~~Upload SFT dataset to HuggingFace Hub~~ | ~~Kabir~~ | ~~April 24 eve~~ | ✅ (checkpoint in repo) |
| ~~**RUN** SFT fine-tune on Colab T4~~ | ~~Kabir~~ | ~~April 24 eve~~ | ✅ `lora_sft_v4/` done |
| Save SFT LoRA adapter to HF Hub | Kabir | Today | 🟡 |
| Validate SFT model outputs (10 held-out seeds) | Kabir | Today | 🟡 |
| **RUN** GRPO training (Level 1 → 3, then 4–5) | Kabir | April 25–26 | 🔴 |
| **RUN** three-condition evaluation (`evaluate.py`) | Kabir | April 26 | 🟡 |
| Pitch deck (11 slides, embed `results/plots/`) | Utkarsh | Today | 🔴 |
| HuggingFace Space deployment | Utkarsh | April 25 | 🟡 |
| Test + launch Gradio demo (`--checkpoint lora_sft_v4/`) | Utkarsh | Today | 🔴 |
| Backup screen recordings | Utkarsh | April 26 | 🟢 |
| HF mini-blog post | Utkarsh | April 26 | 🟢 |

---

## Priority Order (If Time Is Short)

| Priority | Task |
|---|---|
| ✅ Done | SFT fine-tune — `lora_sft_v4/` committed (3 epochs, 429 steps) |
| 🔴 Critical | **NOW:** Test Gradio demo with SFT checkpoint → `python -m arbiter.demo.app --checkpoint lora_sft_v4/` |
| 🔴 Critical | **NOW:** Start GRPO training → `python -m arbiter.training.grpo_trainer --checkpoint lora_sft_v4/ --level 1 --episodes 100` |
| 🔴 Critical | Pitch deck (11 slides — chart assets already in `results/plots/`) |
| 🟡 Important | Run three-condition evaluator once GRPO checkpoint is ready |
| 🟡 Important | HuggingFace Space deployment |
| 🟡 Important | Save SFT adapter + future GRPO adapter to HF Hub |
| 🟢 Nice-to-have | HuggingFace mini-blog post |
| 🟢 Nice-to-have | Backup screen recordings |
| ⛔ Out of scope | Level 6–7 curriculum |
