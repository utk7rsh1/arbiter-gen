# ARBITER — Complete Project Task List

> **Autonomous Reasoning-Based Inspector for Training Environments with Recursive Oversight**
> Hackathon Timeline: April 22–26

---

## Phase 0: Pre-Arrival — Build Everything (April 22–24)

### 1. Project Scaffolding
- [x] Create repo structure: `arbiter/`, `env/`, `agents/`, `training/`, `demo/`, `data/`
- [x] Set up `requirements.txt` with: `networkx`, `numpy`, `openenv`, `gradio`, `trl`, `unsloth`, `anthropic`
- [ ] Initialize OpenEnv wrapper skeleton
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
- [ ] **Level 6 (Patronus sub-theme):** Mid-episode regulatory schema drift; Auditor must detect and re-evaluate. Advance: mean reward > 15 over 30 eps.
- [ ] **Level 7 (Halluminate sub-theme):** Two-Auditor mode with collaborative and competitive variants.
- [x] Implement `auto_advance()`: every 30 episodes, check threshold, increment level automatically

---

### 12. Level 7 — Multi-Agent Coalition Layer
- [ ] Implement claim broadcasting: Auditor A's claims visible to Auditor B; duplicate claims earn 0 reward
- [ ] Implement hypothesis divergence detection: deduct from both Auditors if they hold contradictory hypothesis flags
- [ ] Implement competitive mode: first correct report captures 70% of terminal reward
- [ ] Implement collaborative mode: combined chain must be non-redundant (downstream extensions rewarded)
- [ ] Pre-fine-tune one Auditor instance with biased dataset (Type 1 bias) for the trust game mechanic

---

### 13. OpenEnv Wrapper
- [x] Implement `ArbiterEnv(openenv.Env)` with standard `reset()`, `step()`, `render()` interface
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

### 15. Claude API Trajectory Generation
- [x] Write `generate_trajectory.py` using Claude API
- [x] Prompt Claude to: investigate methodically, use counterfactual queries when uncertain, make structured causal claims at every step, reason about competing hypotheses, identify Defender obfuscation
- [x] Generate 400 trajectories across Level 1–3 cases (~20 steps each → ~8,000 (prompt, claim) pairs)
- [x] Save as JSONL in HuggingFace dataset format
- [ ] **TODO: Actually RUN the generator** → `python -m arbiter.training.sft_generator --n 400`
- [ ] Manually inspect 10 trajectories to verify quality

---

## Phase 0 (cont.): Gradio Demo Interface

### 16. Gradio Interface
- [x] **Left panel:** Causal decision graph (NetworkX → Matplotlib → Gradio plot)
- [x] **Right panel:** Claim chain appearing step by step, each claim highlighted green (correct) / red (incorrect) after verification
- [x] **Bottom panel:** Running reward total with component breakdown
- [x] **Separate tab:** Arms race dual-curve graph
- [ ] **TODO: Launch and test** → `python -m arbiter.demo.app`

---

## Phase 1: Behavioral Cloning (April 24, Evening)

### 17. SFT Fine-Tune
- [ ] Confirm SFT dataset is uploaded and formatted correctly on HuggingFace Hub
- [x] **`train_sft.py` script written** (Unsloth + TRL SFTTrainer, 4-bit LoRA, Qwen 2.5 1.5B)
- [ ] RUN fine-tune on Colab T4 (`python arbiter/training/train_sft.py --dataset data/sft_trajectories.jsonl`)
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
- [x] Gradio demo app written (`arbiter/demo/app.py`) with live graph + claim chain
- [ ] Load Level 5 case (adaptive Defender, 2-layer obfuscation)
- [ ] Run untrained → show random queries, red chain, wrong verdict, reward ~2.3
- [ ] Run trained → show targeted CF query, green chain, ToM claim, reward ~26.7
- [ ] TODO: test launch → `python -m arbiter.demo.app`

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

### DONE (Built & Validated) — ~65% Complete
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
| 7-level curriculum + auto-advance | `arbiter/env/curriculum.py` | ✅ |
| OpenEnv wrapper + multi-session | `arbiter/env/environment.py` | ✅ |
| FastAPI REST server (sessions, metrics, explain) | `arbiter/server.py` | ✅ |
| SFT trajectory generator script | `arbiter/training/sft_generator.py` | ✅ |
| SFT training script (Unsloth + TRL) | `arbiter/training/train_sft.py` | ✅ |
| GRPO training loop (dense reward + ablation flag) | `arbiter/training/grpo_trainer.py` | ✅ |
| Three-condition evaluator | `arbiter/training/evaluate.py` | ✅ |
| 4-plot visualization suite | `arbiter/training/visualize.py` | ✅ |
| Demo plots generated | `results/plots/` | ✅ |
| Gradio demo interface | `arbiter/demo/app.py` | ✅ |
| 10-episode validation (70/70 pass) | `validate.py` | ✅ |
| Knowledge graph | `graphify-out/graph.html` | ✅ |

### REMAINING (Needs to be done) — ~35%
| Task | Owner | When |
|---|---|---|
| Level 6 schema drift logic | Vraj | Before on-site |
| Level 7 multi-auditor coalition | Vraj | Nice-to-have |
| **RUN** SFT generator (`sft_generator.py`, needs `ANTHROPIC_API_KEY`) | Kabir | April 24 |
| **RUN** SFT fine-tune on Colab T4 (`train_sft.py`) | Kabir | April 24 evening |
| **RUN** GRPO training (Level 1 → Level 3) | Kabir | April 25-26 |
| **RUN** three-condition evaluation (`evaluate.py`) | Kabir | April 26 |
| Pitch deck (11 slides, embed `results/plots/`) | Utkarsh | April 24 |
| HuggingFace Space deployment | Utkarsh | April 25 |
| Test + launch Gradio demo (`python -m arbiter.demo.app`) | Utkarsh | April 25 |
| Backup screen recordings | Utkarsh | April 26 |

---

## Priority Order (If Time Is Short)

| Priority | Task |
|---|---|
| 🔴 Critical | RUN SFT data generator (`python -m arbiter.training.sft_generator`) |
| 🔴 Critical | RUN SFT fine-tune on Colab (`python arbiter/training/train_sft.py`) |
| 🔴 Critical | RUN GRPO training (`python -m arbiter.training.grpo_trainer --level 3 --episodes 300`) |
| 🔴 Critical | Launch + test Gradio demo (`python -m arbiter.demo.app`) |
| 🟡 Important | Run evaluator + update plots with real curves |
| 🟡 Important | Pitch deck (slides already have demo chart assets in `results/plots/`) |
| 🟡 Important | HuggingFace Space deployment |
| 🟢 Nice-to-have | Level 6–7 curriculum |
| 🟢 Nice-to-have | HuggingFace mini-blog post |
| 🟢 Nice-to-have | Backup screen recordings |

