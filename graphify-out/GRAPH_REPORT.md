# Graph Report - Arbiter  (2026-04-23)

## Corpus Check
- 26 files · ~61,277 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 189 nodes · 220 edges · 16 communities detected
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]

## God Nodes (most connected - your core abstractions)
1. `step()` - 12 edges
2. `run_full_episode()` - 8 edges
3. `format_claim_chain()` - 7 edges
4. `run_agent_step()` - 7 edges
5. `_get_env()` - 6 edges
6. `draw_graph()` - 6 edges
7. `run_query()` - 6 edges
8. `new_episode()` - 6 edges
9. `draw_reward_panel()` - 5 edges
10. `run_episode_with_model()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `step_endpoint()` --calls--> `step()`  [INFERRED]
  arbiter\server.py → test_steps.py
- `run_query()` --calls--> `step()`  [INFERRED]
  arbiter\demo\app.py → test_steps.py
- `run_agent_step()` --calls--> `step()`  [INFERRED]
  arbiter\demo\app.py → test_steps.py
- `run_full_episode()` --calls--> `step()`  [INFERRED]
  arbiter\demo\app.py → test_steps.py
- `run_episode_with_model()` --calls--> `step()`  [INFERRED]
  arbiter\training\evaluate.py → test_steps.py

## Hyperedges (group relationships)
- **Three-Agent ARBITER System** — arbiter_idea_Auditor, arbiter_idea_Defender, arbiter_idea_MetaOverseer [EXTRACTED 1.00]
- **Training Pipeline (SFT→GRPO→Arms Race)** — arbiter_idea_SFT, arbiter_idea_GRPO, arbiter_idea_ArmsRace [EXTRACTED 1.00]
- **Three Anomaly Types** — arbiter_idea_ProxyDiscrimination, arbiter_idea_AdversarialInjection, arbiter_idea_ModelDrift [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (23): Exception, generate_action(), grpo_update(), GRPO Training Loop for ARBITER.  Runs dense-reward GRPO reinforcement learning, Run one complete episode with the LLM.     Returns (total_reward, step_rewards,, Simplified GRPO update:     - Compute advantage = reward - mean(batch_rewards), Query the LLM for the next action given observation., run_episode() (+15 more)

### Community 1 - "Community 1"
Cohesion: 0.1
Nodes (21): BaseModel, CreateSessionRequest, explain_endpoint(), _get_env(), global_metrics(), leaderboard(), FastAPI REST Server for ARBITER — OpenEnv-compliant.  Endpoints:   POST /sess, Post-hoc ground truth explanation for the current episode. (+13 more)

### Community 2 - "Community 2"
Cohesion: 0.17
Nodes (21): build_demo(), draw_graph(), draw_reward_panel(), format_claim_chain(), _format_claim_text(), _generate_llm_action(), _get_env(), _load_checkpoint() (+13 more)

### Community 3 - "Community 3"
Cohesion: 0.11
Nodes (21): ARBITER System, Adversarial Injection (Type 2), Arms Race Dynamic, Auditor Agent, Causal Decision Graph, Counterfactual Claim, 7-Level Curriculum, Defender Agent (+13 more)

### Community 4 - "Community 4"
Cohesion: 0.18
Nodes (8): _AnthropicClient, _GeminiClient, generate_trajectory(), _GroqClient, main(), SFT Trajectory Generator for ARBITER.  Generates 400 training trajectories using, Groq inference — very fast, generous free tier., Run one episode with the LLM as the Auditor. Returns list of (prompt, response)

### Community 5 - "Community 5"
Cohesion: 0.24
Nodes (9): generate_llm_action(), load_model(), Three-Condition Evaluation for ARBITER.  Runs 10 held-out Level 3 episodes und, Generate an action from an LLM., Run one episode. Returns metrics dict., Load a model and tokenizer for inference., Simple rule-based agent for the untrained baseline (no GPU needed for demo)., rule_based_action() (+1 more)

### Community 6 - "Community 6"
Cohesion: 0.39
Nodes (5): getEdgeStyle(), getNodeStyle(), getPos(), renderEdge(), renderNode()

### Community 7 - "Community 7"
Cohesion: 0.39
Nodes (5): getBadge(), getClaimType(), isVisible(), renderClaimCard(), renderVerification()

### Community 8 - "Community 8"
Cohesion: 0.25
Nodes (7): load_log(), make_demo_data(), Arms Race Visualization for ARBITER.  Generates 4 publication-quality plots:, Exponential moving average smoothing., Load JSONL reward log., Generate plausible synthetic training curves for demo/presentation., smooth()

### Community 11 - "Community 11"
Cohesion: 0.5
Nodes (3): load_trajectories(), SFT Training Script for ARBITER.  Fine-tunes Qwen 2.5 1.5B on the generated tr, Load JSONL trajectories and format as chat turns.

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (1): ARBITER — Autonomous Reasoning-Based Inspector for Training Environments with Re

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Render the observable causal graph with color-coded nodes.

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Format claims as colored HTML.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Run one episode with Claude as the Auditor. Returns list of (prompt, response) p

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Project Scaffolding

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Pitch Deck (11 Slides)

## Knowledge Gaps
- **51 isolated node(s):** `ARBITER — End-to-End Integration Test ====================================== Val`, `Parse raw_text as JSON (with regex fallback) and validate against the     ARBITE`, `Load a LoRA adapter. Returns (model, tokenizer, backend_str) or (None, None, Non`, `Run one episode driven by the LLM.     Returns (total_reward, list_of_raw_action`, `Rule-based heuristic agent — no model. Provides the floor for comparison.` (+46 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 15`** (2 nodes): `__init__.py`, `ARBITER — Autonomous Reasoning-Based Inspector for Training Environments with Re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Render the observable causal graph with color-coded nodes.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Format claims as colored HTML.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Run one episode with Claude as the Auditor. Returns list of (prompt, response) p`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Project Scaffolding`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Pitch Deck (11 Slides)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `step()` connect `Community 0` to `Community 1`, `Community 2`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.242) - this node is a cross-community bridge._
- **Why does `step_endpoint()` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `generate_trajectory()` connect `Community 4` to `Community 0`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `step()` (e.g. with `run_model_episode()` and `baseline_episode_reward()`) actually correct?**
  _`step()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ARBITER — End-to-End Integration Test ====================================== Val`, `Parse raw_text as JSON (with regex fallback) and validate against the     ARBITE`, `Load a LoRA adapter. Returns (model, tokenizer, backend_str) or (None, None, Non` to the rest of the system?**
  _51 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._