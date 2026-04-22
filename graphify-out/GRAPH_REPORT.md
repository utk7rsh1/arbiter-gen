# Graph Report - .  (2026-04-22)

## Corpus Check
- Corpus is ~9,220 words - fits in a single context window. You may not need a graph.

## Summary
- 51 nodes · 58 edges · 11 communities detected
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.84)
- Token cost: 6,400 input · 800 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Environment API|Environment API]]
- [[_COMMUNITY_Training Pipeline|Training Pipeline]]
- [[_COMMUNITY_Training Pipeline|Training Pipeline]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Causal Graph Engine|Causal Graph Engine]]
- [[_COMMUNITY_Environment API|Environment API]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]

## God Nodes (most connected - your core abstractions)
1. `new_episode()` - 6 edges
2. `format_claim_chain()` - 5 edges
3. `ARBITER System` - 5 edges
4. `Auditor Agent` - 5 edges
5. `Causal Decision Graph` - 5 edges
6. `draw_graph()` - 4 edges
7. `run_query()` - 4 edges
8. `Arms Race Dynamic` - 4 edges
9. `_get_env()` - 3 edges
10. `draw_reward_panel()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `10-Episode Validation (70/70)` --references--> `ARBITER System`  [EXTRACTED]
  TASK_LIST.md → Arbiter-IDEA.txt
- `Gradio Demo Interface` --references--> `Arms Race Dynamic`  [EXTRACTED]
  TASK_LIST.md → Arbiter-IDEA.txt

## Hyperedges (group relationships)
- **Three-Agent ARBITER System** — arbiter_idea_Auditor, arbiter_idea_Defender, arbiter_idea_MetaOverseer [EXTRACTED 1.00]
- **Training Pipeline (SFT→GRPO→Arms Race)** — arbiter_idea_SFT, arbiter_idea_GRPO, arbiter_idea_ArmsRace [EXTRACTED 1.00]
- **Three Anomaly Types** — arbiter_idea_ProxyDiscrimination, arbiter_idea_AdversarialInjection, arbiter_idea_ModelDrift [EXTRACTED 1.00]

## Communities

### Community 0 - "Environment API"
Cohesion: 0.22
Nodes (11): ARBITER System, Adversarial Injection (Type 2), Auditor Agent, Causal Decision Graph, Counterfactual Claim, Meta-Overseer Agent, Model Drift (Type 3), OpenEnv Wrapper (+3 more)

### Community 1 - "Training Pipeline"
Cohesion: 0.2
Nodes (10): Arms Race Dynamic, 7-Level Curriculum, Defender Agent, Dense Intermediate Reward, GRPO RL Training, Reward Function, SFT Behavioral Cloning, Theory-of-Mind Claim (+2 more)

### Community 2 - "Training Pipeline"
Cohesion: 0.5
Nodes (4): generate_trajectory(), main(), SFT Trajectory Generator for ARBITER.  Generates 400 training trajectories using, Run one episode with Claude as the Auditor. Returns list of (prompt, response) p

### Community 3 - "Community 3"
Cohesion: 0.67
Nodes (3): check(), Manual validation of 10 hand-crafted ARBITER episodes.  Confirms:   1. Correct c, run_validation()

### Community 4 - "Causal Graph Engine"
Cohesion: 0.5
Nodes (4): draw_graph(), draw_reward_panel(), Render the observable causal graph with color-coded nodes., run_query()

### Community 5 - "Environment API"
Cohesion: 0.67
Nodes (4): build_demo(), _get_env(), main(), new_episode()

### Community 6 - "Community 6"
Cohesion: 0.67
Nodes (3): format_claim_chain(), _format_claim_text(), Format claims as colored HTML.

### Community 7 - "Community 7"
Cohesion: 0.67
Nodes (1): ARBITER Gradio Demo Interface.  Panels:   Left:   Causal decision graph (Network

### Community 8 - "Community 8"
Cohesion: 1.0
Nodes (1): ARBITER — Autonomous Reasoning-Based Inspector for Training Environments with Re

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (1): Project Scaffolding

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (1): Pitch Deck (11 Slides)

## Knowledge Gaps
- **16 isolated node(s):** `Manual validation of 10 hand-crafted ARBITER episodes.  Confirms:   1. Correct c`, `ARBITER — Autonomous Reasoning-Based Inspector for Training Environments with Re`, `ARBITER Gradio Demo Interface.  Panels:   Left:   Causal decision graph (Network`, `Render the observable causal graph with color-coded nodes.`, `Format claims as colored HTML.` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 7`** (3 nodes): `draw_arms_race()`, `ARBITER Gradio Demo Interface.  Panels:   Left:   Causal decision graph (Network`, `app.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 8`** (2 nodes): `__init__.py`, `ARBITER — Autonomous Reasoning-Based Inspector for Training Environments with Re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 12`** (1 nodes): `Project Scaffolding`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `Pitch Deck (11 Slides)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Auditor Agent` connect `Environment API` to `Training Pipeline`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `ARBITER System` connect `Environment API` to `Training Pipeline`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **What connects `Manual validation of 10 hand-crafted ARBITER episodes.  Confirms:   1. Correct c`, `ARBITER — Autonomous Reasoning-Based Inspector for Training Environments with Re`, `ARBITER Gradio Demo Interface.  Panels:   Left:   Causal decision graph (Network` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._