# ARBITER — Frontend Revamp Implementation Plan
> Written: April 23, 2026 · For demo on April 25–26
> Audience: Meta engineers, AI researchers, hackathon judges
> Stack: React + Tailwind + react-flow + recharts · Connects to existing FastAPI at `arbiter/server.py`

---

## 0. Strategic Direction

### Who Is Watching
Meta engineers ship production ML infrastructure daily. They will not be impressed by a clean Gradio form. They will be impressed by something that looks like it belongs in an internal research dashboard — dense, purposeful, no wasted pixels, everything carrying meaning.

### Aesthetic Direction
**Research Operations Terminal.** Think Bloomberg Terminal meets internal Meta AI tooling. Dark background (`#0A0A0F`), near-black panels with razor-thin borders, a single electric accent color (`#00E5FF` — cyan) for things that matter, red (`#FF3B3B`) for anomalies and penalties, green (`#00FF87`) for confirmations. Monospace font (`JetBrains Mono`) for all data. Sharp sans-serif (`IBM Plex Sans`) for all labels. Zero gradients except on the arms race chart. Zero rounded corners except on claim cards. Everything should feel like it was built by engineers for engineers.

### The One Thing They Will Remember
The **live causal graph** with the anomaly hidden in plain sight — visible to judges, invisible to the agent — and watching it get found in real time. That moment where the claim card flashes green and the red edge in the graph lights up because the agent just named it: that is the demo. Everything else supports that moment.

---

## 1. Tech Stack

| Need | Library | Why |
|---|---|---|
| Framework | React 18 | Component model, state management |
| Styling | Tailwind CSS (CDN) | Utility-first, no build step needed |
| Graph rendering | `react-flow` | Purpose-built for node/edge graphs, handles zoom/pan/dynamic updates natively |
| Charts | `recharts` | Clean API, animatable, composable |
| Fonts | Google Fonts: `JetBrains Mono`, `IBM Plex Sans` | Research terminal aesthetic |
| Backend connection | `fetch` to FastAPI | Already running at `arbiter/server.py` |
| State | React `useState` + `useReducer` | No external state library needed at this scale |

**Do not use:** Gradio's built-in layout system, any component library (MUI, Chakra, shadcn), Chart.js, D3 directly.

---

## 2. File Structure

Replace `arbiter/demo/app.py` with a proper React frontend. Keep the FastAPI server intact — the frontend talks to it via REST.

```
arbiter/demo/
├── frontend/
│   ├── index.html              ← single HTML shell, loads React via CDN
│   ├── App.jsx                 ← root component, layout, global state
│   ├── components/
│   │   ├── EpisodeControls.jsx      ← top bar: model selector, level, seed, speed
│   │   ├── CausalGraph.jsx          ← react-flow graph panel
│   │   ├── ClaimChain.jsx           ← scrollable claim card feed
│   │   ├── RewardPanel.jsx          ← live reward breakdown bars
│   │   ├── HypothesisTracker.jsx    ← three hypothesis status cards
│   │   ├── ArmsRaceChart.jsx        ← animated dual-line chart
│   │   ├── ContrastPanel.jsx        ← untrained vs trained side-by-side
│   │   └── StatsDrawer.jsx          ← collapsible Q&A stats sidebar
│   └── hooks/
│       ├── useEpisode.js            ← episode state machine, step logic
│       └── useBackend.js            ← all fetch calls to FastAPI
├── app.py                      ← keep for --checkpoint loading, serve frontend as static
```

---

## 3. API Endpoints You Will Call

These all exist in `arbiter/server.py` already. Confirm each one before building against it.

| Endpoint | Method | What it returns | Used by |
|---|---|---|---|
| `/session/new` | POST | `session_id`, initial graph JSON | Episode start |
| `/session/{id}/step` | POST | action result, claim verification, reward delta | Each step |
| `/session/{id}/render` | GET | full graph state with node colors | Graph panel |
| `/session/{id}/state` | GET | hypothesis flags, budget, step count | Controls |
| `/metrics` | GET | aggregate stats across all sessions | Stats drawer |

**One thing to verify:** Does `/session/{id}/render` return node-level state (queried/correct/incorrect) or just the raw graph? If only raw graph, you need to track node color state client-side from the step responses.

---

## 4. Component-by-Component Build Plan

---

### 4.1 Episode Controls Bar (Build first — nothing works without it)

**What it is:** A full-width top bar. The cockpit of the demo.

**Elements:**
- `ARBITER` wordmark — left, monospace, cyan, small caps
- Level selector — pill buttons: `L1` `L2` `L3` `L4` `L5` — active level in cyan, others in dim gray
- Model selector — three-way toggle: `UNTRAINED` | `SFT ONLY` | `FULL ARBITER` — this determines which checkpoint `app.py` loads for inference
- Episode seed — a number input field, monospace, labeled `SEED`
- Speed selector — `MANUAL` | `1×` | `2×` | `5×`
- Step / Pause / Reset — three icon buttons, right-aligned
- Progress bar — thin, full-width, along the bottom edge of the bar. Fills cyan as steps complete. Shows `STEP 4 / 20` as text overlay

**State it owns:**
```javascript
{ level, modelMode, seed, speed, currentStep, maxSteps, isRunning }
```

**Implementation note:** The model selector toggle is cosmetic at this stage — it tells `app.py` via a query param which checkpoint to load when a new session is created. Wire: `POST /session/new?checkpoint=lora_grpo` vs `?checkpoint=base`.

---

### 4.2 Causal Graph Panel (Build second — the centrepiece)

**What it is:** Left panel, ~45% of viewport width, full height below the controls bar. The causal decision graph rendered as an interactive node-edge diagram.

**Node visual spec:**

| Node type | Shape | Color | Size |
|---|---|---|---|
| Input feature (explicit) | Circle | `#1A1A2E` with `#4A4A6A` border | 40px |
| Input feature (proxy — e.g. zip_code) | Circle | `#1A1A2E` with `#6A4A6A` border, dashed | 40px |
| Hidden/latent node | Circle | `#1A0A2E` with `#8B00FF` border, pulsing glow | 44px |
| Decision record | Rounded square | `#0D1B2A` with `#1E3A5F` border | 36px |
| Outcome node | Hexagon | `#0A1A0A` with `#1A4A1A` border | 52px |
| Policy node | Diamond | `#1A1A0A` with `#4A4A1A` border, dashed | 48px |

**Node state changes (real-time, as agent acts):**

| State | Visual change |
|---|---|
| Default | Colors as above |
| Queried by agent | Border turns yellow `#FFD700`, subtle outer glow |
| Correct claim made | Border turns green `#00FF87`, glow intensifies for 0.5s then settles |
| Incorrect claim made | Border turns orange `#FF6B00` |
| Part of anomaly chain (judge view only) | Border turns red `#FF3B3B`, fill slightly red-tinted |

**Edge visual spec:**

| Edge type | Style | Color |
|---|---|---|
| True causal edge | Solid arrow | `#2A4A6A` |
| Policy edge (claimed) | Dashed arrow | `#4A4A2A` |
| Anomalous edge (judge view) | Solid arrow, animated dash | `#FF3B3B` |
| Queried path | Highlighted | `#00E5FF` with glow |

**The Judge Toggle:**
A small button in the top-right corner of this panel: `[ JUDGE VIEW: OFF ]`. When turned on, anomalous edges and hidden nodes reveal their true color. When off, the graph looks exactly as the agent sees it. During the pitch: start with judge view ON so judges can see what the agent is hunting for. Then turn it off to show the agent's perspective. Then watch it find the anomaly anyway.

**Implementation:**
```javascript
// react-flow node with dynamic styling
const nodeTypes = { feature: FeatureNode, decision: DecisionNode, ... }

// Update node state from step response
const onStepResult = (result) => {
  setNodes(nodes => nodes.map(n =>
    n.id === result.queried_node
      ? { ...n, data: { ...n.data, state: 'queried' } }
      : n
  ))
}
```

Feed initial graph from `/session/{id}/render`. Update node states from each `/step` response.

---

### 4.3 Claim Chain Panel (Build third)

**What it is:** Right side, top ~55% of the right column. A scrollable vertical feed of claim cards appearing one by one as the agent acts.

**Claim card anatomy:**

```
┌─────────────────────────────────────────────┐
│ [CAUSAL]  Step 3   HIGH confidence          │
│ zip_code_cluster_7 ──→ loan_denied          │
│ via internal_risk_score                     │
│ Evidence: rec_441, rec_447, rec_512         │
│ ✓ cause_feature  ✓ effect  ✓ mechanism      │  ← per-field verification
└─────────────────────────────────────────────┘  ← border: green if correct, red if not
```

**Claim type badges:**
- `CAUSAL` — blue badge
- `COUNTERFACTUAL` — cyan badge (special — this is ARBITER's signature move)
- `THEORY-OF-MIND` — purple badge (only appears Level 4+)

**Counterfactual card — special layout:**
```
┌─────────────────────────────────────────────┐
│ [COUNTERFACTUAL]  Step 5   HIGH             │
│ If rec_441 had zip_code_cluster_3...        │
│ Predicted: APPROVED    Actual: APPROVED ✓   │  ← side by side
│ Reward: +2.0                                │
└─────────────────────────────────────────────┘
```

**Meta-Overseer contradiction warning — appears between two claim cards:**
```
⚠  META-OVERSEER: Contradiction detected
   Step 2 asserts zip_code → risk_score
   Step 6 asserts risk_score → zip_code
   Penalty: −1.0
```
Rendered as a full-width warning bar between the conflicting cards, amber background.

**Animation:** Each card slides in from the right (CSS `translateX` transition, 200ms). Border color appears with a 300ms flash before settling. This draws the eye at the exact right moment without being distracting.

---

### 4.4 Reward Breakdown Panel (Build fourth)

**What it is:** Right side, bottom ~30% of the right column. Live reward accounting.

**Layout:**
- Large running total, top center: `24.3 pts` in cyan, monospace, large font
- Below: a horizontal bar for each reward component
- Each bar fills proportionally to its max possible value
- Penalties render as red bars going left from a center zero line

```
Claim Reward     ████████████░░░  8.0 / 12.0
Counterfactual   ████░░░░░░░░░░░  4.0 / 6.0
Theory of Mind   ███████████████  3.0 / 3.0  ← this one pulses when earned
Chain Bonus      ████████░░░░░░░  4.0 / 8.0
Consistency      ██░░░░░░░░░░░░░  −1.0        ← red bar
Budget Eff.      ░░░░░░░░░░░░░░░  (end only)
Verdict          ░░░░░░░░░░░░░░░  (end only)
```

**Delta popup:** When a reward component is earned, a small floating label appears next to the bar: `+2.0` fades in, moves up 10px, fades out over 800ms. Pure CSS animation.

---

### 4.5 Hypothesis Tracker (Build fifth)

**What it is:** Three cards, always visible, positioned either below the graph panel or as a horizontal strip above the claim chain. Shows the agent's current theory of the case.

**Card states:**

| Status | Visual |
|---|---|
| `ACTIVE` | Yellow border, glowing, label in amber |
| `WEAKENED` | Orange border, dimmed |
| `ELIMINATED` | Gray, crossed-out label, opacity 40% |
| `CONFIRMED` (final verdict) | Green border, pulsing, bold |

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   H1             │  │   H2             │  │   H3             │
│ PROXY DISCRIM.   │  │ ADV. INJECTION   │  │ MODEL DRIFT      │
│                  │  │                  │  │                  │
│   ● ACTIVE       │  │   ○ WEAKENED     │  │   ✗ ELIMINATED   │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

Updates in real time from `FLAG_HYPOTHESIS` actions in the step response.

---

### 4.6 Arms Race Chart — Separate Tab (Build sixth)

**What it is:** A full-screen chart tab. Accessible via tab navigation in the top bar: `LIVE DEMO` | `ARMS RACE` | `COMPARISON` | `STATS`

**Chart spec:**
- Two lines: Auditor reward (cyan `#00E5FF`) and Defender evasion rate (red `#FF3B3B`)
- X-axis: Training episodes (0–500)
- Y-axis left: Auditor reward (0–30)
- Y-axis right: Defender evasion % (0–100%)
- Shaded area between lines: green when Auditor > Defender, red when Defender > Auditor
- Four `ReferenceLine` annotations at inflection episodes with text labels directly on chart

**The play-through animation — critical for the pitch:**
A `▶ REPLAY` button in the top-right corner. When clicked:
1. Chart resets to episode 0
2. Data points are fed in via `setInterval` — 10 episodes per frame at 60fps
3. Both lines draw themselves simultaneously over ~8 seconds
4. Annotation labels fade in as the playhead crosses each inflection point
5. At the end, a subtle glow pulses on both lines

This is what you show during the 90-second demo section. The chart drawing itself while you narrate is more compelling than a static image.

**Data source:** Load from `results/plots/` JSON (synthetic now, real curves after GRPO). The chart component should accept a `dataUrl` prop so swapping synthetic → real data is a one-line change.

---

### 4.7 Untrained vs Trained Contrast Panel — Separate Tab (Build seventh)

**What it is:** The `COMPARISON` tab. The headline result made visual.

**Layout:** Side by side, left = Untrained, right = Full ARBITER

```
┌────────────────────────┐    ┌────────────────────────┐
│  UNTRAINED             │    │  FULL ARBITER          │
│  Base Qwen 2.5 1.5B    │    │  SFT + GRPO            │
│                        │    │                        │
│  [claim chain replay]  │    │  [claim chain replay]  │
│                        │    │                        │
│  Final Score:  2.3     │    │  Final Score:  26.7    │
│  Verdict: WRONG ✗      │    │  Verdict: CORRECT ✓    │
└────────────────────────┘    └────────────────────────┘
                    Δ + 24.4 pts
```

**Replay controls:** A single shared timeline slider at the bottom. Dragging it steps through both trajectories simultaneously — same step number, different choices. This lets a judge scrub to step 3 and see: left side makes a vague relational claim, right side immediately fires a counterfactual query.

**Implementation note:** Both trajectories are pre-loaded from stored JSON (pinned seed). No live inference needed here — just replaying recorded data. This means this panel works even if the live demo is having issues.

---

### 4.8 Stats Drawer — Collapsible (Build last)

**What it is:** A right-side drawer that slides in when a `STATS` button is clicked. For Q&A only — hidden during main demo.

**Contents:**
- Three-condition comparison table: Untrained / SFT Only / Full ARBITER
  - Mean reward ± std deviation
  - Verdict accuracy %
  - Claim accuracy %
- Claim accuracy by type: Causal / Counterfactual / Theory-of-Mind
- Defender evasion by obfuscation type: link_substitution / record_injection / proxy_laundering / timestamp_manipulation
- Total training episodes completed, current level, advance threshold

All numbers pull from `/metrics` endpoint. Refresh button to re-fetch live.

---

## 5. Global State Architecture

```javascript
// App.jsx — top-level state
const [session, setSession] = useState(null)        // current session_id
const [graphState, setGraphState] = useState(null)  // node/edge data
const [claims, setClaims] = useState([])            // all claims so far
const [reward, setReward] = useState({              // reward component breakdown
  claim: 0, counterfactual: 0, tom: 0,
  chain: 0, consistency: 0, budget: 0, verdict: 0
})
const [hypotheses, setHypotheses] = useState({
  h1: 'ACTIVE', h2: 'ACTIVE', h3: 'ACTIVE'
})
const [step, setStep] = useState(0)
const [isRunning, setIsRunning] = useState(false)
const [activeTab, setActiveTab] = useState('LIVE')  // LIVE | ARMS_RACE | COMPARISON | STATS
```

Each `/step` response updates: `graphState`, `claims`, `reward`, `hypotheses`, `step`.

---

## 6. Integration Points With Training Pipeline

These are the exact places where the frontend needs to be ready to accept real data as Kabir's training pipeline completes:

| Training milestone | What changes in frontend | How to wire |
|---|---|---|
| SFT checkpoint (`lora_sft/`) done | Model selector "SFT ONLY" option becomes active | Pass `?checkpoint=lora_sft` to `/session/new` |
| GRPO checkpoint (`lora_grpo/`) done | Model selector "FULL ARBITER" becomes active | Pass `?checkpoint=lora_grpo` to `/session/new` |
| `evaluate.py` runs | Stats drawer shows real numbers instead of placeholders | `/metrics` endpoint auto-updates |
| Real reward curves available | Arms race chart loads real data | Swap `dataUrl` prop from synthetic JSON to real JSON |
| Contrast panel trajectories recorded | Both replays show real agent behavior | Replace stored JSON files for both trajectories |

**The frontend should be fully functional with fake/synthetic data from day one.** Real data slots in without code changes — only data file swaps and a checkpoint path change.

---

## 7. Build Order and Time Estimates

Given you're working right now (April 23) and demo is April 25–26:

| Order | Component | Estimated time | Can demo without it? |
|---|---|---|---|
| 1 | Episode controls bar + backend connection | 2 hrs | No |
| 2 | Causal graph panel (static first, then dynamic) | 3 hrs | No |
| 3 | Claim chain feed | 2 hrs | No |
| 4 | Reward breakdown panel | 1.5 hrs | Yes |
| 5 | Hypothesis tracker | 1 hr | Yes |
| 6 | Arms race chart + play animation | 2 hrs | Yes (show PNG fallback) |
| 7 | Contrast panel with replay | 2 hrs | Yes (describe verbally) |
| 8 | Stats drawer | 1 hr | Yes |

**Total: ~14.5 hours.** Spread across April 23–25 this is very doable.

Stop after item 5 if time is short. Items 1–5 give you a fully functional live demo. Items 6–8 are what takes it from good to memorable.

---

## 8. Serving the Frontend

The existing `app.py` (Gradio) gets replaced. The FastAPI server in `arbiter/server.py` serves the React frontend as static files:

```python
# In arbiter/server.py — add these lines
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="arbiter/demo/frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("arbiter/demo/frontend/index.html")
```

The `--checkpoint` argument that `app.py` previously handled gets passed as a startup environment variable instead:

```bash
ARBITER_CHECKPOINT=lora_grpo python -m arbiter.server
```

This means the existing `integration_test.py` Stage 4 (subprocess launch + port polling) still works with zero changes.

---

## 9. The Demo Script — Mapped to UI

This is the exact sequence to run during the 90-second demo, mapped to which UI element is active:

| Time | Action | UI element active |
|---|---|---|
| 0:00 | Set Judge View ON, show graph with hidden red anomaly | Causal graph |
| 0:10 | "The agent can't see this red chain. Watch." | Causal graph |
| 0:15 | Switch model to UNTRAINED, hit Step through 8 steps | Claim chain going red |
| 0:35 | Show final wrong verdict, reward: 2.3 | Reward panel |
| 0:40 | Reset, switch to FULL ARBITER | Episode controls |
| 0:45 | Hit Step — watch counterfactual query fire on step 1 | Claim chain — cyan COUNTERFACTUAL card |
| 1:00 | Theory-of-Mind card appears — +3.0 bonus | Reward panel delta popup |
| 1:10 | Correct verdict submitted, reward 26.7 | Reward panel, green verdict |
| 1:20 | Switch to ARMS RACE tab, hit Replay | Arms race chart drawing itself |
| 1:40 | Let it finish — point to inflection at episode 150 | Arms race chart |
| 1:50 | "We didn't program this. The environment produced it." | — |

---

## 10. One Thing That Must Not Be Wrong

The untrained model run and the trained model run in the contrast panel **must use the exact same episode seed.** If they run on different episodes, a judge could reasonably ask "did you just pick an easy case for the trained model?" Pin the seed. Hardcode it if necessary. The entire before/after story depends on same case, different agent.

---

*This plan is complete as of April 23, 2026. All component specs are designed to connect to the existing FastAPI backend without changes. Real training data slots in via data file swaps and a checkpoint path change — no frontend code changes required when Kabir's pipeline completes.*
