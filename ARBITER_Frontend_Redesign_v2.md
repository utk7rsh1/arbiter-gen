# ARBITER — Frontend Redesign v2
## Inspired by BioSync · Research-Grade Intelligence Dashboard
> April 23, 2026 · Full redesign plan for all pages

---

## 0. The Aesthetic Translation

### What BioSync Does That Works
Look at the screenshot carefully. It isn't "dark mode with accents." It's a **hybrid** — a warm white shell holding dark feature panels. The light background makes the dark panels feel like windows into something deeper. The brain scan panel is black and glowing; the cards around it are clean white with shadows. This contrast is what creates the sense of depth and sophistication.

The other thing BioSync does: **every number feels significant**. `82/100`, `0.96`, `1.2 cm³` — all displayed large, with context underneath. No number is orphaned. Every metric has a label, a comparison, and a status dot. The dashboard doesn't just show data — it *interprets* it.

### The ARBITER Translation
ARBITER's equivalent of the brain scan is the **causal graph** — it should occupy a large dark panel and glow. The equivalent of "AI Analytics Insights" is the **live reward breakdown** and **hypothesis tracker**. The equivalent of the metric cards at the bottom are **claim accuracy stats, episode progress, and model comparison figures**.

The fundamental aesthetic principle: **light shell, dark feature panels, surgical precision in typography, color only where it means something.**

---

## 1. Design System

### Color Palette

```css
:root {
  /* Shell */
  --bg-shell:        #F0F2F5;   /* warm light gray — the outer background */
  --bg-card:         #FFFFFF;   /* white cards */
  --bg-card-hover:   #FAFBFC;
  --shadow-card:     0 2px 12px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.05);
  --shadow-card-lg:  0 8px 32px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.06);

  /* Dark Feature Panels */
  --bg-panel-dark:   #0D1117;   /* graph panel, arms race panel */
  --bg-panel-inner:  #090C14;   /* inner surfaces within dark panels */
  --bg-node-default: #161B27;   /* graph node fill */

  /* Semantic Colors — every one means something specific */
  --cyan:            #00C4E0;   /* Auditor, correct claims, selections */
  --cyan-glow:       rgba(0, 196, 224, 0.20);
  --green:           #10B981;   /* confirmed, correct verdict */
  --green-glow:      rgba(16, 185, 129, 0.20);
  --red:             #EF4444;   /* anomaly, incorrect, Defender evasion */
  --red-glow:        rgba(239, 68, 68, 0.20);
  --amber:           #F59E0B;   /* active hypothesis, queried nodes, policy */
  --amber-glow:      rgba(245, 158, 11, 0.20);
  --purple:          #8B5CF6;   /* hidden/latent nodes, theory-of-mind */
  --purple-glow:     rgba(139, 92, 246, 0.20);
  --orange:          #F97316;   /* partial credit, weakened hypothesis */

  /* Typography */
  --text-primary:    #0F172A;   /* headings, key numbers */
  --text-secondary:  #475569;   /* labels, secondary info */
  --text-muted:      #94A3B8;   /* ghost labels, hints */
  --text-on-dark:    #E2E8F0;   /* text inside dark panels */
  --text-muted-dark: #4A5568;   /* muted text inside dark panels */

  /* Borders */
  --border-light:    #E2E8F0;   /* card borders on light bg */
  --border-dark:     #1E2840;   /* borders inside dark panels */

  /* Radius */
  --radius-card:     16px;
  --radius-pill:     9999px;
  --radius-node:     50%;
}
```

### Typography

```
Display / Wordmark:  "Syne" — geometric, slightly unusual, confident
                     (Google Fonts: weights 700, 800)

Body / Labels:       "DM Sans" — clean, medical-grade, slightly warm
                     (Google Fonts: weights 300, 400, 500, 600)

Data / Monospace:    "DM Mono" — matches DM Sans family, for all numbers
                     (Google Fonts: weights 300, 400, 500)
```

Import:
```html
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
```

### Spacing & Grid
- Base unit: 8px
- Card padding: 24px
- Card gap: 16px
- Section gap: 24px
- Page padding: 32px

---

## 2. Graph Library — Upgrade to G6

**Retire `react-flow`.** It's a diagramming tool. The causal graph needs physics, fluid layout, and beautiful visual effects that react-flow cannot deliver.

**Use `@antv/g6` version 5.**

G6 is Alibaba's open-source graph visualization engine, used in production at Ant Financial for complex network graphs. It has:
- Built-in force-directed layout with physics (nodes settle naturally, not stacked)
- GPU-accelerated canvas rendering (smooth at 60fps with 100+ nodes)
- Native animated edge drawing (edges can travel, pulse, dash-animate)
- Combo grouping (group nodes by type with visual containers)
- Built-in glow effects, halos, and node state management
- Smooth camera transitions (zoom, pan with easing)

```bash
npm install @antv/g6
# or via CDN:
<script src="https://unpkg.com/@antv/g6@5/dist/g6.min.js"></script>
```

**For animations between states:** `framer-motion` for all React component animations (card entries, panel transitions, tab switches).

```bash
npm install framer-motion
```

**For all charts (reward bars, arms race, comparison):** `recharts` stays — it's the right tool for these.

---

## 3. Page Layout — Master Grid

Every page uses the same shell:

```
┌─────────────────────────────────────────────────────────────┐
│  TOPBAR — white, floating, pill nav, shadow                 │ 64px
├─────────────────────────────────────────────────────────────┤
│  CONTROLS STRIP — light gray, episode controls              │ 52px
├───────────────────────────────┬─────────────────────────────┤
│                               │                             │
│   MAIN FEATURE PANEL          │   RIGHT COLUMN              │
│   (dark — graph, arms race)   │   (white cards)             │
│                               │                             │
│                               │                             │
│   58% width                   │   42% width                 │
│                               │                             │
├───────────────────────────────┴─────────────────────────────┤
│  BOTTOM STRIP — 3 metric cards side by side                 │ 140px
└─────────────────────────────────────────────────────────────┘
```

The **light shell** sits behind everything. Cards float on it with shadows. The dark panel is inset — it feels like a window into a deeper, more dangerous layer.

---

## 4. Topbar

### Visual Spec
- Background: `#FFFFFF`, full width, `box-shadow: 0 1px 3px rgba(0,0,0,0.08)`
- Height: 64px
- Left: `ARBITER` in Syne 800, 20px, `var(--text-primary)` — with a small live status indicator to the right: a green dot (`#10B981`) that pulses (2s CSS animation, opacity 1→0.3→1)
- Center: Tab navigation as a pill group — single rounded container `border: 1px solid var(--border-light)`, background `var(--bg-shell)`, with active tab as a white filled pill with shadow that **slides** between options (the active indicator is a white background absolutely positioned, transitioning `left` with 200ms ease)
  - Tabs: `LIVE DEMO` · `ARMS RACE` · `COMPARISON` · `STATS`
- Right: Model indicator badge (shows which checkpoint is loaded) + session info in DM Mono 11px muted

### Tab Transition Animation
When switching tabs, the content area does a **crossfade with a 12px vertical slide** — outgoing content fades out while moving up 12px, incoming content fades in from 12px below. Framer Motion `AnimatePresence` with `initial={{ opacity: 0, y: 12 }}` and `exit={{ opacity: 0, y: -12 }}`. Duration 250ms, ease "easeOut".

---

## 5. Controls Strip

### Visual Spec
- Background: `var(--bg-shell)` — blends with page background, barely visible
- Height: 52px
- `border-bottom: 1px solid var(--border-light)`
- Horizontal flex, 32px horizontal padding

### Elements left to right:
1. **Level pills**: `L1` `L2` `L3` `L4` `L5` — each a pill button. Active: white background, `var(--shadow-card)`, `var(--text-primary)` text. Inactive: no background, `var(--text-muted)`. Hover inactive: `var(--bg-card)` background, 150ms transition.

2. **Separator**: 1px vertical line, `var(--border-light)`, height 20px

3. **Model selector**: Three-segment pill — same sliding active indicator as topbar nav. `UNTRAINED` / `SFT ONLY` / `FULL ARBITER`. Active segment color changes based on selection:
   - Untrained → active segment `var(--text-muted)` text, gray indicator
   - SFT Only → active segment `var(--amber)` text, amber-tinted indicator
   - Full ARBITER → active segment `var(--cyan)` text, cyan-tinted indicator

4. **Separator**

5. **Seed field**: `SEED` label in 10px DM Sans muted + number input, 52px wide, monospace, borderless inside a pill container

6. **Speed pills**: `MANUAL` `1×` `2×` `5×`

7. **Spacer (flex-grow)**

8. **Step progress**: `STEP 4 / 20` in DM Mono, with a thin progress bar below the text — the bar spans the full width of this pill, fills cyan

9. **Control buttons**: Three icon buttons — Play `▶`, Pause `⏸`, Reset `↺`. Each 36×36, rounded 8px, white background, shadow. Hover: scale(1.04) with 100ms transition.

---

## 6. Page: LIVE DEMO

This is the main page. It has four zones.

---

### 6.1 Main Feature Panel — Causal Graph (Dark, left 58%)

**The BioSync brain scan equivalent.**

#### Container
- Background: `var(--bg-panel-dark)` = `#0D1117`
- Border-radius: `var(--radius-card)` = 16px
- No border — just the dark surface against the light shell creates natural separation
- Padding: 20px
- Box-shadow: `inset 0 1px 0 rgba(255,255,255,0.04)` — a hairline internal highlight at the top edge

#### Header row (inside dark panel)
- Left: `CAUSAL GRAPH` in 10px DM Sans, letter-spacing 0.12em, `var(--text-muted-dark)` — the BioSync style: small, all-caps, dim header
- Center: Node/edge count — `17 NODES · 9 EDGES` in DM Mono 10px, muted
- Right: **JUDGE VIEW toggle** — a pill toggle switch. OFF state: dark pill with muted text. ON state: pill fills red `var(--red)` with white text, a red outer glow `box-shadow: 0 0 12px var(--red-glow)`. Transition: 200ms. This is important — when Judge View is ON, it should feel like an alert, like you've turned on x-ray vision.

#### The G6 Graph
G6 configuration for ARBITER's causal graph:

**Layout:** `force` layout with these physics params:
```javascript
layout: {
  type: 'force',
  preventOverlap: true,
  nodeSpacing: 40,
  linkDistance: 120,
  alphaDecay: 0.02,    // slow settling — nodes float gracefully into position
  velocityDecay: 0.4,
}
```
This produces a naturally spaced graph where nodes float into position over ~2 seconds when an episode loads. The settling animation is beautiful and free — you don't code it, physics produces it.

**Node visual specs in G6:**

| Node type | Shape | Fill | Border | Size | Special |
|---|---|---|---|---|---|
| Explicit input feature | circle | `#0D1420` | `#1E3A5F` 1.5px | 36px | — |
| Proxy feature (zip_code, surname) | circle | `#130D1F` | `#4C1D95` 1.5px dashed | 36px | Purple dashed border |
| Hidden/latent node | circle | `#0F0820` | `#8B5CF6` 2px | 40px | Animated halo ring |
| Decision record | rect (r=6) | `#0A1420` | `#1E3A5F` 1px | 28×20px | Small, numerous |
| Outcome node | custom hexagon | `#0A1A0F` | `#065F46` 1.5px | 48px | — |
| Policy node | diamond | `#1A1305` | `#92400E` 1.5px dashed | 40px | Amber glow |

**Hidden node halo animation** (G6 custom shape):
```javascript
// A ring that expands and fades around the hidden node
// G6 allows afterDraw on custom nodes
afterDraw(cfg, group) {
  const r = cfg.size / 2;
  const back = group.addShape('circle', {
    attrs: {
      x: 0, y: 0, r,
      fill: 'none',
      stroke: '#8B5CF6',
      opacity: 0.6,
    },
    name: 'halo'
  });
  back.animate(
    { r: r + 12, opacity: 0 },
    { repeat: true, duration: 2000, easing: 'easeCubic', delay: 0 }
  );
}
```

**Edge animations:**

Standard causal edges: G6's default cubic bezier curves, `stroke: '#1E3A5F'`, `lineWidth: 1.5`, arrows.

Anomaly edges (judge view ON):
```javascript
// Animated dash traveling along the edge — electricity effect
stroke: '#EF4444',
lineWidth: 2,
lineDash: [6, 4],
// G6 edge animate:
edge.animate({ lineDashOffset: -20 }, { repeat: true, duration: 800 })
```

Queried path (when agent traverses): stroke transitions to `#00C4E0`, `lineWidth: 2.5`, glow filter applied.

**Node state transitions** (G6 `graph.setItemState`):
- `queried`: border → amber, outer ripple animation (one-shot circle expanding from node center, 600ms)
- `claimed-correct`: border → green, fill gets very subtle green tint, one-shot pulse
- `claimed-incorrect`: border → orange
- `anomalous` (judge view): border → red, traveling dash halo ring

**On episode load:** Nodes start at the center (opacity 0, scale 0.5) and spread out via force layout over 1.5 seconds. They fade in simultaneously. This is the most beautiful moment in the whole UI — the graph assembles itself.

**Legend:** Bottom-left corner of the dark panel, small:
- Four colored dots with labels: `● Queried` (amber) `● Correct` (green) `● Incorrect` (orange) `● Anomaly` (red)
- Hidden behind a `LEGEND` pill toggle — shows/hides with a 150ms fade

---

### 6.2 Right Column (White cards, 42%)

Three cards stacked vertically.

---

#### 6.2a Claim Chain Card (top, ~55% of right column height)

**White card.** `border-radius: 16px`, `box-shadow: var(--shadow-card)`.

Header: `CLAIM CHAIN` label (DM Sans 10px, letter-spacing 0.12em, `var(--text-secondary)`) + a live counter badge: `4 CLAIMS` — filled pill, cyan background `rgba(0,196,224,0.1)`, cyan text `var(--cyan)`, DM Mono 11px.

Right of header: three filter pills — `CF` `CAUSAL` `TOM` (counterfactual / causal / theory-of-mind). Each toggleable — active: filled, inactive: ghost outline. Filters which claim types are visible.

**Claim cards inside:**

Each claim card:
- Background: `var(--bg-card)` — on the white card, these sub-cards sit on white — so give them `background: var(--bg-shell)` (the light gray), 12px border-radius, 12px padding
- Left border: 3px solid, color = claim type: causal `var(--cyan)`, counterfactual `var(--purple)`, theory-of-mind `var(--purple)` darker
- Claim type badge: small filled pill, 10px DM Sans, all-caps

Card content layout:
```
[CAUSAL]  Step 3                                    HIGH ●
zip_code_cluster_7  ──→  loan_denied
via internal_risk_score
Evidence: rec_441, rec_447, rec_512
──────────────────────────────────────────
✓ cause   ✓ effect   ✓ mechanism   ✗ confidence
```

The field verification row appears 400ms after the card enters — with a staggered reveal: each `✓` or `✗` fades in 60ms apart. Green checkmarks, red crosses. This is the tension moment.

**Counterfactual card — special layout:**
```
[COUNTERFACTUAL]  Step 5                            HIGH ●
"What if rec_441 had zip_code_cluster_3?"

PREDICTED          →          ACTUAL
APPROVED                      APPROVED   ✓ +2.0 pts
```
The predicted/actual comparison is the most visually interesting element in the whole demo. Center it with generous space. When they match: both values pulse green briefly.

**Card entry animation (framer-motion):**
```javascript
initial={{ opacity: 0, x: 20, scale: 0.98 }}
animate={{ opacity: 1, x: 0, scale: 1 }}
transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
```

After 400ms, the border color transitions: if correct → `var(--green)`, if incorrect → `var(--red)`. 300ms CSS transition. The card also gets a very subtle background tint — `rgba(16, 185, 129, 0.03)` for correct, `rgba(239, 68, 68, 0.03)` for incorrect.

**Meta-Overseer contradiction warning** — appears between two cards as a full-width bar:
- Background: `rgba(245, 158, 11, 0.08)`, border-left: 3px solid amber, 12px border-radius
- Icon: `⚠` + `META-OVERSEER: Contradiction — Step 2 ↔ Step 6 · Penalty −1.0`
- Entry: slides down from the card above it, 200ms

---

#### 6.2b Reward Breakdown Card (middle, ~25% of right column)

**White card.** This is the BioSync "AI Analytics Insights" equivalent.

Layout: Two columns inside the card.

**Left column:** The total score — displayed large like BioSync's `12%` and `30%` numbers.
- `TOTAL SCORE` label: 10px DM Sans muted, all-caps
- The number: 48px DM Mono 500, `var(--text-primary)` — but with a cyan shadow when score is high: `text-shadow: 0 0 24px var(--cyan-glow)`
- `/ 35.0 MAX` in 14px DM Mono muted below it
- When it increments: the number ticks up digit by digit (odometer animation — CSS counter animation or JS interval incrementing by 0.1). A `+2.0` delta label slides up from above and fades, in cyan.

**Right column:** Component bars — like BioSync's green-to-red gradient sliders.
Each bar:
- Label: 11px DM Sans `var(--text-secondary)`, left
- Value: 11px DM Mono `var(--text-primary)`, right
- Bar: 6px height, `border-radius: var(--radius-pill)`, track `var(--border-light)`, fill in component color
- Width animates from 0 on episode start, transitions smoothly on each update

```
Claim Reward    ████████░░░  8.0 / 12.0   [cyan fill]
Counterfactual  ████░░░░░░░  4.0 / 6.0    [purple fill]
Theory of Mind  ███████████  3.0 / 3.0    [purple fill — when earned, bar pulses]
Chain Bonus     ████████░░░  4.0 / 8.0    [green fill]
Consistency     ██░░░░░░░░░  −1.0 / −3.0  [red fill — grows rightward for penalties]
```

BioSync uses a green-to-red gradient on its sliders. For ARBITER: each bar has a single semantic color (not gradient) because each component means a different thing. Gradient would confuse the color coding.

---

#### 6.2c Hypothesis Tracker (bottom, ~20% of right column)

**Three cards side by side** in a horizontal strip — exactly like BioSync's `Neurodegenerative Risk` / `Chronic Disease Markers` pair.

Each hypothesis card:
- White background, 12px border-radius, padding 16px
- `H1` / `H2` / `H3` label: 10px DM Sans muted
- Hypothesis name: 13px DM Sans 600, `var(--text-primary)` — `PROXY DISCRIM.` / `ADV. INJECTION` / `MODEL DRIFT`
- Status indicator: colored dot + status text
  - ACTIVE: `var(--amber)` pulsing dot (2s opacity animation) + "Active" in amber
  - WEAKENED: `var(--orange)` static dot + "Weakened" in orange
  - ELIMINATED: `var(--text-muted)` dot + "Eliminated" in muted, hypothesis name gets `text-decoration: line-through`
  - CONFIRMED: `var(--green)` dot + "Confirmed" in green, card border becomes green, subtle green tint background

Status change transition: 300ms — the dot color transitions, the text transitions, the card border transitions. Not instant. Smooth.

---

### 6.3 Bottom Strip — Three Metric Cards

Horizontal row of three equal-width white cards. Sit below both the main panel and right column. BioSync's bottom cards equivalent.

**Card 1: Episode Progress**
- Large number: current episode / total target (e.g., `147 / 300`)
- A miniature sparkline below showing reward trend over last 20 episodes (recharts `AreaChart`, 60px height, no axes, cyan fill, very subtle)
- Status badge: current level indicator `● L3 ACTIVE`

**Card 2: Claim Accuracy**
- Headline: accuracy % in large DM Mono — `78%`
- Below: three mini accuracy bars for each claim type
  - `CAUSAL` ████████░░ 82%
  - `CF` ██████░░░░ 61%
  - `TOM` █████░░░░░ 54%
- Each bar fills on mount with a 600ms staggered animation

**Card 3: Budget Remaining**
- A **radial gauge** — BioSync's arc gauge (the `22` with the colored arc). This is the most visually distinctive element in the BioSync screenshot.
- G6 or a custom SVG gauge: an arc from 270° to 270° (full circle starting at top), colored segments: green (12-20 budget), amber (6-12), red (0-6)
- Center: remaining budget number, large DM Mono
- Below: `BUDGET REMAINING` label
- As budget depletes: the arc shrinks counterclockwise, color transitions from green through amber to red. 300ms transition per step.

Implementation:
```javascript
// SVG arc gauge
const gaugeArc = (value, max, r) => {
  const angle = (value / max) * 270 - 135; // -135° to +135°
  const rad = angle * Math.PI / 180;
  const x = 50 + r * Math.cos(rad);
  const y = 50 + r * Math.sin(rad);
  return `M 50 50 L ${x} ${y}`; // simplified — use stroke-dasharray for proper arc
}
// Use stroke-dasharray/stroke-dashoffset on a circle path for smooth animation
```

---

## 7. Page: ARMS RACE

Full-width dark panel — the entire content area is dark. This page should feel like a mission control room.

### Layout
- Dark panel background `var(--bg-panel-dark)` fills the content area (inside the white shell, which has `border-radius: 16px` and `margin: 16px`)
- Header: `ADVERSARIAL TRAINING DYNAMICS` in Syne 700, 18px, `var(--text-on-dark)` — left-aligned
- Sub: `500 training episodes · Levels 1–5` in DM Sans 13px, muted-dark

### The Chart

**Full width, tall** — 65% of the page height. recharts `ComposedChart`.

**Lines:**
- Auditor reward: `var(--cyan)` `#00C4E0`, lineWidth 2.5, dot={false}
- Defender evasion: `var(--red)` `#EF4444`, lineWidth 2, dot={false}, strokeDasharray="0" (solid)

**Shaded area between lines:**
- When Auditor > Defender: filled area `rgba(0, 196, 224, 0.08)` — cyan tint
- When Defender > Auditor: filled area `rgba(239, 68, 68, 0.08)` — red tint
- Use recharts `ReferenceArea` components for each region

**Inflection annotations:**
Four `ReferenceLine` components with custom label components — not just text, but small **card-style labels**: white rounded pill (on the dark bg: `background: #1E2840`, border: `var(--border-dark)`) with:
- Episode number in DM Mono 10px cyan
- One-line description in DM Sans 11px light
- A small directional arrow pointing down to the line

**Grid:** Very subtle — `strokeDasharray="2 4"`, stroke `#1E2840`, opacity 0.5. Just enough to feel like graph paper, not enough to distract.

**Axes:** Left axis cyan labels (Auditor reward), right axis red labels (Defender evasion %). Both in DM Mono 11px.

### The Replay Animation

This is the centerpiece. A `▶ REPLAY` button — white pill, shadow, top-right of the chart — when clicked:

1. Chart instantly resets: all data removed, both lines at x=0
2. `setInterval` runs at 16ms (60fps)
3. Each tick adds 2 data points to state — both lines draw simultaneously
4. The chart renders with `isAnimationActive={false}` on recharts (you're controlling animation manually via data) — this gives smooth 60fps vs recharts' janky default animation
5. When playhead crosses each inflection episode: the annotation label **fades in** (framer-motion `animate={{ opacity: 1 }}` triggered by a condition)
6. At episode 500: lines stop drawing, a subtle glow pulses along both lines once

**The shaded region** builds as the lines draw — recharts ReferenceArea components update dynamically. This means the green/red fill between lines appears in real time as the lines draw. Visual drama.

### Bottom Stats Row (below chart)

Four white stat cards (same style as bottom strip elsewhere):

| Card | Content |
|---|---|
| Peak Auditor Reward | `26.7` large, episode it occurred |
| Arms Race Equilibrium | `ep. 320` when stable, description |
| Defender Adaptations | `3` strategy shifts detected |
| ToM Emergence | `ep. 204` when theory-of-mind first paid off |

---

## 8. Page: COMPARISON

The "before vs after" page. Split screen.

### Layout

Two equal panels side by side, both white cards with slightly different header treatment:

**Left panel: UNTRAINED**
- Header badge: `BASE MODEL` in gray pill
- Dim overall — apply `filter: saturate(0.7)` to the entire panel. The untrained model's world should look slightly desaturated, slightly dead.

**Right panel: FULL ARBITER**
- Header badge: `SFT + GRPO` in cyan pill  
- Full color, full saturation

Both panels contain a **mini claim chain** (condensed card size) replaying the same episode.

**Center divider:** A thin vertical line `var(--border-light)` with a centered label: the delta score `Δ +24.4 pts` in a white pill with shadow, floating over the divider.

### Shared Timeline Scrubber

At the bottom of both panels, a single shared timeline:
- A horizontal track with episode steps as tick marks (1–20)
- A draggable thumb — dragging it steps both panels simultaneously to that step
- When scrubbing: both claim chains update in sync. A judge can drag to step 3 and compare left (vague relational claim, red card) vs right (targeted counterfactual query, green card).

Implementation:
```javascript
const [sharedStep, setSharedStep] = useState(0);
// Both panels receive sharedStep as prop and display claims[0..sharedStep]
```

**Scrubber styling:** Track: `var(--border-light)` 2px height, `border-radius: pill`. Thumb: 16px circle, white, `var(--shadow-card)`, cyan border when dragging. Fill left of thumb: `var(--cyan)` tint.

### Final Verdict Cards (visible at step 20)

When the scrubber reaches step 20, verdict cards animate in below each claim chain:

**Untrained verdict (incorrect):**
- Red background `rgba(239, 68, 68, 0.06)`, red border
- `✗ SUBMITTED: TYPE 3 (DRIFT)` — wrong anomaly type
- `TRUE ANOMALY: TYPE 1` below in muted
- Score: `2.3 pts` in gray

**Full ARBITER verdict (correct):**
- Green background `rgba(16, 185, 129, 0.06)`, green border
- `✓ SUBMITTED: TYPE 1 (PROXY DISCRIMINATION)` 
- `AFFECTED GROUP: ZIP CLUSTER 7` in muted
- Score: `26.7 pts` in green, large

The `Δ +24.4` in the center divider glows briefly when both verdict cards are visible.

---

## 9. Page: STATS

The Q&A drawer becomes a full page in this redesign.

### Layout

Three columns of white cards.

**Column 1: Three-Condition Comparison Table**
- Table styling: DM Sans 13px, header row `var(--bg-shell)`, alternating rows white/shell, no harsh borders — just `border-bottom: 1px solid var(--border-light)` between rows
- Three columns: Untrained / SFT Only / Full ARBITER
- Rows: Mean Reward / Std Dev / Verdict Accuracy / Claim Accuracy
- Best value in each row highlighted in cyan: `background: rgba(0,196,224,0.08)`, `color: var(--cyan)`, `font-weight: 500`

**Column 2: Claim Accuracy Breakdown**
- Three horizontal bar charts — one per claim type
- CAUSAL: cyan bars
- COUNTERFACTUAL: purple bars
- THEORY-OF-MIND: purple bars, darker
- Each bar animates width on page load (staggered, 100ms delay each)

**Column 3: Defender Evasion Breakdown**
- Four bars — one per obfuscation type
- `link_substitution` / `record_injection` / `proxy_laundering` / `timestamp_manipulation`
- Color: red at full evasion success, green at 0% — each bar is a gradient
- Shows which tricks still work against the trained model

**Bottom:** Episode log — a scrollable table of the last 20 episodes with: episode number, level, anomaly type, reward earned, verdict correct/incorrect. Monospace, compact, like a terminal log but white-card styled.

---

## 10. Animations — Complete Spec

### Page Load (every page)
Staggered card reveal:
```javascript
// Each card gets animation-delay: index * 60ms
initial={{ opacity: 0, y: 16 }}
animate={{ opacity: 1, y: 0 }}
transition={{ delay: index * 0.06, duration: 0.3, ease: 'easeOut' }}
```
The graph panel animates separately: fades in over 400ms, then G6 starts its force layout and nodes settle.

### Episode Start
1. Graph panel: brief flash of `#00C4E0` at 5% opacity (like a scanner activating) — 200ms
2. G6 nodes: appear at center, opacity 0 → 1 over 300ms as force layout spreads them
3. Claim chain: clears with a 150ms fade
4. Reward bars: reset to 0 with 200ms transition
5. Hypothesis cards: all reset to ACTIVE with amber border pulse
6. Budget gauge: arc resets to full (20) counterclockwise, 400ms

### Claim Entry (happens on every step)
1. Card slides in from right: 200ms, cubic-bezier(0.16, 1, 0.3, 1)
2. 400ms pause (suspense)
3. Left border color transition: 300ms to green or red
4. Subtle background tint: 300ms
5. Field verification checkmarks stagger in: 60ms each
6. Reward delta floats up from reward panel: 600ms fade-up

### Score Increment
- Number ticks up in DM Mono (requestAnimationFrame loop from old to new value, ~300ms)
- Delta label: `+2.0` appears at score, slides up 8px, fades out, 700ms total

### Tab Switch
- Framer Motion AnimatePresence
- Outgoing: `opacity: 0, y: -8`, 200ms
- Incoming: `opacity: 1, y: 0`, 200ms, 50ms delay

### Arms Race Replay
- requestAnimationFrame loop adding 2 data points per frame
- Total duration ~8 seconds for 500 episodes
- Annotation labels fade in at exact episode crossing
- Completion: `box-shadow` glow pulses on chart container once

---

## 11. Integration Points

| Training Pipeline Event | Frontend Change |
|---|---|
| SFT checkpoint ready (`lora_sft/`) | Model selector "SFT ONLY" becomes clickable (was disabled/grayed) |
| GRPO checkpoint ready (`lora_grpo/`) | "FULL ARBITER" becomes clickable |
| `evaluate.py` completes | Stats page populates with real numbers (replace placeholder `--` values) |
| Real reward curves exist | Arms Race page: swap `dataUrl` prop from `synthetic_curves.json` to `real_curves.json` |
| Contrast trajectories recorded | Comparison page: swap trajectory JSON files — zero code change |

The frontend is designed to be **fully functional with synthetic data**. Every data source is a prop or a fetched URL. No hardcoded numbers in component logic.

---

## 12. Build Order

| Priority | Component | Est. Time | Demo without it? |
|---|---|---|---|
| 1 | Design system setup (CSS vars, fonts, shell layout) | 1 hr | No |
| 2 | Topbar + controls strip | 1.5 hrs | No |
| 3 | G6 graph panel (static load + node types) | 3 hrs | No |
| 4 | G6 node state transitions + edge animations | 2 hrs | Barely |
| 5 | Claim chain card feed | 2 hrs | No |
| 6 | Reward breakdown card + gauge | 1.5 hrs | Yes |
| 7 | Hypothesis tracker cards | 1 hr | Yes |
| 8 | Bottom metric strip | 1 hr | Yes |
| 9 | Arms Race page + replay animation | 2.5 hrs | Yes |
| 10 | Comparison page + scrubber | 2 hrs | Yes |
| 11 | Stats page | 1 hr | Yes |
| 12 | Page transition animations + polish | 1.5 hrs | Yes |

**Total: ~20 hours.** Items 1–7 (10 hours) give you a complete, impressive LIVE DEMO page. Items 8–12 make it a 10/10 across all pages.

---

## 13. The One Thing That Makes It Unforgettable

The **graph assembling itself on episode load**. G6's force layout with the physics parameters above means nodes start clustered at the center and spread outward over 1.5 seconds, settling into their positions. This happens automatically — you don't animate it, physics does. The hidden node (purple, pulsing halo) drifts into position last because it has fewer connections pulling it. The policy node (amber diamond) settles in the center of the graph.

Judges will watch this happen and feel the complexity of the system before a single step is taken. The graph *is* the AI system. Watching it assemble is watching a world being born.

That moment, plus the claim card's 400ms pause before verdict, plus the arms race drawing itself — those three things are what they will describe to their colleagues the next day.

---

*This plan supersedes v1. All component specs are designed to connect to the existing FastAPI backend at `arbiter/server.py` without modification. G6 replaces react-flow entirely. framer-motion handles all React component transitions. recharts handles all charts.*
