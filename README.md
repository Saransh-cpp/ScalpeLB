# ScalpeLB

Inertial Local-Search Load Balancing of MoEs on GPUs.

Built for
[MOE Dynamic Load Balancing Competition](https://www.codabench.org/competitions/16778/#/pages-tab)
by [LauzHack](https://lauzhack.com/2026-huawei/info) and
[Huawei](https://moe-competition-docs.readthedocs.io).

## Introduction

Builds on top of (and improves) [DeepSeek's EPLB](https://github.com/deepseek-ai/EPLB)
algorithm to optimize the placement of experts on GPUs in terms of:

- PAR: Peak-average ratio of device loads
- Transit: the number of experts that move to a different GPU

Instead of recomputing the whole layout every cycle (and paying that transit every
cycle), ScalpeLB establishes a good variance-aware placement ONCE, and then
_maintains_ it with a small bounded number of surgical swaps per layer.

- Cycle 1: variance-aware placement (mean + K\*std replication + balanced packing,
  recency-weighted on layers whose traffic regime shifts mid-window), aligned to the
  seed -> one-time placement cost.
- Cycle 2+ (per layer): try cheap bounded swap-maintenance (relocate hottest expert
  off the hottest GPU onto the coldest GPU, <= BUDGET swaps). Keep it only if its
  PAR stays within DRIFT_TOL of a fresh variance placement; otherwise the layer
  has DRIFTED (frozen replica counts went stale) -> re-place it fully.

The full-repack (EPLB) approach re-pays the entire placement transit every cycle
and its greedy packer churns even stable experts. ScalpeLB pays placement once and
only moves what reduces the max load -> far less transit, PAR no worse (often better).
The drift guard is essential: pure swap-maintenance freezes replica counts, which blows
up PAR on non-stationary / mixed traces (a newly-hot expert can't get more copies); the
per-layer re-place fixes that while keeping low transit on the stable majority. Swaps and
re-placement both preserve coverage -> no NaN-PAR risk.

## Results

ScalpeLB achieves a mean PAR of 1.71 with 638465 total transits on a hidden test set
composed of real-world MoE traces from multiple frontier models - ShareGPT (Qwen3),
ShareGPT (DeepSeek R1), LmSys(Qwen3), LmSys (DeepSeek R1), WildChat (Qwen3), WildChat
(DeepSeek R1), Mix (DeepSeek R1) - collected by Huawei. Furthermore, the algorithm
achieved a
[composite score](https://moe-competition-docs.readthedocs.io/en/latest/competition-context.html#composite-score)
of 110.94 (beating DeepSeek EPLB's composite score - 100).

The algorithm was placed 8th (out of 41 participants and 26 submissions) on a
[very tight final leaderboard](https://www.codabench.org/competitions/16778/#/results-tab):

<img width="1165" height="606" alt="image" src="https://github.com/user-attachments/assets/9aaba33e-6f9e-4eb8-a7bd-fabbb46b216d" />

Moreover, ScalpeLB's variance-hedged inertia natively excels on highly-skewed,
stationary traffic (e.g., LmSys (DeepSeek R1)). It inherently struggles on uniform-load
architectures (e.g., Qwen3, where variance hedging over-provisions) and on violent
context-switching traces (e.g., Mix (DeepSeek R1), where layout inertia becomes a
liability), requiring dynamic fallbacks to plain-sum global repacking.

## Algorithm

The state machine runs once per routing cycle. Cycle 1 builds a placement from
scratch; every cycle after that tries to _keep_ that placement and only repairs the
layers that have gone stale. Each numbered step below maps to a function in
[scalpelb.py](scalpelb.py).

### Step 0: Variance-aware planning weight (`_placement_weight`)

The input `hotness` is a rolling window of per-expert traffic, shape
`[window, n_layers, n_experts]`. Two summaries are derived from it:

- `weight`: the **planning** signal that drives placement. By default it is
  `mean(window) + K * std(window)` — adding `K * std` over-provisions experts whose
  load is _spiky_, not just high on average, so a bursty expert gets extra replicas
  before it melts a GPU. (Refined per-layer by shift-gated recency, below.)
- `real = sum(window)`: the **measurement** signal (raw observed load), used only
  to score placements in the drift check so the guard reacts to what actually
  happened, not to the hedged estimate.

`K` is auto-selected by expert count: variance hedging helps high-count models
(`K = 2.0` for `n_experts >= 192`, e.g. DeepSeek-R1's 256) but hurts low-count,
more-uniform models (`K = 0.0` for Qwen3-style 128), where it scored worse than
plain mean.

**Shift-gated recency.** A uniform `mean` over the window lags at a context flip:
when the traffic regime changes mid-window (the "Mix" trace), a newly-hot expert is
diluted by the stale early steps, ends up under-provisioned, and PAR explodes. To
catch this per layer, the window is split into halves and the **Total-Variation
distance** between the two halves' _normalized_ load distributions is measured
(`tv = 0.5 * Σ|new − old|`, in `[0, 1]`). TV is volume-invariant, so ordinary
stationary jitter doesn't trip it (measured ceiling ≈ 0.143). On the layers where
`tv > SHIFT_TV`, the planning weight switches to a **linear recency ramp** — a
time-weighted mean + `K`·(weighted std) that leans on the most recent steps — so the
new regime is provisioned immediately. Layers that haven't shifted (and the entire
stationary case) keep the exact uniform `mean + K*std`, so the refinement is zero-cost
when nothing has changed.

### Step 1: One-time global placement (cycle 1 only)

This is a NumPy reimplementation of DeepSeek's EPLB, run once to get a strong
starting layout, then aligned to the running table so the first move is cheap.

1. **Replication** (`_replicate`): start with one physical slot per logical expert,
   then hand out the remaining redundant slots greedily, such that each goes to the expert
   with the highest _per-replica_ load (`weight / current_replica_count`). Hot
   experts accumulate copies; the load each copy carries shrinks as copies are added.
2. **Balanced packing** (`_balanced_packing`): sort all physical replicas by load
   (descending) and drop each into the lightest GPU that still has a free slot. This
   is a longest-processing-time bin-packing heuristic that levels the per-GPU totals.
3. **Minimal-transit alignment** (`_match_devices` + `_align_layer`): the ideal
   layout above is a _set_ of experts per GPU; their physical slot positions are
   arbitrary. To avoid moving experts that are already in the right place, each
   current GPU is matched to the ideal pack it overlaps with most (greedy
   max-overlap), then experts already present are pinned to their slots and only the
   leftover slots are filled. Same balanced placement, far fewer experts in transit.

### Step 2: Bounded swap-maintenance (cycle 2+)

For every layer, ScalpeLB first tries to repair the _existing_ layout in place
rather than rebuild it.

1. **Scalpel swap** (`_scalpel_layer`): up to `BUDGET` times, find the hottest and
   coldest GPU, and swap that GPU pair's hottest and coldest expert, but only if the
   swap strictly lowers the peak load. It stops early the moment no swap helps
   (usually after ~2 swaps), so the transit cost is tiny and bounded.
2. **Drift guard** (`_layer_par`): replica _counts_ are frozen during swaps, so a
   newly-hot expert can't get more copies. A layer is flagged **drifted** when its
   maintained PAR exceeds a fresh placement's PAR by more than `DRIFT_TOL`. Drifted
   layers are re-placed via Step 1's alignment (new replica counts); everyone else
   keeps the cheap swaps.
3. **Heavy-drift escape hatch** (`HEAVY_FRAC`): if more than `HEAVY_FRAC` of layers
   drift in a single cycle, the trace is in a non-stationary / mixed regime where
   per-layer patching can't keep up, so _every_ layer is re-placed for that cycle.

### Coverage & safety

Both swaps and re-placement only ever permute or recount experts that are already
deployed, so every logical expert keeps at least one replica - `_layer_par` never
divides by zero and PAR never goes NaN. `rebalance` also wraps the whole step in a
try/except that returns a trivially valid round-robin layout (`_safe_fallback`) if
anything throws, so a single bad cycle can never crash the serving loop.

### Parameters

ScalpeLB has five knobs, all balancing the two competing objectives - **PAR**
(balance quality) against **Transit** (experts moved). The defaults below were found
by an offline grid sweep over the competition traces (the value in each grid cell
scored by PAR and transit), so they are the best static operating point for _those_
traces; the next section covers how a production system would set them dynamically
instead.

| Parameter    | What it controls                                                                                                         | Default          | Turn it **up** when…                                                   | Turn it **down** when…                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------ | ---------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `K`          | Variance buffer in the planning weight `mean + K*std`; controls how aggressively spiky experts are over-replicated.      | auto (2.0 / 0.0) | Traffic is bursty/chaotic and you have spare replica slots.            | Traffic is smooth/uniform, or redundant slots are scarce.                  |
| `SHIFT_TV`   | TV-distance threshold for detecting a within-window load-distribution shift; gates the recency-weighted planning weight. | 0.2              | Only dramatic regime flips should trigger recency; ignore minor drift. | You want to react to subtler within-window shifts (risk: firing on noise). |
| `BUDGET`     | Max swaps per layer per cycle; the hard ceiling on transit.                                                              | 8                | Interconnect is idle; you want the tightest possible balance.          | Interconnect is congested; minimize experts in flight.                     |
| `DRIFT_TOL`  | How far a maintained layer's PAR may exceed a fresh placement's before it's re-placed.                                   | 0.2              | You favor stability/low transit over chasing every PAR gain.           | You need PAR held tight and can afford the extra re-placements.            |
| `HEAVY_FRAC` | Fraction of drifted layers that triggers a full re-place of every layer that cycle.                                      | 0.5              | Drift is usually localized; keep patching as long as possible.         | Traffic shifts globally; bail to a full rebuild sooner.                    |

`K` is auto-selected from `n_experts` (`K_HIGH = 2.0` at or above `K_SWITCH_NE = 192`,
`K_LOW = 0.0` below) unless passed explicitly to `ScalpelBalancer`. `BUDGET` converges
early in practice (~2 swaps); the default of 8 just leaves headroom for drift cycles.
`SHIFT_TV = 0.2` sits just above the measured stationary TV ceiling (~0.143), so it is
provably silent on stationary traces and fires only on genuine regime flips; set it
above 1 to disable shift-gated recency entirely.

#### Tuning in production

These five constants are deliberately exposed as `ScalpelBalancer` constructor
arguments rather than buried in the algorithm. In a live hyperscale system (the kind
of infrastructure serving DeepSeek-R1 or GPT-4) they would **not** be static. They
are exactly the surface a telemetry-driven control plane tunes online. ScalpeLB
already ships a primitive version of this: `K` is auto-selected from the model's
expert count, and `SHIFT_TV` already reacts to live regime shifts per layer. A full
controller would go further, recomputing the knobs per window from live signals:

- **Dynamic `K` (variance buffer).** A background service watches the burstiness of
  the last few minutes of routing traffic (the same `hotness` stream the balancer
  consumes). Chaotic, spiking query mixes dial `K` **up** to pre-provision hot experts
  before they melt a GPU; smooth traffic dials it **down** to reclaim scarce replica
  slots.
- **Dynamic `BUDGET` (transit ceiling).** The controller reads interconnect health.
  When the GPU fabric is congested by other workloads it lowers `BUDGET` to 1–2 swaps
  so rebalancing doesn't compete for bandwidth; when the fabric is idle it raises it
  (~10) to balance compute as tightly as possible.
- **`DRIFT_TOL` / `HEAVY_FRAC` (SLA / churn).** These map naturally onto a serving
  SLA: tighten them when the deployment must hold a strict tail-latency target, loosen
  them to prioritize layout stability and minimize migration churn.

In other words, the static defaults in this repo are best understood as a single
hand-tuned operating point of a control loop that, in production, would be closed.

## Usage

The primary entry point is the `rebalance` function. It is designed to be called
sequentially at each routing cycle in a stateful loop.

```python
import numpy as np
from scalpelb import rebalance

# 1. Define hardware constraints
n_device = 8         # Number of physical GPUs available
n_red_expert = 16    # Total redundant expert replica slots to allocate

# 2. Collect rolling traffic/routing frequencies from the cluster telemetry
# Shape: [window_size, num_layers, num_experts]
hotness_window = np.random.rand(10, 58, 256)

# 3. Compute optimal cycle placement
# The internal state machine persists across calls to minimize transit
success, layer_indices, placement, _ = rebalance(
    hotness=hotness_window,
    n_device=n_device,
    n_red_expert=n_red_expert
)

# `placement` shape: [num_layers, num_gpus, experts_per_gpu]
```

## Future work

The knobs are currently set by an **offline grid sweep** over the competition
traces, which was exhaustive, static, and tied to the traces it was run on (`SHIFT_TV`
is the exception — it is pinned just above the measured stationary noise floor rather
than swept). The "Tuning in production" section above describes the first step beyond
that: a control plane that sets the knobs from live telemetry instead of a frozen sweep
result. The natural next step is to make that loop **learned** rather than
hand-engineered: instead of either a grid sweep or hand-written telemetry → knob rules,
train a small controller (telemetry features → `K`, `SHIFT_TV`, `BUDGET`, `DRIFT_TOL`,
`HEAVY_FRAC`) by gradient descent on a `PAR + λ * Transit` objective measured over
incoming traffic.

The obstacle is that the current pipeline is **non-differentiable**; hence, replication,
balanced packing, and the scalpel swaps are all built from `argmax` / `argsort` /
greedy selection, which have no usable gradient. Making the objective differentiable
end-to-end would require relaxing those discrete steps, e.g.:

- soft, temperature-controlled assignment (softmax / Gumbel-softmax) in place of the
  hard `argmax` replication and swap choices;
- a Sinkhorn / optimal-transport relaxation of the balanced-packing assignment;
- straight-through estimators so the forward pass stays discrete (and valid) while
  gradients still flow to the controller.

With a differentiable surrogate of PAR/Transit, the knobs, or a controller network
predicting them per window, could be trained online against the live `hotness`
stream, closing the loop the production section only sketches. This is exploratory and
not implemented; the shipped algorithm remains the discrete, deterministic version
documented above.

## License

This code repository is released under the MIT License.
