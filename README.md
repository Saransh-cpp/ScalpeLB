# ScalpeLB

Inertial Local-Search Load Balancing of MoEs on GPUs.

Built for
[MOE Dynamic Load Balancing Competition](https://www.codabench.org/competitions/16778/#/pages-tab)
by [LauzHack](https://lauzhack.com/2026-huawei/info) and Huawei.

## Algorithm

Builds on top of (and improves) [DeepSeek's EPLB](https://github.com/deepseek-ai/EPLB)
algorithm to optimize the placement of experts on GPUs in terms of:
* PAR: Peak-average ratio of device loads
* Transit: the number of experts that move to a different GPU

Instead of recomputing the whole layout every cycle (and paying that transit every
cycle), ScalpeLB establishs a good variance-aware placement ONCE, and then
*maintains* it with a small bounded number of surgical swaps per layer.

* Cycle 1: variance-aware placement (mean + K*std replication + balanced packing),
    aligned to the seed -> one-time placement cost.
* Cycle 2+ (per layer): try cheap bounded swap-maintenance (relocate hottest expert
    off the hottest GPU onto the coldest GPU, <= BUDGET swaps). Keep it only if its
    PAR stays within DRIFT_TOL of a fresh variance placement; otherwise the layer
    has DRIFTED (frozen replica counts went stale) -> re-place it fully.

The full-repack (EBLP) approach re-pays the entire placement transit every cycle
and its greedy packer churns even stable experts. ScalpeLB pays placement once and
only moves what reduces the max load -> far less transit, PAR no worse (often better).
The drift guard is essential: pure swap-maintenance freezes replica counts, which blows
up PAR on non-stationary / mixed traces (a newly-hot expert can't get more copies); the
per-layer re-place fixes that while keeping low transit on the stable majority. Swaps and
re-placement both preserve coverage -> no NaN-PAR risk.

## Parameters

* K: variance weight for replication/placement (mean + K*std). ~1-2.5.
* BUDGET: max swaps per layer per cycle (transit ceiling). Converges early (~2);
8 leaves headroom for drift.
* DRIFT_TOL: re-place a layer when its maintained PAR exceeds a fresh placement's by
more than this. 0.2 keeps the stationary win AND stays robust on drift.

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
