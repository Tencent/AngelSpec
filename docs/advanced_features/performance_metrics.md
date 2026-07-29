# Performance Metrics

AngelSpec includes an opt-in metrics system to identify bottlenecks between the asynchronous
inference and synchronous training stages.

## Enabling

Set `debug.enable_perf_metrics: true` (or override on the CLI). When disabled, no overhead is
added to the training loop.

```bash
python3 -m angelspec.train_entry --config configs/default.yaml debug.enable_perf_metrics=true
```

## Metrics

All metrics are logged under the `perf/` namespace (tied to `train/step`).

**Training (per optimizer step)**

| Metric | Unit | Description |
|--------|------|-------------|
| `perf/step_time` | s | Wall-clock time of a training step (data fetch + compute + optimizer) |
| `perf/data_time` | s | Time in the data iterator (Ray queue get + Mooncake fetch + collation + H2D) |
| `perf/compute_time` | s | GPU time for forward + backward + optimizer, via CUDA events |
| `perf/train_capacity` | samples/s | `global_batch_size / step_time` — throughput if data were always ready |

**Inference (aggregated between steps)**

| Metric | Unit | Description |
|--------|------|-------------|
| `perf/infer_capacity` | samples/s | System inference capacity across engines |
| `perf/infer_batch_time` | s | Average time of a single engine generate call |

**Pipeline health**

| Metric | Unit | Description |
|--------|------|-------------|
| `perf/dispatch_wait` | s | Time the loop waited for enough samples to dispatch. High → inference-bound |

## Progress bar

With metrics enabled, the progress bar adds fields:

```
Training: 10%|== | loss=0.350, acc=0.830, thru=12.1, I=18.5, T=13.2, wait=0.1s, pool=24, epoch=1/10
```

| Key | Meaning |
|-----|---------|
| `thru` | Realized end-to-end pipeline throughput (samples/s); in steady state ≈ `min(I, T)` |
| `I` | Inference capacity — how fast inference *could* produce |
| `T` | Training capacity — how fast training *could* consume |
| `wait` | Dispatch wait time |
| `pool` | Current sample-pool size |

## Finding the bottleneck

```
if dispatch_wait >> 0:      bottleneck = inference   # training is starved
elif pool grows over time:  bottleneck = training    # inference outpaces training
```

Within training:

```
if data_time >> compute_time:   bottleneck = Mooncake transfer / data loading
elif compute_time >> data_time: bottleneck = GPU compute
```

Interpreting capacities:

- **`I > T`** — training is the bottleneck; the pool tends to grow.
- **`T > I`** — inference is the bottleneck; `dispatch_wait` is high.
- **`I ≈ T`** — balanced; `thru` approaches both.

## Implementation notes

Compute time uses CUDA events (`torch.cuda.Event(enable_timing=True)`) rather than
`torch.cuda.synchronize()`, so no extra synchronization is introduced — timings are extracted at
an existing sync point. Data-loading time uses wall-clock (`time.time()`) because that work is
CPU/network-bound and the GPU stream may be idle.
