# Multi-Node Training

AngelSpec scales across nodes on both sides of the pipeline: training can span multiple nodes,
and a large target model can be served with tensor parallelism across nodes. Both are driven by
config on top of a running Ray cluster.

## Starting the Ray cluster

Start Ray manually on each node before launching AngelSpec. On the head node:

```bash
ray start --head --port 6379 --node-ip-address <HEAD_IP> --num-gpus <N> \
  --temp-dir /tmp/ray_$(id -u) --disable-usage-stats
```

On each worker node:

```bash
ray start --address <HEAD_IP>:6379 --num-gpus <N> \
  --temp-dir /tmp/ray_$(id -u) --disable-usage-stats
```

Then launch AngelSpec on the head node. It reads `RAY_ADDRESS` (default `"auto"`) to find the
cluster and blocks until all expected GPUs are visible.

## Training across nodes

`RayTrainGroup` creates `training_num_nodes × training_num_gpus_per_node` actors and spreads them
across nodes automatically.

```yaml
training:
  training_num_nodes: 2
  training_num_gpus_per_node: 8
```

## Inference across nodes

When a target model is too large for one node, the inference backend can shard it with
tensor parallelism across nodes. For example, with SGLang:

```yaml
inference:
  inference_num_gpus: 16
  inference_num_gpus_per_node: 8
  sglang:
    nnodes: 2
```

This creates one inference replica spanning two nodes (8 GPUs each), participating in a single
tensor-parallel group. vLLM supports multi-node tensor parallelism as well, using its Ray-based
distributed executor.

## Custom node placement

By default AngelSpec reserves a single unified placement group and assigns bundles to training or
inference according to `placement_strategy` (`training_first` or `inference_first`). Set
`placement_strategy: custom` to choose the nodes for each role explicitly, while still reserving
everything in one unified group.

IP-based placement uses Ray's per-node resource labels:

```yaml
training:
  placement_strategy: custom
  training_num_nodes: 2
  training_num_gpus_per_node: 8
  training_node_ips: [10.0.0.1, 10.0.0.3]

inference:
  inference_num_gpus: 16
  inference_num_gpus_per_node: 8
  inference_node_ips: [10.0.0.2, 10.0.0.4]
```

Ray label selectors (`training_node_selectors` / `inference_node_selectors`) are also supported
where the Ray version allows placement-group label selection. The configured node order is
preserved; for multi-node inference it determines the `node_rank` passed to the backend. Set only
one of `*_node_ips` or `*_node_selectors` per role.

The number of configured training nodes must equal `training_num_nodes`; the number of inference
nodes must equal `ceil(inference_num_gpus / inference_num_gpus_per_node)`.

## Example layout

```
Node 0 (head):   Ray head + training GPUs
Node 1 (worker): inference GPUs (TP node_rank=0)
Node 2 (worker): inference GPUs (TP node_rank=1)
```

## Networking

Multi-node runs use Mooncake over RDMA for hidden-state transfer, and NCCL/Gloo for collective
communication. On multi-NIC machines, set `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` /
`TP_SOCKET_IFNAME` to the interface carrying the node IP (find it with
`ip -o addr show | grep <your_node_ip>`). Cluster-specific network settings belong in your launch
environment, not in the config or code.
