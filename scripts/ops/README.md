# Cluster operations

```bash
cp scripts/ops/cluster.env.example scripts/ops/cluster.env
vim scripts/ops/cluster.env
```

```bash
bash scripts/ops/cluster.sh ray-start
bash scripts/ops/cluster.sh train-start
bash scripts/ops/cluster.sh status
bash scripts/ops/cluster.sh logs
bash scripts/ops/cluster.sh train-stop
bash scripts/ops/cluster.sh mooncake-clean
bash scripts/ops/cluster.sh ray-stop
bash scripts/ops/cluster.sh stop
bash scripts/ops/cluster.sh restart
```

`restart` runs `train-stop → mooncake-clean → ray-stop → ray-start → train-start`.

Pass a config and OmegaConf overrides after `train-start` or `restart`.
