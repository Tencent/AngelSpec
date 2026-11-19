# Cluster operations

`scripts/ops/cluster.sh` starts and stops Ray, launches training and cleans up Mooncake on the
nodes of a hostfile. Copy the example environment file, set the repository path, hostfile,
Python, SSH, network interfaces, GPU counts and default training config in it, and keep it on
a path every node can read:

```bash
cp scripts/ops/cluster.env.example scripts/ops/cluster.env
export ANGELSPEC_CLUSTER_ENV=$PWD/scripts/ops/cluster.env
```

| Command | Action |
|---|---|
| `ray-start` / `ray-stop` | Start or stop Ray on every node |
| `train-start [config] [overrides...]` | Launch training on the running Ray cluster; the config defaults to the one in the environment file |
| `train-stop` | Stop training only |
| `status` | Show Ray, training and Mooncake processes |
| `logs` | Follow the latest training log |
| `mooncake-clean` | Remove Mooncake masters left behind by a crash |
| `stop` | Stop training, Mooncake and Ray |
| `restart [config]` | Clean up, restart Ray and relaunch training |

```bash
bash scripts/ops/cluster.sh train-start configs/vllm_hy_v4_flash_dfly2_256k.yaml \
  training.num_train_steps=100
```

`restart` resumes from a checkpoint only when `training.load_path` points to one.
