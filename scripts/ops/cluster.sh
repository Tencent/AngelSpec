#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ANGELSPEC_CLUSTER_ENV:-$SCRIPT_DIR/cluster.env}"
[[ -r "$ENV_FILE" ]] || {
    echo "missing $ENV_FILE; copy cluster.env.example and edit it"
    exit 2
}
ENV_FILE="$(cd -- "$(dirname -- "$ENV_FILE")" && pwd)/$(basename "$ENV_FILE")"
source "$ENV_FILE"

ROOT="${ANGELSPEC_ROOT:?set ANGELSPEC_ROOT in $ENV_FILE}"
HOSTFILE="${ANGELSPEC_HOSTFILE:-/etc/angelspec/hostfile}"
PYTHON="${ANGELSPEC_PYTHON:-python3}"
SSH_USER="${ANGELSPEC_SSH_USER:-root}"
SSH_KEY="${ANGELSPEC_SSH_KEY:-$HOME/.ssh/id_rsa}"
RAY_PORT="${ANGELSPEC_RAY_PORT:-6379}"
RAY_MIN_WORKER_PORT="${ANGELSPEC_RAY_MIN_WORKER_PORT:-10002}"
RAY_MAX_WORKER_PORT="${ANGELSPEC_RAY_MAX_WORKER_PORT:-19999}"
CPUS_PER_NODE="${ANGELSPEC_CPUS_PER_NODE:-16}"
GPUS_PER_NODE="${ANGELSPEC_GPUS_PER_NODE:-8}"
OBJECT_STORE_MEMORY="${ANGELSPEC_OBJECT_STORE_MEMORY:-200000000000}"
DEFAULT_CONFIG="${ANGELSPEC_DEFAULT_CONFIG:-configs/vllm_hy_v4_flash_dfly2_256k.yaml}"

if [[ -n "${NODE_IP_LIST:-}" ]]; then
    mapfile -t NODES < <(tr ',' '\n' <<< "$NODE_IP_LIST" | sed 's/:.*//' | sed '/^$/d')
else
    mapfile -t NODES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$HOSTFILE")
fi
if [[ -n "${ANGELSPEC_NUM_NODES:-}" ]]; then
    NODES=("${NODES[@]:0:ANGELSPEC_NUM_NODES}")
fi
[[ ${#NODES[@]} -gt 0 ]] || {
    echo "no nodes found in $HOSTFILE"
    exit 2
}

HEAD="${ANGELSPEC_HEAD:-${NODES[0]}}"
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10)
[[ -r "$SSH_KEY" ]] && SSH+=(-i "$SSH_KEY")

quote() {
    printf '%q' "$1"
}

remote() {
    local host="$1"
    local command="$2"
    local setup
    setup="source $(quote "$ENV_FILE"); ulimit -n 1048576 2>/dev/null || true; export PYTHONPATH=$(quote "$ROOT"):\${PYTHONPATH:-}; $command"
    "${SSH[@]}" "$SSH_USER@$host" "bash -lc $(quote "$setup")"
}

run_all() {
    local command="$1"
    local pids=()
    local rc=0
    for host in "${NODES[@]}"; do
        remote "$host" "$command" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    return "$rc"
}

ray_start() {
    local ports="--min-worker-port=$(quote "$RAY_MIN_WORKER_PORT") --max-worker-port=$(quote "$RAY_MAX_WORKER_PORT")"
    remote "$HEAD" "ray start --head --node-ip-address=$(quote "$HEAD") --port=$(quote "$RAY_PORT") --num-cpus=$(quote "$CPUS_PER_NODE") --num-gpus=$(quote "$GPUS_PER_NODE") --object-store-memory=$(quote "$OBJECT_STORE_MEMORY") --include-dashboard=false --disable-usage-stats $ports"
    local pids=()
    local rc=0
    for host in "${NODES[@]}"; do
        [[ "$host" == "$HEAD" ]] && continue
        remote "$host" "ray start --address=$(quote "$HEAD:$RAY_PORT") --node-ip-address=$(quote "$host") --num-cpus=$(quote "$CPUS_PER_NODE") --num-gpus=$(quote "$GPUS_PER_NODE") --object-store-memory=$(quote "$OBJECT_STORE_MEMORY") --disable-usage-stats $ports" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    [[ "$rc" -eq 0 ]] || return "$rc"
    remote "$HEAD" "RAY_ADDRESS=$(quote "$HEAD:$RAY_PORT") ray status"
}

ray_stop() {
    run_all "ray stop -f >/dev/null 2>&1 || true"
}

ray_status() {
    remote "$HEAD" "timeout 20s ray status --address=$(quote "$HEAD:$RAY_PORT")"
}

mooncake_clean() {
    remote "$HEAD" "RAY_ADDRESS=$(quote "$HEAD:$RAY_PORT") timeout 10s $(quote "$PYTHON") $(quote "$ROOT/scripts/ops/kill_mooncake_actor.py") 2>/dev/null || true"
    run_all "pkill -TERM -x mooncake_master 2>/dev/null || true; sleep 2; pkill -KILL -x mooncake_master 2>/dev/null || true"
}

train_start() {
    local config="${1:-$DEFAULT_CONFIG}"
    shift || true
    [[ "$config" = /* ]] || config="$ROOT/$config"
    [[ -f "$config" ]] || {
        echo "config not found: $config"
        exit 2
    }

    local name
    name="$(basename "${config%.yaml}")"
    local log="${ANGELSPEC_LOG:-$ROOT/logs/${name}_$(date +%Y%m%d_%H%M%S).log}"
    local pid_file="$ROOT/logs/cluster_train.pid"
    local latest_log="$ROOT/logs/cluster_train.log"
    local args=""
    local arg
    if [[ -n "${ANGELSPEC_TRAIN_NODES:-}" ]]; then
        args+=" training.training_num_nodes=$(quote "$ANGELSPEC_TRAIN_NODES")"
    fi
    if [[ -n "${ANGELSPEC_TRAIN_GPUS_PER_NODE:-}" ]]; then
        args+=" training.training_num_gpus_per_node=$(quote "$ANGELSPEC_TRAIN_GPUS_PER_NODE")"
    fi
    if [[ -n "${ANGELSPEC_INFER_GPUS:-}" ]]; then
        args+=" inference.inference_num_gpus=$(quote "$ANGELSPEC_INFER_GPUS")"
    elif [[ -n "${ANGELSPEC_INFER_NODES:-}" ]]; then
        args+=" inference.inference_num_gpus=$(quote "$((ANGELSPEC_INFER_NODES * GPUS_PER_NODE))")"
    fi
    for arg in "$@"; do
        args+=" $(quote "$arg")"
    done

    remote "$HEAD" "mkdir -p $(quote "$ROOT/logs"); cd $(quote "$ROOT"); export RAY_ADDRESS=$(quote "$HEAD:$RAY_PORT"); nohup $(quote "$PYTHON") -u -m angelspec.train_entry --config $(quote "$config")$args >$(quote "$log") 2>&1 </dev/null & echo \$! >$(quote "$pid_file"); printf '%s\n' $(quote "$log") >$(quote "$latest_log")"
    echo "training started: head=$HEAD log=$log"
}

train_stop() {
    local pid_file="$ROOT/logs/cluster_train.pid"
    remote "$HEAD" "if [[ -s $(quote "$pid_file") ]]; then kill -TERM \$(cat $(quote "$pid_file")) 2>/dev/null || true; fi; pkill -TERM -f '[a]ngelspec.train_entry' 2>/dev/null || true; pkill -TERM -f '[t]rain_retry.sh' 2>/dev/null || true"
}

status() {
    ray_status || true
    run_all "printf '%s train=%s mooncake=%s raylet=%s\n' \$(hostname) \$(pgrep -fc '[a]ngelspec.train_entry' || true) \$(pgrep -xc mooncake_master || true) \$(pgrep -xc raylet || true)"
}

logs() {
    local latest_log="$ROOT/logs/cluster_train.log"
    remote "$HEAD" "test -s $(quote "$latest_log") && log=\$(cat $(quote "$latest_log")) && tail -f \"\$log\""
}

stop_all() {
    train_stop || true
    sleep 3
    mooncake_clean || true
    ray_stop || true
}

usage() {
    echo "usage: $0 {ray-start|ray-stop|ray-status|train-start|train-stop|mooncake-clean|stop|restart|status|logs} [config] [overrides...]"
}

case "${1:-}" in
    ray-start)
        ray_start
        ;;
    ray-stop)
        ray_stop
        ;;
    ray-status)
        ray_status
        ;;
    train-start)
        shift
        train_start "$@"
        ;;
    train-stop)
        train_stop
        ;;
    mooncake-clean)
        mooncake_clean
        ;;
    stop)
        stop_all
        ;;
    restart)
        shift
        stop_all
        ray_start
        train_start "$@"
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        usage
        exit 2
        ;;
esac
