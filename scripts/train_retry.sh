#!/bin/bash
# Auto-retry + checkpoint-resume wrapper for angelspec.train_entry.
#
# Runs the training driver fresh on each attempt (so its Ray actors are reaped
# when it exits) and relaunches on non-zero exit. Resume is automatic: the
# framework saves iter_XXX/ + latest_checkpointed_iteration.txt every
# save_interval steps and checkpoint.load() picks the latest up on restart
# (continual_training=false → full resume of weights+optimizer+LR+RNG+step).
#
# Retry policy: an attempt that ran >= PROGRESS_SEC counts as real progress and
# resets the failure counter (long runs tolerate occasional transient hiccups —
# mooncake transfer / NCCL timeouts); an attempt that dies quickly (config/env
# error) burns one of MAX_ATTEMPTS. After MAX_ATTEMPTS consecutive fast failures
# the wrapper exits with the child's return code.
#
# LIMITATION: a crash that leaves GPU-holding orphans in a DIFFERENT container /
# PID namespace (e.g. a vLLM worker OOM) cannot be reclaimed from here. The
# cleanup below is best-effort and same-namespace only; if a node stays occupied
# it is named in a WARNING so it can be reset at the platform level.
#
# Usage:
#   MAX_ATTEMPTS=5 RAY_ADDRESS=<head>:6379 scripts/train_retry.sh \
#       --config configs/foo.yaml training.save_interval=100 <overrides...>

set -uo pipefail

MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
PROGRESS_SEC=${PROGRESS_SEC:-900}      # attempt lasting this long resets the counter
RETRY_BACKOFF_SEC=${RETRY_BACKOFF_SEC:-45}
GPU_BUSY_MIB=${GPU_BUSY_MIB:-2000}     # per-GPU "still occupied" threshold after cleanup
PY=${PY:-/usr/bin/python3}

_child_pid=""
_stopping=0
_on_signal() {
    _stopping=1
    [[ -n "$_child_pid" ]] && kill -TERM "$_child_pid" 2>/dev/null
}
trap _on_signal INT TERM

_local_cleanup() {
    pkill -9 -f "angelspec.train_entry" 2>/dev/null || true
    pkill -9 -f "mooncake_master" 2>/dev/null || true
}

# Best-effort cluster-wide GPU reclaim + occupancy report via Ray. Kills
# GPU-holding compute apps that ARE signal-reachable; reports any node still
# occupied (likely cross-namespace orphans needing a platform reset).
_cluster_cleanup() {
    [[ -z "${RAY_ADDRESS:-}" ]] && return 0
    "$PY" - <<'PYEOF' 2>/dev/null | grep -viE "INFO|FutureWarning|warnings.warn" || true
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"])
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
busy_mib = int(os.environ.get("GPU_BUSY_MIB", "2000"))

@ray.remote(num_cpus=1)
def clean_and_report():
    import os, signal, socket, subprocess
    def smi(q):
        return subprocess.run(
            ["nvidia-smi", f"--query-{q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.split()
    mypid = os.getpid()
    for p in smi("compute-apps=pid"):
        try:
            p = int(p)
            if p != mypid:
                os.kill(p, signal.SIGKILL)
        except Exception:
            pass  # not reachable (different namespace) or already gone
    used = [int(x) for x in smi("gpu=memory.used")]
    return socket.gethostname(), max(used) if used else 0

tasks = [
    clean_and_report.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)
    ).remote()
    for n in ray.nodes() if n["Alive"]
]
ips = [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]]
for ip, (_, mx) in zip(ips, ray.get(tasks)):
    flag = "  <-- STILL OCCUPIED (may need platform reset)" if mx > busy_mib else ""
    print(f"[cleanup] {ip}: max_gpu_used={mx} MiB{flag}")
PYEOF
}

attempt=0
fails=0
rc=0
while :; do
    attempt=$((attempt + 1))
    echo "=== [train_retry] attempt #${attempt} (consecutive_fails=${fails}/${MAX_ATTEMPTS}) $(date '+%F %T') ==="
    start=$(date +%s)

    "$PY" -u -m angelspec.train_entry "$@" &
    _child_pid=$!
    wait "$_child_pid"
    rc=$?
    _child_pid=""
    elapsed=$(( $(date +%s) - start ))

    if [[ "$_stopping" -eq 1 ]]; then
        echo "=== [train_retry] interrupted by signal; not retrying (rc=${rc}) ==="
        exit $rc
    fi
    if [[ "$rc" -eq 0 ]]; then
        echo "=== [train_retry] training completed cleanly after ${attempt} attempt(s) ==="
        exit 0
    fi

    if [[ "$elapsed" -ge "$PROGRESS_SEC" ]]; then
        echo "=== [train_retry] attempt ran ${elapsed}s (>=${PROGRESS_SEC}) — real progress; resetting fail counter ==="
        fails=0
    else
        fails=$((fails + 1))
    fi
    if [[ "$fails" -ge "$MAX_ATTEMPTS" ]]; then
        echo "=== [train_retry] FAILED: ${fails} consecutive fast failures (last rc=${rc}); giving up ==="
        exit $rc
    fi

    echo "=== [train_retry] attempt failed (rc=${rc}, ran ${elapsed}s); cleanup + resume from latest checkpoint in ${RETRY_BACKOFF_SEC}s ==="
    _local_cleanup
    _cluster_cleanup
    sleep "$RETRY_BACKOFF_SEC"
done
