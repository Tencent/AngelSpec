import os

# Environment variables that should be forwarded to all Ray actors.
# NOTE: TORCHINDUCTOR_CACHE_DIR is intentionally excluded — each node should
# use its own node-local default (/tmp/torchinductor_$USER/) to avoid
# cross-node triton kernel cache corruption over NFS.
_ANGELSPEC_ENV_KEYS = [
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "CUDA_LAUNCH_BLOCKING",
    "GLOO_SOCKET_IFNAME",
    "HF_HOME",
    "HF_TOKEN",
    "MC_LOG_LEVEL",
    "MODELOPT_MAX_TOKENS_PER_EXPERT",
    "NCCL_DEBUG",
    "NCCL_IB_DISABLE",
    "NCCL_IB_HCA",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_P2P_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    "SGLANG_DISABLE_CUDNN_CHECK",
    "SGLANG_VLM_CACHE_SIZE_MB",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
    "TORCHINDUCTOR_FX_GRAPH_CACHE",
    "ANGELSPEC_LOG_DIR",
    "ANGELSPEC_LOG_LEVEL",
    "ANGELSPEC_MTP_LOSS_CHUNK",
    "ANGELSPEC_DFLASH_LOSS_CHUNK",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PATH",
    "no_proxy",
    "NO_PROXY",
    "TP_SOCKET_IFNAME",
    "CUTE_DSL_CACHE_DIR",
    "ANGELSPEC_FLASH_ATTN_OPT_LEVEL",
]

# Prevent Ray from overriding VISIBLE_DEVICES so actors manage GPU assignment themselves.
# Reference: https://github.com/ray-project/ray/blob/161849364/python/ray/_private/accelerators/
_RAY_NOSET_VISIBLE_DEVICES_KEYS = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


def get_angelspec_env_vars() -> dict[str, str]:
    """Return common environment variables for all Ray actors.

    Includes:
    - ANGELSPEC_* variables (e.g. log level) from the current process
    - RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES = "1" to prevent Ray from
      overriding device visibility

    Intended for use with ``ray.remote(runtime_env={"env_vars": ...})``.
    Call-site env vars merged after this dict take higher priority.
    """
    env = {k: "1" for k in _RAY_NOSET_VISIBLE_DEVICES_KEYS}
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(k, "8")
    env.update({k: os.environ[k] for k in _ANGELSPEC_ENV_KEYS if k in os.environ})
    # Blank any HTTP proxy in actors: mooncake's transfer engine reaches the
    # metadata server over intra-cluster IPs and a proxy (often exported by a
    # login shell's .bashrc) makes the metadata PUT fail (http=400 -> -900).
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env[k] = ""
    return env
