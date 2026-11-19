# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""SwanLab experiment tracking, mirroring the structure of ``angelspec/utils/wandb.py``.

Key differences vs. wandb:
- ``swanlab.init`` uses ``workspace`` (wandb's ``entity``) and ``experiment_name``
  (wandb's ``name``); ``parallel="shared"`` replaces wandb's shared ``Settings``.
- SwanLab has no ``define_metric``: the x-axis is passed explicitly via
  ``swanlab.log(..., step=...)`` instead of embedding ``<group>/step`` counters.
- Proxy: SwanLab follows the standard ``HTTP_PROXY`` / ``HTTPS_PROXY`` env vars
  (its HTTP client reads them directly), so there is no ``Settings`` proxy
  equivalent here.
"""

import logging
import os
import uuid
from copy import deepcopy

import swanlab

logger = logging.getLogger(__name__)

# Tracker credentials live in args but must not be uploaded with the run config.
_SECRET_ARGS = ("wandb_key", "swanlab_key")


def _is_offline_mode(args) -> bool:
    """Detect whether SwanLab should run in offline mode.

    Priority order:
    1) args.swanlab_mode if provided
    2) SWANLAB_MODE environment variable
    """
    if args.swanlab_mode:
        return args.swanlab_mode == "offline"
    return os.environ.get("SWANLAB_MODE") == "offline"


def _is_disabled_mode(args) -> bool:
    if args.swanlab_mode:
        return args.swanlab_mode == "disabled"
    return os.environ.get("SWANLAB_MODE") == "disabled"


def _needs_login(args) -> bool:
    """Cloud/local modes require credentials; offline/disabled do not."""
    if args.swanlab_mode:
        return args.swanlab_mode not in ("offline", "disabled")
    return os.environ.get("SWANLAB_MODE") not in ("offline", "disabled")


def init_swanlab_primary(args):
    if not args.use_swanlab:
        args.swanlab_run_id = None
        return

    if args.swanlab_mode:
        if args.swanlab_mode == "offline":
            logger.info("SwanLab offline mode enabled. Data will be saved locally.")
        elif args.swanlab_mode == "local":
            logger.info("SwanLab local mode enabled. Data is saved locally and watchable.")
        elif args.swanlab_mode == "disabled":
            logger.info("SwanLab disabled mode enabled. No data will be logged.")
        elif args.swanlab_mode == "online":
            logger.info("SwanLab online mode enabled. Data will be uploaded to cloud.")

    # Only perform explicit login when the mode actually uploads data.
    if _needs_login(args) and args.swanlab_key is not None:
        swanlab.login(api_key=args.swanlab_key, host=args.swanlab_host)

    base_group = args.swanlab_group or args.swanlab_project or "default"
    if args.swanlab_exp_name:
        group = base_group
        experiment_name = args.swanlab_exp_name
    elif args.swanlab_random_suffix:
        random_suffix = uuid.uuid4().hex[:8]
        group = base_group + "_" + random_suffix
        experiment_name = f"{group}-RANK_{args.rank}"
    else:
        group = base_group
        experiment_name = base_group

    init_kwargs = {
        "workspace": args.swanlab_workspace,
        "project": args.swanlab_project,
        "group": group,
        "experiment_name": experiment_name,
        "config": _compute_config_for_logging(args),
    }

    # Resume an existing run (e.g. crash recovery): reattach so metrics continue
    # on the same curve instead of opening a new run.
    if args.swanlab_run_id:
        init_kwargs["id"] = args.swanlab_run_id
        init_kwargs["resume"] = "allow"
        logger.info(f"SwanLab resuming existing run id={args.swanlab_run_id}")

    if args.swanlab_mode:
        init_kwargs["mode"] = args.swanlab_mode

    # Distributed training: multiple processes upload to the same experiment.
    # Only meaningful for cloud/local modes (offline keeps one writer per process).
    if not (_is_offline_mode(args) or _is_disabled_mode(args)):
        init_kwargs["parallel"] = "shared"

    if args.swanlab_dir:
        os.makedirs(args.swanlab_dir, exist_ok=True)
        init_kwargs["logdir"] = args.swanlab_dir
        logger.info(f"SwanLab logs will be stored in: {args.swanlab_dir}")

    run = swanlab.init(**init_kwargs)

    _init_swanlab_common()

    # Set swanlab_run_id in args for easy access throughout the training process.
    # Newer SwanLab APIs expose this via run.public.run_id; keep lightweight
    # fallbacks for compatibility with older layouts.
    args.swanlab_run_id = (
        getattr(getattr(run, "public", None), "run_id", None)
        or getattr(run, "id", None)
        or getattr(getattr(swanlab, "run", None), "id", None)
    )


def _compute_config_for_logging(args):
    output = {k: deepcopy(v) for k, v in args.__dict__.items() if k not in _SECRET_ARGS}

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    return output


def init_swanlab_secondary(args, router_addr=None):
    # SwanLab has no OpenMetrics forwarding equivalent (wandb exposes this via
    # x_stats_open_metrics_endpoints), so router_addr is unused; it is kept only
    # to mirror the wandb call-site signature.
    del router_addr

    swanlab_run_id = getattr(args, "swanlab_run_id", None)
    if swanlab_run_id is None:
        return

    if args.swanlab_mode:
        os.environ["SWANLAB_MODE"] = args.swanlab_mode

    if _needs_login(args) and args.swanlab_key is not None:
        swanlab.login(api_key=args.swanlab_key, host=args.swanlab_host)

    init_kwargs = {
        "id": swanlab_run_id,
        "workspace": args.swanlab_workspace,
        "project": args.swanlab_project,
        "config": _compute_config_for_logging(args),
        "resume": "allow",
        "reinit": True,
    }

    if args.swanlab_mode:
        init_kwargs["mode"] = args.swanlab_mode

    if not (_is_offline_mode(args) or _is_disabled_mode(args)):
        init_kwargs["parallel"] = "shared"

    if args.swanlab_dir:
        os.makedirs(args.swanlab_dir, exist_ok=True)
        init_kwargs["logdir"] = args.swanlab_dir

    swanlab.init(**init_kwargs)

    _init_swanlab_common()


def _init_swanlab_common():
    # SwanLab has no define_metric equivalent: the x-axis for every log() call is
    # supplied via the explicit ``step`` argument (see the log() helper below), so
    # there is nothing to register here. Kept to mirror wandb._init_wandb_common().
    pass


def is_active() -> bool:
    # swanlab.run is a submodule on some releases, so it is never None; get_run()
    # is the public signal for an initialized run.
    get_run = getattr(swanlab, "get_run", None)
    if callable(get_run):
        try:
            return get_run() is not None
        except RuntimeError:  # swanlab >= 0.10 raises when no run is active
            return False
    return getattr(swanlab, "run", None) is not None


def log(metrics: dict, step: int | None = None) -> None:
    """Log metrics to SwanLab.

    SwanLab (unlike wandb) attaches the x-axis via ``swanlab.log(..., step=...)``
    rather than embedding ``<group>/step`` counters into the metrics dict. This
    helper strips those wandb-style counters and forwards the remaining values.
    """
    if not is_active():
        return

    if step is None:
        step = _extract_step(metrics)

    data = {k: v for k, v in metrics.items() if not k.endswith("/step")}
    swanlab.log(data, step=step)


def _extract_step(metrics: dict) -> int | None:
    """Best-effort x-axis extraction from wandb-style step counters."""
    for key in ("train/step", "inference/step", "eval/step"):
        if metrics.get(key) is not None:
            return metrics[key]
    return None
