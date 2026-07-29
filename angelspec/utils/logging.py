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

import datetime
import logging
import os

import torch.distributed as dist

from angelspec.utils import wandb as wandb_utils

_LOG_FORMAT = "[%(asctime)s] %(filename)s:%(lineno)d %(levelname)s %(message)s"
_tb_writer = None


def _get_logger_level():
    level_str = os.getenv("ANGELSPEC_LOG_LEVEL", "INFO").upper()
    try:
        log_level = getattr(logging, level_str)
    except ValueError:
        logging.warning("Invalid log level: %s, defaulting to WARNING", level_str)
    return log_level


def setup_logger(log_level=None, actor_name=None, ip_addr=None):
    logger_name = "AngelSpec" if actor_name is None else f"AngelSpec-{actor_name}"
    _logger = logging.getLogger(logger_name)
    _logger.handlers.clear()
    _logger.propagate = False
    if log_level is None:
        log_level = _get_logger_level()
    _logger.setLevel(log_level)
    handler = logging.StreamHandler()
    if ip_addr is None:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    else:
        rank = os.environ.get("RANK", 0)
        handler.setFormatter(
            logging.Formatter(
                f"[%(asctime)s{ip_addr} RANK:{rank}] %(filename)s:%(lineno)d %(levelname)s %(message)s"
            )
        )
    handler.setLevel(log_level)
    _logger.addHandler(handler)
    return _logger


logger = setup_logger()


def setup_file_logging(
    role: str,
    rank: int | str,
    group: int = 0,
    log_dir: str | None = None,
) -> None:
    """Add a FileHandler to the module-level logger for per-role/per-node/per-rank file logging."""
    if log_dir is None:
        log_dir = os.environ.get("ANGELSPEC_LOG_DIR")
    if log_dir is None:
        return

    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)

    try:
        from angelspec.utils.misc import get_current_node_ip

        node_ip = get_current_node_ip()
    except Exception:
        node_ip = "unknown"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join(log_dir, role, node_ip)
    filename = f"{role}_g{group}_rank{rank}_{timestamp}.log"
    filepath = os.path.join(dir_path, filename)

    try:
        os.makedirs(dir_path, exist_ok=True)
        fh = logging.FileHandler(filepath)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        fh.setLevel(logger.level)
        logger.addHandler(fh)
        logger.info(f"File logging enabled: {filepath}")
    except OSError:
        logger.warning(f"Could not set up file logging at {filepath} — NFS may not be available")


def print_with_rank(message):
    if dist.is_available() and dist.is_initialized():
        logger.info(f"rank {dist.get_rank()}: {message}")
    else:
        logger.info(f"non-distributed: {message}")


def print_on_rank0(message):
    if dist.get_rank() == 0:
        logger.info(message)


def init_tracking(args, primary: bool = True, **kwargs):
    global _tb_writer
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
        if getattr(args, "use_tensorboard", False) and getattr(args, "output_dir", None):
            from torch.utils.tensorboard import SummaryWriter

            tb_log_dir = os.path.join(args.output_dir, "runs")
            os.makedirs(tb_log_dir, exist_ok=True)
            _tb_writer = SummaryWriter(log_dir=tb_log_dir)
            logger.info(f"TensorBoard writer initialized at {tb_log_dir}")
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def get_tb_writer():
    """Return the module-level TensorBoard SummaryWriter, or None if not initialized."""
    return _tb_writer
