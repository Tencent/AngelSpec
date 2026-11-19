import random

import numpy as np
import torch
import torch.distributed as dist


def evaluate_named_cache(trainer, seed):
    cache = getattr(trainer, "_eval_cache", None) or []
    mesh = getattr(trainer, "grad_sync_mesh", None)
    group = mesh.get_group() if mesh is not None else None
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if dist.is_initialized():
        limits = torch.tensor([len(cache), -len(cache)], device=device, dtype=torch.int64)
        dist.all_reduce(limits, op=dist.ReduceOp.MIN, group=group)
        low, negative_high = limits.cpu().tolist()
        if low != -negative_high:
            raise RuntimeError(
                "Named eval cache lengths differ across ranks; regenerate the cache"
            )
    if not cache:
        return {}
    batch_size = (
        getattr(trainer.args, "eval_micro_batch_size", None) or trainer.args.micro_batch_size
    )
    if batch_size < 1:
        raise ValueError("eval_micro_batch_size must be positive")
    if getattr(trainer._eval_collator, "usp_sequence_sharded", False):
        batch_size = 1
    modes = [(module, module.training) for module in trainer.model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            trainer.model.eval()
            metrics = []
            for start in range(0, len(cache), batch_size):
                batch = trainer._eval_collator(cache[start : start + batch_size])
                if not trainer._defer_batch_h2d():
                    batch = {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                metrics.append(trainer.eval_forward(batch))
            return trainer._aggregate_named_eval_metrics(metrics)
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
