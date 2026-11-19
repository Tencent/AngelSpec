import re

import torch
import torch.distributed as dist


def resolve_metric_sources(args):
    def resolve():
        if not getattr(args, "mtp_source_metrics", False):
            return ()
        names = getattr(args, "mtp_metric_sources", None)
        if not names:
            path = getattr(args, "bin_data_list", None)
            if not path:
                raise ValueError("mtp_source_metrics requires bin_data_list or mtp_metric_sources")
            from angelspec.data.megatron_bin import _parse_bin_data_list

            names = [kind for _, _, kind in _parse_bin_data_list(path)]
        if isinstance(names, str):
            raise ValueError("mtp_metric_sources must be a list of source names")
        if any(not isinstance(name, str) for name in names):
            raise ValueError("Metric source names must be strings")
        names = tuple(sorted(set(names)))
        if len(names) > 32 or any(
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name) is None
            or name == "unknown"
            for name in names
        ):
            raise ValueError("Use at most 32 metric source names; 'unknown' is reserved")
        return (*names, "unknown")

    if not dist.is_initialized():
        return resolve()
    from angelspec.utils.distributed import get_gloo_group

    payload = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = (resolve(), None)
        except Exception as exc:
            payload[0] = ((), f"{type(exc).__name__}: {exc}")
    dist.broadcast_object_list(payload, src=0, group=get_gloo_group())
    names, error = payload[0]
    if error is not None:
        raise ValueError(error)
    return tuple(names)


def batch_source_ids(batch, source_names, device, *, doc_ids=None):
    unknown = len(source_names) - 1
    lookup = {name: index for index, name in enumerate(source_names)}
    shape = batch["input_ids"].shape
    if "ctx_doc_ids" in batch:
        docs = (
            doc_ids
            if doc_ids is not None
            else batch["ctx_doc_ids"].to(device=device, dtype=torch.long)
        )
        sources = batch.get("doc_sources", ())
        table = torch.tensor(
            [unknown, *(lookup.get(name, unknown) for name in sources)],
            dtype=torch.long,
            device=device,
        )
        indices = docs + 1
        indices = torch.where((indices >= 0) & (indices < table.numel()), indices, 0)
        return table[indices]
    sources = batch.get("data_sources")
    if sources is None:
        sources = [None] * shape[0]
    if len(sources) != shape[0]:
        raise ValueError("data_sources must contain one source per batch row")
    rows = torch.tensor(
        [lookup.get(name, unknown) for name in sources], dtype=torch.long, device=device
    )
    return rows[:, None].expand(shape)


@torch.no_grad()
def grouped_acceptance(source_ids, correct, correct_gt, num_sources):
    totals = []
    for source in range(num_sources):
        mask = source_ids == source
        totals.append(torch.stack(((mask & correct).sum(), (mask & correct_gt).sum(), mask.sum())))
    return torch.stack(totals)


def source_metric_log(source_names, source_totals, overall_totals):
    expected = overall_totals[:, [2, 3, 4]].double().cpu()
    totals = source_totals.detach().cpu()
    if not torch.allclose(totals.sum(0).double(), expected, rtol=2e-6, atol=1e-4):
        raise RuntimeError("MTP source metric totals do not match overall correct/count totals")
    result = {}
    for name, values in zip(source_names, totals.tolist()):
        prefix = f"train_{name}"
        result[f"{prefix}/supervised_tokens"] = values[0][2]
        acc, acc_gt = [], []
        for depth, (correct, correct_gt, count) in enumerate(values):
            result[f"{prefix}/tokens_{depth}"] = count
            acc.append(correct / count if count else 0.0)
            acc_gt.append(correct_gt / count if count else 0.0)
            if count:
                result[f"{prefix}/acc_{depth}"] = acc[-1]
                result[f"{prefix}/acc_gt_{depth}"] = acc_gt[-1]
        if not values[0][2]:
            continue
        for suffix, rates in (("", acc), ("_gt", acc_gt)):
            product, length = 1.0, 0.0
            for rate in rates:
                product *= rate
                length += product
            result[f"{prefix}/avg_acc{suffix}"] = sum(rates) / len(rates)
            result[f"{prefix}/simulated_acc_len{suffix}"] = length
    return result
