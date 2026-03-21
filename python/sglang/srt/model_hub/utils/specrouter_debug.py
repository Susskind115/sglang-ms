from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional


SPECROUTER_DEBUG_ENV = "SPECROUTER_DEBUG_TRACE"


def specrouter_debug_enabled() -> bool:
    value = os.environ.get(SPECROUTER_DEBUG_ENV, "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _tensor_summary(value: Any) -> Optional[Dict[str, Any]]:
    if value is None or not hasattr(value, "shape"):
        return None

    summary: Dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(getattr(value, "dtype", None)),
        "device": str(getattr(value, "device", None)),
        "numel": int(value.numel()),
    }
    if len(value.shape) > 0:
        summary["len"] = int(value.shape[0])
    return summary


def _enum_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    return getattr(value, "name", str(value))


def describe_spec_info(spec_info: Any) -> Dict[str, Any]:
    if spec_info is None:
        return {"type": None}

    accept_length_cpu = getattr(spec_info, "accept_length_cpu", None)
    summary: Dict[str, Any] = {
        "type": type(spec_info).__name__,
        "capture_hidden_mode": _enum_name(
            getattr(spec_info, "capture_hidden_mode", None)
        ),
        "enabled_skip_extend": bool(
            getattr(spec_info, "enabled_skip_extend", False)
        ),
        "topk_p": _tensor_summary(getattr(spec_info, "topk_p", None)),
        "topk_index": _tensor_summary(getattr(spec_info, "topk_index", None)),
        "hidden_states": _tensor_summary(getattr(spec_info, "hidden_states", None)),
        "verified_id": _tensor_summary(getattr(spec_info, "verified_id", None)),
        "drafted_id": _tensor_summary(getattr(spec_info, "drafted_id", None)),
        "accept_length": _tensor_summary(getattr(spec_info, "accept_length", None)),
        "seq_lens_for_draft_extend": _tensor_summary(
            getattr(spec_info, "seq_lens_for_draft_extend", None)
        ),
        "req_pool_indices_for_draft_extend": _tensor_summary(
            getattr(spec_info, "req_pool_indices_for_draft_extend", None)
        ),
        "accept_length_for_draft_extend": _tensor_summary(
            getattr(spec_info, "accept_length_for_draft_extend", None)
        ),
        "next_out_cache_loc": _tensor_summary(
            getattr(spec_info, "next_out_cache_loc", None)
        ),
        "filtered_out_cache_loc": _tensor_summary(
            getattr(spec_info, "filtered_out_cache_loc", None)
        ),
        "kv_indptr": _tensor_summary(getattr(spec_info, "kv_indptr", None)),
        "kv_indices": _tensor_summary(getattr(spec_info, "kv_indices", None)),
        "accept_length_cpu_len": (
            len(accept_length_cpu) if accept_length_cpu is not None else None
        ),
    }
    return summary


def describe_batch(batch: Any) -> Dict[str, Any]:
    if batch is None:
        return {"type": None}

    reqs = getattr(batch, "reqs", None)
    finished_count = None
    if reqs is not None:
        finished_count = sum(1 for req in reqs if req.finished())

    return {
        "type": type(batch).__name__,
        "forward_mode": _enum_name(getattr(batch, "forward_mode", None)),
        "req_count": len(reqs) if reqs is not None else None,
        "finished_req_count": finished_count,
        "req_pool_indices": _tensor_summary(getattr(batch, "req_pool_indices", None)),
        "seq_lens": _tensor_summary(getattr(batch, "seq_lens", None)),
        "out_cache_loc": _tensor_summary(getattr(batch, "out_cache_loc", None)),
        "spec_info": describe_spec_info(getattr(batch, "spec_info", None)),
    }


def describe_strategy(strategy: Any) -> Optional[Dict[str, Any]]:
    if strategy is None:
        return None

    return {
        "mode": _enum_name(getattr(strategy, "mode", None)),
        "current_chain_ids": list(getattr(strategy, "current_chain_ids", [])),
        "num_current_chain": getattr(strategy, "num_current_chain", None),
    }


def describe_chain_diff(diff: Any) -> Optional[Dict[str, Any]]:
    if diff is None:
        return None

    return {
        "switch_mode": bool(getattr(diff, "switch_mode", False)),
        "new_mode": _enum_name(getattr(diff, "new_mode", None)),
        "reload_models": list(getattr(diff, "reload_models", []) or []),
        "unload_models": list(getattr(diff, "unload_models", []) or []),
        "check_reload": bool(getattr(diff, "check_reload", False)),
        "non_diff": bool(getattr(diff, "non_diff", False)),
    }


def specrouter_debug_log(
    logger: logging.Logger, event: str, **payload: Any
) -> None:
    if not specrouter_debug_enabled():
        return
    logger.info(
        "[SPECDBG] %s %s",
        event,
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True),
    )
