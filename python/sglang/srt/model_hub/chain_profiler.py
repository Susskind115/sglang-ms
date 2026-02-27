"""Lightweight performance profiler for model hub scheduling.

This profiler adapts the PerformanceProfiler used in the sdc reference
system.  It provides NVTX hooks (when available) and keeps per-stage timing
statistics that can later be summarised or exported for diagnostics.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import bisect
from collections import defaultdict

import numpy as np

try:  # NVTX is optional; fall back to pure time based profiling otherwise.
    import torch.cuda.nvtx as nvtx  # type: ignore

    _NVTX_AVAILABLE = True
except Exception:  # pragma: no cover - defensive fallback
    _NVTX_AVAILABLE = False
    nvtx = None  # type: ignore[misc]


class ChainPerformanceProfiler:
    """Per-stage performance recorder with optional NVTX annotations.

    The design mirrors ``cite/sdc/utils/profiler.py`` but strips the
    project-specific dependencies so it can live entirely inside
    ``model_hub``.  Timings are stored per range name (optionally
    namespaced by model id) and can be aggregated to feed the scheduler.
    """

    #: Colour palette reused from the sdc profiler for NVTX traces.
    _COLOR_MAP: Dict[str, int] = {
        "prefill": 0xFF0000,
        "draft": 0x00FF00,
        "verify": 0x0000FF,
        "autoregressive": 0x8888FF,
        "total": 0xFFFFFF,
        "default": 0x808080,
    }

    def __init__(
        self,
        enable_nvtx: bool = True,
        log_to_console: bool = False,
        summary_to_console: bool = False,
    ) -> None:
        self.enable_nvtx = enable_nvtx and _NVTX_AVAILABLE
        self.log_to_console = log_to_console
        self.summary_to_console = summary_to_console

        self._timers: Dict[str, List[float]] = {}
        self._last_update: Dict[str, float] = {}
        self._counters: Dict[str, List[Any]] = {}
        self._ema_timers: Dict[str, float] = {}
        self._final_timers: Dict[str, float] = {}
        # self._clear_flag: Dict[str, bool] = {}
        # self._LUT_timers: Dict[str, InterpolatedLookupTable] = {}
        self._LUT_timers: Dict[str, InterpolatedLookupTable] = defaultdict(InterpolatedLookupTable)
        self._clear_flag = defaultdict(bool)
        # self._model_decode_timers: Dict[str, float] = {}

        self._thread_local = threading.local()
        self._thread_local.active_ranges = []  # type: ignore[attr-defined]
        self._summary: Optional[Dict[str, Dict[str, Any]]] = None

        self._max_window_len = 9
        self._ema_alpha = 0.7 # 增量权重

        # self.protect_length = 7
        self.protect_length = 3

    # ------------------------------------------------------------------
    # Core range helpers
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def range(
        self,
        name: Union[str, List[str]],
        counts: int,
        verify_K: float = 1.0,
        model_id: Optional[str] = None,
        *,
        ignore_level: bool = False,
        color_id: Optional[int] = None,
    ):
        """Measure a code block.

        ``name`` is combined with ``model_id`` to form the key used for
        statistics.  When ``ignore_level`` is False, nested ranges receive
        an ``L{depth}_`` prefix mirroring the sdc profiler output.
        """

        depth = len(getattr(self._thread_local, "active_ranges", []))
        if isinstance(name, str):
            name = [name]
        range_name_list = []
        for n in name:
            if ignore_level:
                range_name_list.append(f"{n}_{model_id}" if model_id else n)
            else:
                prefix = f"L{depth}_"
                range_name_list.append(f"{prefix}{n}_{model_id}" if model_id else f"{prefix}{n}")
        
        # verifyK_range_name = None
        # if "verify" in name:
        #     if ignore_level:
        #         verifyK_range_name = f"verifyK_{model_id}" if model_id else "verifyK"
        #     else:
        #         prefix = f"L{depth}_"
        #         verifyK_range_name = f"{prefix}verifyK_{model_id}" if model_id else f"{prefix}verifyK"

        # color = color_id if color_id is not None else self._COLOR_MAP.get(
        #     name, self._COLOR_MAP["default"]
        # )

        start = time.perf_counter()
        stack: Optional[List[Tuple[str, float]]] = getattr(
            self._thread_local, "active_ranges", None
        )
        if stack is None:
            stack = []
            self._thread_local.active_ranges = stack  # type: ignore[attr-defined]
        stack.append((range_name_list, start))

        if self.enable_nvtx:
            nvtx.range_push(range_name_list[0])  # type: ignore[call-arg]

        try:
            yield
        finally:
            end = time.perf_counter()

            popped_name, _ = stack.pop()
            assert (
                popped_name[0] == range_name_list[0]
            ), f"Range mismatch: expected {range_name_list[0]}, got {popped_name[0]}"

            if self.enable_nvtx:
                nvtx.range_pop()  # type: ignore[misc]

            # if counts*verify_K == 0:
            #     return
            if "verify" not in name:
                if counts*verify_K == 0:
                    elapsed = self._final_timers[range_name_list[0]]/verify_K
                else:
                    elapsed = (end - start)/(counts*verify_K)
            else:
                elapsed = (end - start)
            # self._timers.setdefault(range_name, []).append(elapsed)
            for i, range_name in enumerate(range_name_list):
                range_name_box = self._timers.setdefault(range_name, [])
                range_name_ema = self._ema_timers.setdefault(range_name, 0)
                range_name_last_update = self._last_update.setdefault(range_name, -1)

                range_name_last_update = (range_name_last_update+1)%self._max_window_len # update
                if len(range_name_box) >= self._max_window_len:
                    range_name_box[range_name_last_update] = elapsed # update
                else:
                    range_name_box.append(elapsed)
                # if self._clear_flag[range_name]:
                #     self._clear_flag[range_name] = False
                #     range_name_box[:] = [elapsed] * max(len(range_name_box), self.protect_length)

                self._ema_timers[range_name] = elapsed * self._ema_alpha + range_name_ema * (1 - self._ema_alpha) # update
                self._last_update[range_name] = range_name_last_update
                # self._final_timers[range_name] = self.get_mid_time(range_name_box)
                mid_time = self.get_mid_time(range_name_box)
                if len(range_name_box) <= self.protect_length:
                    self._final_timers[range_name] = 0.0
                else:
                    self._final_timers[range_name] = mid_time
                # update _LUT_timers
                if name[i] == "verify":
                    self._LUT_timers[range_name].update(int(counts*verify_K), mid_time)
                    # print(f"LUT_timers: {self._LUT_timers[range_name]._x}, {self._LUT_timers[range_name]._y}")
                elif name[i] == "warmup_decode":
                    special_name = f"verify_{model_id}"
                    self._LUT_timers[special_name].update(1, mid_time)
                    # print(f"LUT_timers: {self._LUT_timers[range_name]._x}, {self._LUT_timers[range_name]._y}")

            # if verifyK_range_name is not None:
            #     # verifyK_range_name = verifyK_name
            #     range_name_box = self._timers.setdefault(verifyK_range_name, [])
            #     range_name_ema = self._ema_timers.setdefault(verifyK_range_name, 0)
            #     range_name_last_update = self._last_update.setdefault(verifyK_range_name, 0)
            #     elapsedK = elapsed / verify_K

            #     range_name_last_update = (range_name_last_update+1)%self._max_window_len # update
            #     if len(range_name_box) >= self._max_window_len:
            #         range_name_box[range_name_last_update] = elapsedK # update
            #     else:
            #         range_name_box.append(elapsedK)
            #     self._ema_timers[verifyK_range_name] = elapsedK * self._ema_alpha + range_name_ema * (1 - self._ema_alpha) # update
            #     self._last_update[verifyK_range_name] = range_name_last_update
            #     self._final_timers[verifyK_range_name] = self.get_mid_time(range_name_box)

            # if name[0] in ["warmup_decode", "verify", "draft"]:
            #     # current_decode_time = self._model_decode_timers.get(model_id, 0.0)
            #     warmup_decode_time = self._final_timers.get(f"warmup_decode_{model_id}", 0.0)
            #     verifyK_time = self._final_timers.get(f"verifyK_{model_id}", 0.0)
            #     draft_time = self._final_timers.get(f"draft_{model_id}", 0.0)
            #     self._model_decode_timers[model_id] = max(warmup_decode_time, verifyK_time, draft_time)

            if self.log_to_console:
                for range_name in range_name_list:
                    indent = " " * depth
                    print(f"[PROFILER]{indent}{range_name}: {elapsed * 1000:.2f} ms")
                
    def record(
        self,
        name: Union[str, List[str]],
        count: float,
        model_id: Optional[str] = None,
        *,
        ignore_level: bool = False,
    ):
        
        depth = len(getattr(self._thread_local, "active_ranges", []))
        if isinstance(name, str):
            name = [name]
        range_name_list = []
        for n in name:
            if ignore_level:
                range_name_list.append(f"{n}_{model_id}" if model_id else n)
            else:
                prefix = f"L{depth}_"
                range_name_list.append(f"{prefix}{n}_{model_id}" if model_id else f"{prefix}{n}")
        

        for i, range_name in enumerate(range_name_list):
            range_name_box = self._timers.setdefault(range_name, [])
            # range_name_ema = self._ema_timers.setdefault(range_name, 0)
            range_name_last_update = self._last_update.setdefault(range_name, -1)

            range_name_last_update = (range_name_last_update+1)%self._max_window_len # update
            if len(range_name_box) >= self._max_window_len:
                range_name_box[range_name_last_update] = count # update
            else:
                range_name_box.append(count)
            self._ema_timers[range_name] = count
            self._last_update[range_name] = range_name_last_update
            self._final_timers[range_name] = self.get_mid_time(range_name_box)

        if self.log_to_console:
            for range_name in range_name_list:
                indent = " " * depth
                print(f"[PROFILER]{indent}{range_name}: {count * 1000:.2f} ms")

    def clear_timer(
        self,
        name: Union[str, List[str]],
        model_id: Optional[str] = None,
        min_n_record: int = 10,
        *,
        ignore_level: bool = False,
    ):
        depth = len(getattr(self._thread_local, "active_ranges", []))
        if isinstance(name, str):
            name = [name]
        range_name_list = []
        for n in name:
            if ignore_level:
                range_name_list.append(f"{n}_{model_id}" if model_id else n)
            else:
                prefix = f"L{depth}_"
                range_name_list.append(f"{prefix}{n}_{model_id}" if model_id else f"{prefix}{n}")

        for i, range_name in enumerate(range_name_list):
            # range_name_box = self._timers.setdefault(range_name, [])
            # range_name_box.clear()
            if range_name in self._timers:
                self._timers.pop(range_name, None)
                # 或者使用 self._timers.pop(range_name, None)
            # print(f"clear_timer: {range_name_box}")
            self._ema_timers.pop(range_name, None)
            self._last_update.pop(range_name, None)
            self._final_timers.pop(range_name, None)
            # if range_name in self._LUT_timers:
            #     if  self._LUT_timers[range_name].n_record > min_n_record:
            #         # force clear
            #         self._timers.pop(range_name, None)
            #         # print(f"clear_timer: {range_name_box}")
            #         self._ema_timers.pop(range_name, None)
            #         self._last_update.pop(range_name, None)
            #         self._final_timers.pop(range_name, None)
            # self._clear_flag[range_name] = True
        
        if self.log_to_console:
            print(f"[PROFILER] Cleared timer: {range_name}")


    # ------------------------------------------------------------------
    # Basic counters / metrics API
    # ------------------------------------------------------------------
    def mark_event(
        self, name: str, value: Any = None, *, color_id: Optional[int] = None
    ) -> None:
        if self.enable_nvtx:
            if color_id is not None:
                nvtx.mark(name, color_id)  # type: ignore[misc]
            else:
                nvtx.mark(name)  # type: ignore[misc]

        if value is not None:
            self._counters.setdefault(name, []).append(value)
            if self.log_to_console:
                print(f"[EVENT] {name}: {value}")

    def record_counter(
        self, name: str, count: int = 1, *, model_id: Optional[str] = None
    ) -> None:
        key = f"{name}_{model_id}" if model_id else name
        self._counters.setdefault(key, []).append(count)
        if self.log_to_console:
            print(f"[COUNTER] {key}: {count}")

    def record_latency(
        self,
        name: str,
        latency: float,
        *,
        model_id: Optional[str] = None,
    ) -> None:
        key = f"{name}_{model_id}" if model_id else name
        self._timers.setdefault(key, []).append(latency)
        if self.log_to_console:
            print(f"[LATENCY] {key}: {latency * 1000:.2f} ms")

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def _print_timer(self):
        for key, value in self._timers.items():
            print(f"{key}: mid: {value}, ema: {self._ema_timers[key]}, final: {self._final_timers[key]}")

    def _get_timer(self, key: str) -> Tuple[Optional[List[float]], Optional[float], Optional[float], Optional[float]]:
        # timer = []
        # ema_times = []
        # for key in key_candidates:
        if key in self._timers and self._timers[key]:
            # timer.append(self._timers[key])
            # ema_times.append(self._ema_timers[key])
            return self._timers[key], self._last_update[key], self._ema_timers[key], self._final_timers[key]
        return None, 0.0, 0.0, 0.0
    
    def get_mid_time(self, timer_list: List[float]) -> Optional[float]:
        arr = np.asarray(timer_list, dtype=np.float64)
        return float(np.median(arr))
    
    def get_model_decode_time(self, model_id: str, stage_name: str) -> Tuple[Optional[float], int]:
        timer, last_update, ema_time, median_val = self._get_timer(f"{stage_name}_{model_id}")
    
    # def get_model_decode_time_dict(self) -> Dict[str, Optional[float]]:
    #     return self._model_decode_timers
    
    def get_model_midtime(self, model_id: str, stage_name: str) -> Tuple[Optional[float], int]:
        timer, last_update, ema_time, median_val = self._get_timer(f"{stage_name}_{model_id}")
        # w = 1.0
        # w = 0.5
        if not timer:
            return None
        # arr = np.asarray(timer, dtype=np.float64)
        # median_val = np.median(arr)
        # stage_hybrid_time = w * median_val + (1.0 - w) * ema_time
        return median_val
    
    def get_model_hybtime(self, model_id: str, stage_name: str) -> Tuple[Optional[float], int]:
        timer, last_update, ema_time, median_val = self._get_timer(f"{stage_name}_{model_id}")
        w = 1.0
        # w = 0.5
        if not timer:
            return None
        # arr = np.asarray(timer, dtype=np.float64)
        # median_val = np.median(arr)
        stage_hybrid_time = w * median_val + (1.0 - w) * ema_time
        return stage_hybrid_time

    def get_model_avgtime(self, model_id: str, stage_name: str) -> Tuple[Optional[float], int]:
        timer, last_update, ema_time, median_val = self._get_timer(f"{stage_name}_{model_id}")
        if not timer:
            return None, 0
        arr = np.asarray(timer, dtype=np.float64)
        return float(arr.mean()), int(arr.size)

    def get_model_lasttime(self, model_id: str, stage_name: str) -> Optional[float]:
            # f"forward_{model_id}",
            # f"time_{model_id}",
            # f"draft_{model_id}",
            # f"verify_{model_id}",
            # f"autoregressive_{model_id}",
            # f"prefill_{model_id}",
        # timer, _ = self._get_timer(
        #         f"{stage_name}_{model_id}"
        # )
        # if not timer:
        #     return None
        timer, last_update, ema_time, median_val = self._get_timer(f"{stage_name}_{model_id}")
        # w = 1.0
        # w = 0.5
        if not timer:
            return None
        return timer[last_update]
    
    def get_model_lut_time(self, model_id: str, stage_name: str, verify_count_key: int, safe_distance: int=0) -> Optional[float]:
        range_name = f"{stage_name}_{model_id}"
        if range_name not in self._LUT_timers:
            return None
        # import logging
        # logger = logging.getLogger(__name__)
        # logger.info(f"LUT_timers: {self._LUT_timers[range_name]._x}, {self._LUT_timers[range_name]._y}")
        return self._LUT_timers[range_name].query(verify_count_key, safe_distance)

    def gen_model_time_dict(
        self, model_ids_tuple: Iterable[Tuple[str, str]]
    ) -> Dict[str, Optional[float]]:
        # mid: (model_id, stage_name)
        return {mid[0]: self.get_model_midtime(mid[0], mid[1]) for mid in model_ids_tuple}

    # ------------------------------------------------------------------
    # Summary / export helpers
    # ------------------------------------------------------------------
    def summarize(self) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {"timers": {}, "counters": {}}

        for name, times in self._timers.items():
            if not times:
                continue
            arr = np.asarray(times, dtype=np.float64)
            summary["timers"][name] = {
                "count": int(arr.size),
                "mean_ms": float(arr.mean() * 1000.0),
                "min_ms": float(arr.min() * 1000.0),
                "max_ms": float(arr.max() * 1000.0),
                "p90_ms": float(np.percentile(arr, 90) * 1000.0),
                "p99_ms": float(np.percentile(arr, 99) * 1000.0),
                "total_ms": float(arr.sum() * 1000.0),
            }

        for name, values in self._counters.items():
            if not values:
                continue
            if all(isinstance(v, (int, float)) for v in values):
                arr = np.asarray(values, dtype=np.float64)
                summary["counters"][name] = {
                    "count": int(arr.size),
                    "mean": float(arr.mean()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "sum": float(arr.sum()),
                }
            else:
                summary["counters"][name] = {"values": values}

        self._summary = summary

        if self.summary_to_console:
            print("\n===== Chain Performance Summary =====")
            for name, stats in summary["timers"].items():
                print(
                    f"{name}: count={stats['count']}, mean={stats['mean_ms']:.2f} ms, "
                    f"min={stats['min_ms']:.2f} ms, max={stats['max_ms']:.2f} ms, "
                    f"p90={stats['p90_ms']:.2f} ms, p99={stats['p99_ms']:.2f} ms"
                )
        return summary

    def export_to_json(self, filename: str) -> None:
        payload = {
            "timers": {k: [float(v) for v in vs] for k, vs in self._timers.items()},
            "counters": {k: list(vs) for k, vs in self._counters.items()},
        }
        with open(filename, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------
    def reset_profiler(self) -> None:
        self._timers.clear()
        self._counters.clear()
        self._thread_local = threading.local()
        self._thread_local.active_ranges = []  # type: ignore[attr-defined]
        self._summary = None

    reset = reset_profiler

class InterpolatedLookupTable:
    __slots__ = ('_x', '_y', 'n_record')

    def __init__(self):
        self._x = []
        self._y = []
        self.n_record = 0

    def update(self, x: float, y: float, alpha: float = 1.0):
        idx = bisect.bisect_left(self._x, x)
        if idx < len(self._x) and self._x[idx] == x:
            self._y[idx] = (1 - alpha) * self._y[idx] + alpha * y
        else:
            self._x.insert(idx, x)
            self._y.insert(idx, y)
            self.n_record += 1

    # def query(self, x: float, safe_distance: int) -> float:
    #     n = len(self._x)
    #     default_val = 0.0
    #     if n == 0:
    #         return default_val

    #     idx = bisect.bisect_left(self._x, x)

    #     # --- Distance Check Start ---
    #     dist_left = abs(x - self._x[idx-1]) if idx > 0 else float('inf')
    #     dist_right = abs(x - self._x[idx]) if idx < n else float('inf')
        
    #     if min(dist_left, dist_right) > safe_distance:
    #         return default_val
    #     # --- Distance Check End ---

    #     if n == 1:
    #         return self._y[0]

    #     i = max(1, min(idx, n - 1))
        
    #     x0, x1 = self._x[i-1], self._x[i]
    #     y0, y1 = self._y[i-1], self._y[i]

    #     if x1 == x0:
    #         return y0
            
    #     return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def query(self, x: float, safe_distance: int) -> float:
        n = len(self._x)
        default_val = 0.0
        if n == 0:
            return default_val

        idx = bisect.bisect_left(self._x, x)

        # --- Distance Check Start ---
        dist_left = abs(x - self._x[idx-1]) if idx > 0 else float('inf')
        dist_right = abs(x - self._x[idx]) if idx < n else float('inf')
        
        if min(dist_left, dist_right) > safe_distance:
            return default_val
        # --- Distance Check End ---

        if n == 1:
            return self._y[0]

        i = max(1, min(idx, n - 1))
        
        x0, x1 = self._x[i-1], self._x[i]
        y0, y1 = self._y[i-1], self._y[i]

        if x1 == x0:
            return y0
            
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

__all__ = ["ChainPerformanceProfiler", "InterpolatedLookupTable"]
