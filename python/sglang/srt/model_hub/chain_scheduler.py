"""Speculative chain scheduler inspired by the sdc reference system.

The implementation keeps the overall data model of
``cite/sdc/utils/scheduler.py`` but removes dependencies on the rest of the
sdc codebase so it can live entirely inside ``model_hub``.  It tracks per
model latency statistics, maintains empirical similarity scores between
models and predicts the expected token latency for different draft chains.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import logging
import itertools

logger = logging.getLogger(__name__)

from sglang.srt.model_hub.chain_profiler import ChainPerformanceProfiler
from sglang.srt.model_hub.utils.nvtx_marker import count_time


class TimeManager:
    def __init__(self, num_models: int, device: str = "cuda", alpha: float = 0.1, dtype: Optional[torch.dtype] = None):
        self.num_models = num_models
        self.device = device
        self.alpha = alpha
        self.dtype = dtype

        self.warmup_decode_base = torch.zeros(num_models, device=device)
        self.warmup_decode_slope = torch.zeros(num_models, device=device)
        self.warmup_verify_slope = torch.zeros(num_models, device=device)
        self.warmup_verify_bias = torch.zeros(num_models, device=device)

        self.runtime_decode_unit = torch.zeros(num_models, device=device)
        self.runtime_verify_unit = torch.zeros(num_models, device=device)
        self.runtime_verify_zero = torch.zeros(num_models, device=device)
        self.runtime_verify_batch = torch.zeros(num_models, device=device)
        
        self.has_runtime_data = torch.zeros(num_models, dtype=torch.bool, device=device)

    def set_warmup_data(self, base_lat: torch.Tensor, dec_slope: torch.Tensor, ver_slope: torch.Tensor, ver_bias: torch.Tensor):
        self.warmup_decode_base = base_lat.to(self.device)
        self.warmup_decode_slope = dec_slope.to(self.device)
        self.warmup_verify_slope = ver_slope.to(self.device)
        self.warmup_verify_bias = ver_bias.to(self.device)
        
        self.runtime_decode_unit = dec_slope.clone().to(self.device)
        self.runtime_verify_unit = ver_slope.clone().to(self.device)

    def update_runtime(self, model_idx: int, phase: str, time_s: float, workload: int):
        unit_time = time_s / max(1, workload)
        
        # if phase == 'decode':
        #     old = self.runtime_decode_unit[model_idx]
        #     self.runtime_decode_unit[model_idx] = (1 - self.alpha) * old + self.alpha * unit_time
        # elif phase == 'verify':
        #     old = self.runtime_verify_unit[model_idx]
        #     self.runtime_verify_unit[model_idx] = (1 - self.alpha) * old + self.alpha * unit_time
        if phase == 'decode':
            # old = self.runtime_decode_unit[model_idx]
            self.runtime_decode_unit[model_idx] = unit_time
        elif phase == 'verify':
            # old = self.runtime_verify_unit[model_idx]
            self.runtime_verify_unit[model_idx] = unit_time
            # self.runtime_verify_batch[model_idx] = record_batch_size
            
        self.has_runtime_data[model_idx] = True

    def get_effective_vectors(self, current_bs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_static = self.warmup_decode_slope * current_bs
        t_dynamic = self.runtime_decode_unit * current_bs
        t_slope_max = torch.max(t_static, t_dynamic)
        
        t_decode_final = torch.max(self.warmup_decode_base, t_slope_max)
        # logger.info(f"t_decode_final: warmup_decode_base: {self.warmup_decode_base}, t_static: {t_static}, t_dynamic: {t_dynamic}")
        # t_verify_slope_final = torch.max(self.warmup_verify_slope, self.runtime_verify_unit)
        t_verify_slope_final = self.warmup_verify_slope
        t_verify_bias_final = self.warmup_verify_bias
        effective_runtime_unit = self.runtime_verify_unit


        # bs_diff = torch.abs(self.runtime_verify_batch - current_bs)
        
        # threshold = 10
        # is_valid_mask = bs_diff <= threshold
        
        # effective_runtime_unit = torch.where(
        #     is_valid_mask, 
        #     self.runtime_verify_unit, 
        #     self.runtime_verify_zero
        # )

        return t_decode_final, t_verify_slope_final, t_verify_bias_final, effective_runtime_unit
    
    def print_time_data(self):
        logger.info(f"warmup_decode_base: {self.warmup_decode_base}")
        logger.info(f"warmup_decode_slope: {self.warmup_decode_slope}")
        logger.info(f"warmup_verify_slope: {self.warmup_verify_slope}")
        logger.info(f"runtime_decode_unit: {self.runtime_decode_unit}")
        logger.info(f"runtime_verify_unit: {self.runtime_verify_unit}")


class CascadeOptimizer:
    def __init__(
        self, 
        model_names: List[str], 
        time_manager: TimeManager,
        compatibility_matrix: torch.Tensor,
        max_window_size: int = 10,
        dtype: Optional[torch.dtype] = None
    ):
        self.model_names = model_names
        self.num_models = len(model_names)
        self.max_window = max_window_size
        self.device = time_manager.device
        self.tm = time_manager
        self.compatibility_matrix = compatibility_matrix
        self.dtype = dtype

        self.E_lookup = torch.zeros(
            (self.num_models, self.num_models, self.max_window), 
            device=self.device,
            dtype=self.dtype
        )
        self.alpha_matrix = torch.zeros((self.num_models, self.num_models), device=self.device, dtype=self.dtype)
        
        self.dp_cost = torch.zeros(self.num_models, device=self.device)
        self.dp_prev = torch.full((self.num_models,), -1, dtype=torch.long, device=self.device)
        self.dp_gamma = torch.zeros(self.num_models, dtype=torch.long, device=self.device)
        self.dp_verify_cache = torch.zeros(self.num_models, device=self.device)
        
        self.window_range = torch.arange(1, self.max_window + 1, device=self.device, dtype=self.dtype)
        
        self.alpha_dirty = True

    def update_alpha(self, alpha_matrix: torch.Tensor):
        self.alpha_matrix = torch.clamp(alpha_matrix.to(self.device), 0.0, 0.999)
        self.alpha_dirty = True

    def _recompute_E_lookup(self):
        alpha_unsqueezed = self.alpha_matrix.unsqueeze(-1)
        exponent = self.window_range + 1.0
        numerator = 1.0 - torch.pow(alpha_unsqueezed, exponent)
        denominator = 1.0 - alpha_unsqueezed
        self.E_lookup = numerator / (denominator + 1e-9)
        self.alpha_dirty = False

        # logger.info(f"E_lookup: {self.E_lookup}, alpha_matrix: {self.alpha_matrix}, window_range: {self.window_range}")
        # logger.info(f"numerator: {numerator}, denominator: {denominator}")
     


    def solve_blind_permutation(self, current_bs: int):
        """
        盲目全排列枚举版本 (修复死循环版)。
        
        修正点：
        1. 增加了 DAG 约束 (u < v)：虽然全排列会生成逆序序列，但我们在检查阶段显式拒绝它们。
           这保证了 dp_prev 不会出现环 (Cycle)。
        """
        # 1. 基础数据准备
        # if self.alpha_dirty:
        self._recompute_E_lookup()

        t_decode, t_verify_slope = self.tm.get_effective_vectors(current_bs)
        t_decode_list = t_decode.tolist()
        t_verify_slope_list = t_verify_slope.tolist()

        # 初始化 DP 表
        self.dp_cost[0] = t_decode_list[0]
        self.dp_prev[0] = -1
        self.dp_gamma[0] = 0
        self.dp_verify_cache[0] = 0.0
        
        # 将其余节点的 Cost 初始化为无穷大
        for i in range(1, self.num_models):
            self.dp_cost[i] = float('inf')
            self.dp_prev[i] = -1  # 默认无前驱

        # 2. 盲目枚举
        # 警告：复杂度极高
        for length in range(2, self.num_models + 1):
            
            # 生成所有排列组合
            all_sequences = itertools.product(range(self.num_models), repeat=length)
            
            for seq in all_sequences:
                # seq: (0, 5, 2)
                
                # [检查1] 起点必须是 0
                if seq[0] != 0:
                    continue
                
                target_idx = seq[-1]
                if target_idx == 0: continue

                # [检查2] 路径有效性验证
                is_valid_path = True
                path_cost = 0.0
                last_node_win_size = 0
                
                # 模拟路径重算
                current_accumulated_cost = t_decode_list[0] 
                
                for step_i in range(len(seq) - 1):
                    u = seq[step_i]
                    v = seq[step_i + 1]
                    
                    # --- [关键修正] 强制拓扑序 ---
                    # 如果出现了 5->2 (u >= v) 或者 2->2，直接视为非法路径。
                    # 这步检查虽然简单，但对于防止死循环至关重要。
                    # 它不会减少 all_sequences 的生成数量，所以依然很慢（低效）。
                    if u >= v:
                        is_valid_path = False
                        break
                    
                    # 检查物理连接
                    if not self.compatibility_matrix[u, v]:
                        is_valid_path = False
                        break
                    
                    # --- 极度低效的 Cost 计算 ---
                    best_step_cost = float('inf')
                    best_step_win = 1
                    
                    draft_cost_unit = current_accumulated_cost
                    target_decode_time = t_decode_list[v]
                    target_verify_unit = t_verify_slope_list[v]
                    
                    for w_idx in range(self.max_window):
                        win_size = w_idx + 1
                        total_tokens = current_bs * (win_size + 1.0)
                        pred_verify_time = target_verify_unit * total_tokens
                        final_verify_time = max(pred_verify_time, target_decode_time)
                        final_gen_time = win_size * draft_cost_unit
                        
                        numerator = final_verify_time + final_gen_time
                        e_val = self.E_lookup[u, v, w_idx].item()
                        
                        if e_val < 1e-9: step_val = float('inf')
                        else: step_val = numerator / e_val
                        
                        if step_val < best_step_cost:
                            best_step_cost = step_val
                            best_step_win = win_size
                    
                    current_accumulated_cost = best_step_cost
                    if step_i == len(seq) - 2:
                        last_node_win_size = best_step_win

                # [更新结果]
                if is_valid_path:
                    # 如果这条路径比之前记录的更短
                    if current_accumulated_cost < self.dp_cost[target_idx]:
                        self.dp_cost[target_idx] = current_accumulated_cost
                        self.dp_prev[target_idx] = seq[-2]
                        self.dp_gamma[target_idx] = last_node_win_size
                        
                        # 同步更新 verify cache
                        prev_node = seq[-2]
                        target_decode_time = t_decode_list[target_idx]
                        target_verify_unit = t_verify_slope_list[target_idx]
                        verify_tokens = current_bs * (last_node_win_size + 1.0)
                        pred_verify = target_verify_unit * verify_tokens
                        final_verify = max(pred_verify, target_decode_time)
                        denom = target_verify_unit * current_bs
                        denom = denom if denom > 1e-9 else 1e-9
                        self.dp_verify_cache[target_idx] = final_verify / denom

    def solve_path_explosion(self, current_bs: int):
        """
        全路径爆炸穷举版本 (Path Explosion Baseline)。
        
        极端低效点：
        1. 抛弃最优子结构：不寻找子问题的最优解，而是列举所有子路径。
        2. 路径存储：显式构建并存储所有从 Root 到 Current 的完整路径列表。
        3. 重复评估：对于每条路径，重新计算整条链上的所有开销。
        """
        # 1. 基础数据准备
        # if self.alpha_dirty:
        self._recompute_E_lookup()

        t_decode, t_verify_slope = self.tm.get_effective_vectors(current_bs)
        t_decode_list = t_decode.tolist()
        t_verify_slope_list = t_verify_slope.tolist()

        # -----------------------------------------------------------
        # 辅助函数 A: 寻找所有到达 target_idx 的“路径链”
        # 返回值示例: [[0, 1, 3], [0, 2, 3], [0, 3]]
        # -----------------------------------------------------------
        def find_all_paths_to(target_idx):
            # Base Case: 起点
            if target_idx == 0:
                return [[0]]
            
            all_paths = []
            # 遍历所有可能的上游节点
            for j in range(target_idx):
                if self.compatibility_matrix[j, target_idx]:
                    # 递归获取到达 j 的所有路径
                    # 这里的递归会产生组合爆炸
                    sub_paths = find_all_paths_to(j)
                    
                    # 将当前节点拼接到每一条子路径后面
                    for path in sub_paths:
                        new_path = path + [target_idx]
                        all_paths.append(new_path)
            
            return all_paths

        # -----------------------------------------------------------
        # 辅助函数 B: 计算一条完整路径的 Cost
        # Path 示例: [0, 2, 5] (表示 0->2->5)
        # -----------------------------------------------------------
        def evaluate_full_path(path_chain):
            # 这里的逻辑是：沿着这条路径，模拟一次完整的推演
            # 我们需要知道这条路径上，每一跳选择了哪个 window_size 才能算分
            # 为了“极致暴力”，我们在评估路径时，甚至需要遍历路径上每一条边的 window_size 组合
            # 但为了保持代码长度可读，这里我们简化为：
            # "路径确定的情况下，针对最后一步 (prev->target)，遍历所有 window_size"
            # 也就是说，我们假设路径上之前的节点的 Cost 已经固定（或者我们可以递归地算，但太复杂了）
            # 为了符合您的要求，我们只计算到达 path_chain[-1] 的成本
            
            # 提取最后一步的关系
            target_idx = path_chain[-1]
            prev_idx = path_chain[-2] if len(path_chain) > 1 else -1
            
            # 如果是起点
            if prev_idx == -1:
                return t_decode_list[0], -1, 0
            
            # 为了计算当前步的 Cost，我们需要知道 prev_idx 的 Cost
            # 在全路径逻辑下，prev_idx 的 Cost 取决于 path_chain[:-1] 这条子路径
            # 这是一个递归评估过程！
            prev_cost, _, _ = evaluate_full_path(path_chain[:-1])
            
            draft_cost_unit = prev_cost
            target_decode_time = t_decode_list[target_idx]
            target_verify_unit = t_verify_slope_list[target_idx]
            
            # 穷举当前这一跳的所有 Window Size
            candidates_for_this_hop = []
            
            for w_idx in range(self.max_window):
                win_size = w_idx + 1
                
                total_tokens = current_bs * (win_size + 1.0)
                pred_verify_time = target_verify_unit * total_tokens
                final_verify_time = max(pred_verify_time, target_decode_time)
                
                final_gen_time = win_size * draft_cost_unit
                numerator = final_verify_time + final_gen_time
                
                e_val = self.E_lookup[prev_idx, target_idx, w_idx].item()
                
                if e_val < 1e-9:
                    cost = float('inf')
                else:
                    cost = numerator / e_val
                
                candidates_for_this_hop.append((cost, prev_idx, win_size))
            
            # 对这一跳的所有 window 选项排序，选最好的
            candidates_for_this_hop.sort(key=lambda x: x[0])
            best_hop = candidates_for_this_hop[0]
            
            return best_hop # (cost, prev, win)

        # 2. 主循环
        self.dp_cost[0] = t_decode_list[0]
        self.dp_prev[0] = -1
        self.dp_gamma[0] = 0
        self.dp_verify_cache[0] = 0.0

        for i in range(1, self.num_models):
            # 第一步：列举所有能到达 i 的路径 [ [0...i], [0...j...i], ... ]
            all_possible_paths = find_all_paths_to(i)
            
            # 第二步：评估每一条路径的最终代价
            path_results = []
            for path in all_possible_paths:
                # 这种写法极其低效，因为 evaluate_full_path 内部又递归调用了自己
                # 导致同一段路径被重复计算了无数次
                final_cost, prev_node, best_win = evaluate_full_path(path)
                path_results.append((final_cost, prev_node, best_win))
            
            # 第三步：在所有路径中选一个最好的
            # 这里就是 "Global Search"
            path_results.sort(key=lambda x: x[0])
            best_of_all_paths = path_results[0]
            
            # 填入结果
            self.dp_cost[i] = best_of_all_paths[0]
            self.dp_prev[i] = best_of_all_paths[1]
            self.dp_gamma[i] = best_of_all_paths[2]

            # 填充 verify cache (保持逻辑一致)
            prev = best_of_all_paths[1]
            win = best_of_all_paths[2]
            target_decode_time = t_decode_list[i]
            target_verify_unit = t_verify_slope_list[i]
            
            verify_tokens = current_bs * (win + 1.0)
            pred_verify = target_verify_unit * verify_tokens
            final_verify = max(pred_verify, target_decode_time)
            
            denom = target_verify_unit * current_bs
            denom = denom if denom > 1e-9 else 1e-9
            self.dp_verify_cache[i] = final_verify / denom
    

    def solve_pure_brute_force(self, current_bs: int):
        """
        纯暴力全空间穷举版本。
        特征：
        1. 递归查找前驱节点成本 (指数级复杂度)。
        2. 显式遍历所有窗口大小 (去除向量化)。
        3. "列举-排序-选择" 逻辑。
        """
        # 1. 准备基础数据
        # if self.alpha_dirty:
        self._recompute_E_lookup()

        t_decode, t_verify_slope = self.tm.get_effective_vectors(current_bs)
        
        # 将 Tensor 转为 Python list/float，确保逻辑上是纯标量计算
        # 这步操作本身就很低效，符合要求
        t_decode_list = t_decode.tolist()
        t_verify_slope_list = t_verify_slope.tolist()

        # 定义递归函数
        def get_all_possibilities_recursive(target_idx):
            """
            返回到达 target_idx 的最优 (cost, prev, win)
            """
            # Base Case: Anchor Model
            if target_idx == 0:
                return t_decode_list[0], -1, 0

            target_decode_time = t_decode_list[target_idx]
            target_verify_unit = t_verify_slope_list[target_idx]
            
            # --- 核心逻辑：列出所有可能性 ---
            # 这是一个列表，用来存放所有 (cost, prev_model, window_size) 的元组
            all_candidates = []

            # 默认方案：不级联 (Self-Decoding)
            # 相当于 prev=-1, win=0
            all_candidates.append((target_decode_time, -1, 0))

            # 遍历每一个可能的上游模型 j
            for j in range(target_idx):
                if not self.compatibility_matrix[j, target_idx]:
                    continue
                
                # [递归] 获取前驱模型 j 的最优成本 (极其低效)
                prev_cost, _, _ = get_all_possibilities_recursive(j)
                draft_cost_unit = prev_cost
                
                # 遍历每一个可能的窗口大小 w
                # 显式 For 循环，替代原来的 torch.min
                for w_idx in range(self.max_window):
                    win_size = w_idx + 1  # 实际窗口大小 (1, 2, 3...)
                    
                    # --- 标量数学计算 ---
                    total_tokens = current_bs * (win_size + 1.0)
                    pred_verify_time = target_verify_unit * total_tokens
                    
                    # 标量 max
                    final_verify_time = pred_verify_time if pred_verify_time > target_decode_time else target_decode_time
                    
                    final_gen_time = win_size * draft_cost_unit
                    numerator = final_verify_time + final_gen_time
                    
                    # 获取对应的接受率 E (标量读取)
                    # 假设 E_lookup 是 [num_models, num_models, window_range]
                    e_val = self.E_lookup[j, target_idx, w_idx].item()
                    
                    # 计算最终 Cost
                    # 避免除以 0
                    if e_val < 1e-6:
                        candidate_cost = float('inf')
                    else:
                        candidate_cost = numerator / e_val
                    
                    # 将这一种可能性加入列表
                    all_candidates.append((candidate_cost, j, win_size))
            
            # --- 排序逻辑 ---
            # 对列表进行全排序，取第一个 (成本最低的)
            # x[0] 是 cost
            all_candidates.sort(key=lambda x: x[0])
            
            best_choice = all_candidates[0]
            return best_choice  # (best_cost, best_prev, best_win)

        # 2. 主循环：为了保持 API 一致性，填充数组
        
        # 初始化
        self.dp_cost[0] = t_decode_list[0]
        self.dp_prev[0] = -1
        self.dp_gamma[0] = 0
        self.dp_verify_cache[0] = 0.0

        for i in range(1, self.num_models):
            # 获取最优解
            cost, prev, win = get_all_possibilities_recursive(i)
            
            # 填入 Tensor (保持外部接口一致)
            self.dp_cost[i] = cost
            self.dp_prev[i] = prev
            self.dp_gamma[i] = win

            # 计算 verify_cache (保持原有逻辑)
            if prev != -1:
                target_decode_time = t_decode_list[i]
                target_verify_unit = t_verify_slope_list[i]
                
                verify_tokens = current_bs * (win + 1.0)
                pred_verify = target_verify_unit * verify_tokens
                final_verify = max(pred_verify, target_decode_time)
                
                denom = target_verify_unit * current_bs
                denom = denom if denom > 1e-9 else 1e-9
                
                v_gamma_max = final_verify / denom
                self.dp_verify_cache[i] = v_gamma_max
            else:
                self.dp_verify_cache[i] = 0.0
    

    def select_max_window_with_tolerance(self, latencies, tolerance=0.02):
        """
        在容忍度范围内选择最大的窗口索引。
        Args:
            latencies: tensor, 延迟数据
            tolerance: float, 允许比最小值大百分之多少 (0.02 代表 2%)
        """
        # logger.info(f"latencies: {latencies}")
        min_latency, min_idx = torch.min(latencies, dim=0)
        cutoff = min_latency * (1 + tolerance)
        valid_indices = torch.where(latencies <= cutoff)[0]
        
        best_idx = valid_indices[-1]
        best_val = latencies[best_idx]
        
        return best_val, best_idx
        
    @torch.inference_mode()
    def solve(self, current_bs: int):
        if self.alpha_dirty:
            self._recompute_E_lookup()

        t_decode, t_verify_slope, t_verify_bias, t_verify_runtime = self.tm.get_effective_vectors(current_bs)
        
        self.dp_cost[0] = t_decode[0]
        self.dp_prev[0] = -1
        self.dp_gamma[0] = 0
        self.dp_verify_cache[0] = 0.0

        for i in range(1, self.num_models):
            target_decode_time = t_decode[i]
            target_verify_unit = t_verify_slope[i]
            target_verify_bias = t_verify_bias[i]
            t_verify_runtime_unit = t_verify_runtime[i]
            
            best_cost = target_decode_time
            best_prev = -1
            best_win = 0
            
            for j in range(i):
                if not self.compatibility_matrix[j, i]:
                    continue
                draft_cost_unit = self.dp_cost[j]
                
                total_tokens_vec = current_bs * (self.window_range + 1.0)
                pred_verify_time = torch.max(target_verify_unit * total_tokens_vec + target_verify_bias, t_verify_runtime_unit)
                final_verify_time = torch.max(pred_verify_time, target_decode_time)
                # final_verify_time = target_decode_time
                
                final_gen_time = self.window_range * draft_cost_unit
                
                numerator = final_verify_time + final_gen_time
                # logger.info(f"E_lookup: {self.E_lookup}, alpha_matrix: {self.alpha_matrix}")
                E_vals = self.E_lookup[j, i, :]
                # logger.info(f"E_vals: {E_vals}")
                
                candidates = numerator / E_vals
                # if i == self.num_models-1:
                #     logger.info(f"draft_cost_unit: {draft_cost_unit}, \
                #         numerator: {numerator}, \
                #         final_verify_time: {final_verify_time}, \
                #         target_verify_unit: {target_verify_unit}, \
                #         t_verify_runtime_unit: {t_verify_runtime_unit}, \
                #         pred_verify_time: {pred_verify_time}, \
                #         target_decode_time: {target_decode_time}, \
                #         final_gen_time: {final_gen_time}, \
                #         E_vals: {E_vals}, \
                #         candidates: {candidates}")
                
                # min_val, min_idx = torch.min(candidates, dim=0)
                min_val, min_idx = self.select_max_window_with_tolerance(candidates, 0.03)

                
                
                # if min_val < best_cost:
                #     best_cost = min_val
                #     best_prev = j
                #     best_win = min_idx.item() + 1
                if min_val < best_cost*(1-0.03):
                    best_cost = min_val
                    best_prev = j
                    best_win = min_idx.item() + 1
                elif min_val < best_cost*(1+0.03):
                    this_win = min_idx.item() + 1
                    if this_win > best_win:
                        best_win = this_win
                        best_cost = min_val
                        best_prev = j

            
            self.dp_cost[i] = best_cost
            self.dp_prev[i] = best_prev
            self.dp_gamma[i] = best_win

            # if best_prev != -1:
            #     verify_tokens = current_bs * (best_win + 1.0)
            #     pred_verify = torch.max(target_verify_unit * verify_tokens + target_verify_bias, t_verify_runtime_unit)
            #     final_verify = max(pred_verify, target_decode_time)
                
            #     denom = target_verify_unit * current_bs
            #     denom = denom if denom > 1e-9 else 1e-9
                
            #     v_gamma_max = final_verify / denom
            #     self.dp_verify_cache[i] = v_gamma_max
            # else:
            #     self.dp_verify_cache[i] = 0.0

    def get_optimal_strategy(self, target_model_name: str, min_accept_length: int) -> List[Dict]:
        # idx = self.model_names.index(target_model_name)
        # chain = []
        # while idx != -1:
        #     prev = self.dp_prev[idx].item()
        #     gamma = self.dp_gamma[idx].item()
        #     chain.append({
        #         "model": self.model_names[idx],
        #         "cost": self.dp_cost[idx].item(),
        #         "input_gamma": gamma if prev != -1 else 0
        #     })
        #     idx = prev
        # return chain[::-1]
        idx = self.model_names.index(target_model_name)
        chain_indices = []
        while idx != -1:
            chain_indices.append(idx)
            idx = self.dp_prev[idx].item()
        
        chain_indices = chain_indices[::-1]
        chain_result = []

        for list_idx, model_idx in enumerate(chain_indices):
            model_name = self.model_names[model_idx]
            cost = self.dp_cost[model_idx].item()
            gamma_curr = max(self.dp_gamma[model_idx].item(), min_accept_length) # input_gamma
            # gamma_curr = self.dp_gamma[model_idx].item()
            
            if list_idx + 1 < len(chain_indices):
                next_model_idx = chain_indices[list_idx + 1]
                num_steps = max(self.dp_gamma[next_model_idx].item(), min_accept_length)
                # num_steps = self.dp_gamma[next_model_idx].item()
            else:
                num_steps = 1
            
            # verify_gamma_max = self.dp_verify_cache[model_idx].item()
            
            # num_draft_tokens = max(gamma_curr + 1, int(math.ceil(verify_gamma_max)))
            
            chain_result.append({
                "model": model_name,
                "cost": cost,
                "input_gamma": gamma_curr,
                # "verify_gamma_max": verify_gamma_max,
                "spec_config": (int(num_steps), 1, int(gamma_curr+1))
            })
            
        return chain_result

class ChainScheduler:
    """Latency and similarity driven chain selector."""

    def __init__(
        self,
        model_ids: Sequence[str],
        profiler: Optional[ChainPerformanceProfiler] = None,
        *,
        strategy: str = "greedy",
        compatibility_matrix: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_window_size: int = 10,
    ) -> None:
        if not model_ids:
            raise ValueError("ChainScheduler requires at least one model id")

        self.full_model_ids = list(model_ids)
        self.model_ids = list(model_ids)
        self.target_model_id = self.model_ids[-1]
        self.strategy = strategy
        self.profiler = profiler

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device
        self.dtype = dtype

        self.n_models = len(self.full_model_ids)
        self._update_id_to_global_index_map()
        self._update_id_to_index_map()

        self.global_time_dict: Dict[str, float] = {
            model_id: 0.0 for model_id in self.full_model_ids
        }

        self.decode_time_dict: Dict[str, float] = {
            model_id: 0.0 for model_id in self.full_model_ids
        }

        self.global_similarity_matrix = torch.eye(
            self.n_models, device=self.device, dtype=self.dtype
        )

        self.last_sorted_model_chain: Optional[Tuple[str, ...]] = None
        self.sublists_cache: List[List[str]] = []

        self.time_manager = TimeManager(self.n_models, device=str(self.device), dtype=self.dtype)
        self.optimizer = CascadeOptimizer(self.full_model_ids, self.time_manager, compatibility_matrix=compatibility_matrix, max_window_size=max_window_size, dtype=self.dtype)
        self._warmup_done = False
    
    def sync_warmup_data(self, warmup_prompt_len1: int, warmup_prompt_len2: int) -> None:
        if not self.profiler:
            return
        

        logger.info(f"_warmup_done {self._warmup_done}")
        if not self._warmup_done:
            # try:
                
            base_lat_list = []
            dec_slope_list = []
            ver_slope_list = []
            ver_bias_list = []
            
            valid = True
            for mid in self.full_model_ids:
                # base = time_data.get(mid) if query_list[(self.full_model_ids.index(mid)*3)][1] == "warmup_decode" else None
                # if base is None:
                base = self.profiler.get_model_midtime(mid, "warmup_decode") or 1e-3
                dec_val = self.profiler.get_model_midtime(mid, "warmup_decode_bs") or 1e-3
                ver_val = self.profiler.get_model_midtime(mid, "warmup_prefill1") or 1e-3
                ver_val2 = self.profiler.get_model_midtime(mid, "warmup_prefill2") or 1e-3
                ver_slope, ver_bias = self._calc_slope_and_bias(ver_val, ver_val2, warmup_prompt_len1, warmup_prompt_len2)

                base_lat_list.append(base)
                dec_slope_list.append(dec_val)
                ver_slope_list.append(ver_slope)
                ver_bias_list.append(ver_bias)
            # logger.info(f" ver_slope_list: {ver_slope_list}, ver_bias_list: {ver_bias_list}")

            if valid:
                self.time_manager.set_warmup_data(
                    torch.tensor(base_lat_list),
                    torch.tensor(dec_slope_list),
                    torch.tensor(ver_slope_list),
                    torch.tensor(ver_bias_list)
                )
                self._warmup_done = True
            # except Exception:
            #     pass
    
    def _calc_slope_and_bias(self, ver_val: float, ver_val2: float, warmup_prompt_len1: int, warmup_prompt_len2: int) -> Tuple[float, float]:
        # logger.info(f"ver_val: {ver_val}, ver_val2: {ver_val2}, warmup_prompt_len1: {warmup_prompt_len1}, warmup_prompt_len2: {warmup_prompt_len2}")
        ver_slope = (ver_val2 - ver_val) / (warmup_prompt_len2 - warmup_prompt_len1) if ver_val2>ver_val else 0.0
        ver_bias = ver_val - ver_slope * warmup_prompt_len1
        return ver_slope, ver_bias

    def sync_stats_from_profiler(self, current_bs: int, model_verify_windows: Dict[str, int], thresh_bs: int=2) -> None:
        if not self.profiler:
            return

        for idx, mid in enumerate(self.full_model_ids):
            # ar_time = runtime_data.get(mid) if (mid, "autoregressive") in query_list else None
            # if ar_time is None: 
            ar_time = self.profiler.get_model_midtime(mid, "draft")
            
            if ar_time:
                self.time_manager.update_runtime(idx, 'decode', ar_time, current_bs)

            # ver_time = runtime_data.get(mid) if (mid, "verifyK") in query_list else None
            # if ver_time is None:
            ver_time = self.profiler.get_model_midtime(mid, "verify")
            record_batch_size = self.profiler.get_model_lasttime(mid, "verify_bs_record")
            # logger.info(f"record_batch_size: {ver_time} {record_batch_size}")

            if ver_time:
                if abs(record_batch_size-current_bs) >= thresh_bs:
                    lut_ver_time = self.profiler.get_model_lut_time(mid, "verify", current_bs*model_verify_windows[mid], safe_distance=50)
                    self.time_manager.update_runtime(idx, 'verify', lut_ver_time, 1)
                    logger.info(f"lut_ver_time: {lut_ver_time}, and clear")
                    self.profiler.clear_timer(name='verify', model_id=mid, ignore_level=True)
                else:
                    self.time_manager.update_runtime(idx, 'verify', ver_time, 1)

            else:
                self.time_manager.update_runtime(idx, 'verify', 0.0, 1)
        
        # self.time_manager.print_time_data()

    def predict_sublists_time(
        self,
        target_model_id: str,
        batch_size: int,
        similarity_matrix: Optional[torch.Tensor] = None,
        min_accept_length: int = 1,
    ) -> Tuple[List[Dict[str, object]], List[str]]:
        threshold = 0.001
        # with count_time("similarity_matrix"):
        if similarity_matrix is not None:
            diff = torch.norm(self.global_similarity_matrix - similarity_matrix)
            # logger.info(f"similarity_matrix changed: {diff}")
            if diff > threshold:
                self.global_similarity_matrix = similarity_matrix
                self.optimizer.update_alpha(self.global_similarity_matrix)

        # with count_time("optimizer.solve"):
        # logger.info(f"predict_sublists_time, self.alpha_matrix: {self.optimizer.alpha_matrix}, {self.full_model_ids}")
        self.optimizer.solve(current_bs=batch_size)
        # self.optimizer.solve_blind_permutation(current_bs=batch_size)
        # self.optimizer.solve_brute_force(current_bs=batch_size)
        # self.optimizer.solve_pure_brute_force(current_bs=batch_size)
            
        optimal_chain_nodes = self.optimizer.get_optimal_strategy(target_model_id, min_accept_length)
        chain_ids = [node["model"] for node in optimal_chain_nodes]
        
        if not optimal_chain_nodes:
             return []

        time_per_token_s = optimal_chain_nodes[-1]["cost"]
        # time_per_token_s = total_time_s
        
        cascade_item = {
            "chain": chain_ids,
            "time_per_token": time_per_token_s,
            "throughput": batch_size / max(1e-6, time_per_token_s),
            "dp_details": optimal_chain_nodes
        }
        
        return cascade_item

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------
    def _calc_id_to_index_map(self, model_sequence: Sequence[str]) -> Dict[str, int]:
        return {model_id: idx for idx, model_id in enumerate(model_sequence)}

    def _update_id_to_global_index_map(self) -> None:
        self.id_to_global_index = self._calc_id_to_index_map(self.full_model_ids)

    def _update_id_to_index_map(self) -> None:
        self.id_to_index = self._calc_id_to_index_map(self.model_ids)

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------
    def get_models_chain(self) -> List[str]:
        return list(self.model_ids)

    def get_global_time_dict(self) -> Dict[str, float]:
        return dict(self.global_time_dict)

    def get_global_similarity_matrix(self) -> torch.Tensor:
        return self.global_similarity_matrix.clone()

    # ------------------------------------------------------------------
    # Time vector helpers
    # ------------------------------------------------------------------
    def init_global_time_dict(self, new_time_dict: Dict[str, float]) -> None:
        for model_id, value in new_time_dict.items():
            if model_id in self.global_time_dict and value is not None:
                self.global_time_dict[model_id] = float(value)

        self.decode_time_dict = self.global_time_dict.copy()

    # def update_global_time_dict(
    #     self, new_time_dict: Dict[str, Optional[float]]
    # ) -> None:
    #     for model_id, value in new_time_dict.items():
    #         if value is None or math.isnan(value):
    #             continue
    #         if model_id in self.global_time_dict:
    #             self.global_time_dict[model_id] = float(value)
    #         # if realtime_syn and model_id in self.id_to_index:
    #         #     idx = self.id_to_index[model_id]
    #         #     self.time_vector[idx] = float(value)
    #         #     self.time_dict[model_id] = float(value)
    
    # def update_decode_time_dict(
    #     self, new_time_dict: Dict[str, Optional[float]],
    # ) -> None:
    #     for model_id, value in new_time_dict.items():
    #         if value is None or math.isnan(value):
    #             continue
    #         if model_id in self.decode_time_dict:
    #             self.decode_time_dict[model_id] = float(value)

    # def _get_time_vector_from_global(
    #     self, model_sequence: Sequence[str]
    # ) -> Tuple[Dict[str, float], torch.Tensor]:
    #     time_dict: Dict[str, float] = {}
    #     values: List[float] = []
    #     for model_id in model_sequence:
    #         time_val = self.global_time_dict.get(model_id, 0.0)
    #         if time_val <= 0.0:
    #             time_val = 1e-3
    #         time_dict[model_id] = float(time_val)
    #         values.append(float(time_val))
    #     time_vector = torch.tensor(values, device=self.device, dtype=torch.float32)
    #     return time_dict, time_vector
    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------
    # def _get_similarity_matrix_from_global(
    #     self, model_sequence: Sequence[str]
    # ) -> torch.Tensor:
    #     indices = torch.tensor(
    #         [self.id_to_global_index[mid] for mid in model_sequence],
    #         device=self.device,
    #         dtype=torch.int64,
    #     )
    #     # print(f"indices: {indices}, global_similarity_matrix: {self.global_similarity_matrix.shape}")
    #     rows = torch.index_select(self.global_similarity_matrix, 0, indices)
    #     return torch.index_select(rows, 1, indices)

    # ------------------------------------------------------------------
    # Chain enumeration helpers
    # ------------------------------------------------------------------
    # def model_product_unique(
    #     self, model_list: Sequence[str]
    # ) -> List[Tuple[str, str]]:
    #     return [
    #         (model_list[i], model_list[j])
    #         for i in range(len(model_list))
    #         for j in range(i + 1, len(model_list))
    #     ]

    # def _get_sorted_model_ids(self, target_model_id: str) -> List[str]:
    #     target_time = self.global_time_dict.get(target_model_id, 0.0)
    #     if target_time <= 0:
    #         target_time = max(self.global_time_dict.values() or [1.0])
    #     filtered = {
    #         mid: time
    #         for mid, time in self.global_time_dict.items()
    #         if time <= target_time + 1e-6
    #     }
    #     if not filtered:
    #         filtered = {mid: self.global_time_dict[mid] for mid in self.full_model_ids}

    #     sorted_models = sorted(filtered.keys(), key=lambda mid: filtered[mid])
    #     if target_model_id not in sorted_models:
    #         sorted_models.append(target_model_id)
    #     else:
    #         sorted_models = [mid for mid in sorted_models if mid != target_model_id]
    #         sorted_models.append(target_model_id)
    #     return sorted_models

    # def _pre_gen_sublists(self, sorted_model_chain: Sequence[str]) -> List[List[str]]:
    #     drafts = list(sorted_model_chain[:-1])
    #     target = sorted_model_chain[-1]
    #     sublists: List[List[str]] = []
    #     for mask in range(1 << len(drafts)):
    #         chain = [drafts[idx] for idx in range(len(drafts)) if mask & (1 << idx)]
    #         chain.append(target)
    #         sublists.append(chain)
    #     return sublists

    # ------------------------------------------------------------------
    # Chain prediction / sampling
    # ------------------------------------------------------------------
    # def predict_sublists_time(
    #     self,
    #     target_model_id: str,
    #     window_size: int,
    #     batch_size: int,
    #     max_new_tokens: Optional[int] = None,
    #     similarity_matrix: Optional[torch.Tensor] = None,
    # ) -> Tuple[List[Dict[str, object]], List[str]]:
    #     # 计算是否需要更新chain
    #     if window_size <= 0:
    #         window_size = 1

    #     sorted_model_chain = self._get_sorted_model_ids(target_model_id)
    #     cache_key = tuple(sorted_model_chain)
    #     if cache_key != self.last_sorted_model_chain:
    #         self.sublists_cache = self._pre_gen_sublists(sorted_model_chain)
    #         self.last_sorted_model_chain = cache_key
    #     sublists = self.sublists_cache

    #     cascade: List[Dict[str, object]] = []
    #     for chain in sublists:
    #         if len(chain) == 0:
    #             continue
    #         predicted = self._predict_time(chain, window_size, batch_size)
    #         cascade.append({"chain": chain, "time_per_token": predicted, "throughput": batch_size/predicted})

    #     # cascade.sort(key=lambda item: item["time_per_token"])
    #     cascade.sort(key=lambda item: item["throughput"], reverse=True)
    #     return cascade, sorted_model_chain

    # def sampling_model_chain(
    #     self,
    #     cascade_model_list: Sequence[Dict[str, object]],
    #     *,
    #     mode: str = "greedy",
    #     temperature: float = 1.0,
    #     k: int = 0,
    #     p: Tuple[float, float] = (0.7, 0.9),
    #     generated_rate: float = 0.0,
    # ) -> List[str]:
    #     if not cascade_model_list:
    #         return list(self.model_ids)

    #     if mode == "greedy" or temperature <= 0:
    #         return list(cascade_model_list[0]["chain"])

    #     if mode == "topk":
    #         top_chains = (
    #             cascade_model_list
    #             if k <= 0
    #             else cascade_model_list[: min(k, len(cascade_model_list))]
    #         )
    #         weights = np.array(
    #             [1.0 / max(1e-6, item["time_per_token"]) for item in top_chains],
    #             dtype=np.float64,
    #         )
    #         weights /= weights.sum()
    #         choice = np.random.choice(len(top_chains), p=weights)
    #         return list(top_chains[int(choice)]["chain"])

    #     if mode == "topp":
    #         base = cascade_model_list[0]["time_per_token"]
    #         rel_perf = np.array(
    #             [base / max(base, item["time_per_token"]) for item in cascade_model_list],
    #             dtype=np.float64,
    #         )
    #         rel_perf /= rel_perf.sum()
    #         threshold = p[0] if generated_rate < 0.5 else p[1]
    #         cumulative = np.cumsum(rel_perf)
    #         k_idx = np.searchsorted(cumulative, threshold) + 1
    #         k_idx = min(max(k_idx, 1), len(cascade_model_list))
    #         choice = random.randint(0, k_idx - 1)
    #         return list(cascade_model_list[choice]["chain"])

    #     return list(cascade_model_list[0]["chain"])

    # ------------------------------------------------------------------
    # Internal prediction formula
    # ------------------------------------------------------------------
    # def _predict_time(self, model_sequence: Sequence[str], window_size: int, batch_size: int, similarity_matrix: Optional[torch.Tensor] = None) -> float:
    #     if len(model_sequence) == 1:
    #         # time_val = self.global_time_dict.get(model_sequence[0], 1.0)
    #         time_val = self.decode_time_dict.get(model_sequence[0], 1e-3)
    #         return float(max(time_val, 1e-3))
    #     time_dict, time_vector = self._get_time_vector_from_global(model_sequence)
    #     similarity_matrix = self._get_similarity_matrix_from_global(model_sequence)
    #     id_to_index = self._calc_id_to_index_map(model_sequence)

    #     window_vec = torch.full(
    #         (len(model_sequence),),
    #         float(max(window_size, 1)),
    #         device=self.device,
    #         dtype=torch.float32,
    #     )
    #     window_vec[0] = 1.0
        
    #     # if torch.distributed.get_rank() == 0:
    #     #     import logging
    #     #     logger = logging.getLogger(__name__)
    #     #     logger.info(f"time_dict: {time_dict}")
    #     #     logger.info(f"similarity_matrix: {similarity_matrix}")

    #     reversed_sequence = list(reversed(model_sequence))
    #     weights = torch.ones(len(model_sequence), device=self.device, dtype=torch.float32)

    #     for i, current_model in enumerate(reversed_sequence):
    #         if i == len(reversed_sequence) - 1:
    #             if i == 0:
    #                 weights[i] = window_vec[i]
    #             else:
    #                 weights[i] = weights[i - 1] * window_vec[i]
    #             continue

    #         draft_model = reversed_sequence[i + 1]
    #         curr_idx = id_to_index[current_model]
    #         draft_idx = id_to_index[draft_model]
    #         alpha = similarity_matrix[curr_idx, draft_idx]
    #         alpha = torch.clamp(alpha, 0.0, 0.999)

    #         next_window = window_vec[i + 1]
    #         denom = 1.0 - torch.pow(alpha, next_window + 1.0)
    #         denom = torch.where(
    #             torch.abs(denom) < 1e-6,
    #             torch.ones_like(denom),
    #             denom,
    #         )
    #         base_weight = window_vec[i] * (1.0 - alpha) / denom

    #         if i == 0:
    #             weights[i] = base_weight
    #         else:
    #             weights[i] = base_weight * weights[i - 1]

    #     weights = torch.flip(weights, dims=[0])
    #     result = torch.sum(weights * time_vector)
    #     return float(result.item())

    # ------------------------------------------------------------------
    # Chain management
    # ------------------------------------------------------------------
    def update_model_chain(self, new_model_ids: Sequence[str]) -> Tuple[List[str], List[str]]:
        old_chain = list(self.model_ids)
        self.model_ids = list(new_model_ids)
        self._update_id_to_index_map()
        # self._update_time_vector_from_global(self.model_ids)
        # self._update_similarity_matrix_from_global(self.model_ids)
        return old_chain, list(self.model_ids)

    # def visualize_model_chain(self) -> str:
    #     if not self.model_ids:
    #         return "模型链尚未初始化"
    #     lines = ["当前模型链信息:"]
    #     lines.append("模型顺序: " + " -> ".join(self.model_ids))
    #     lines.append("时间向量:")
    #     for model_id in self.model_ids:
    #         lines.append(f"  {model_id}: {self.time_dict[model_id]:.4f}s")
    #     lines.append("相似度矩阵:")
    #     header = "      " + " ".join(f"{mid[-6:]:6s}" for mid in self.model_ids)
    #     lines.append(header)
    #     for idx, model_id in enumerate(self.model_ids):
    #         row = f"{model_id[-6:]:6s} " + " ".join(
    #             f"{float(self.similarity_matrix[idx, j]):.4f}" for j in range(len(self.model_ids))
    #         )
    #         lines.append(row)
    #     return "\n".join(lines)


__all__ = ["ChainScheduler"]
