
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from enum import Enum
import logging
import numpy as np

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch, Req, SamplingBatchInfo
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput, EagleVerifyOutput
# from sglang.srt.model_hub.chain_scheduler import SpeculativeParams
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _clone_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.clone()



class InferenceMode(Enum):
    AUTOREGRESSIVE = "autoregressive"
    SPECULATIVE = "speculative"

@dataclass
class SpeculativeParams:
    """定义单个模型在投机推理中的具体参数"""
    num_steps: int        
    eagle_topk: int        
    num_draft_tokens: int  # 如果它要验证，则会一次性验证多少个token

@dataclass
class ChainStrategyDiff:
    switch_mode: bool = False
    new_mode: InferenceMode = None
    # new_current_chain_ids: List[str]
    # new_num_current_chain: int
    unload_models: List[str] = None
    reload_models: List[str] = None
    check_reload: bool = False
    non_diff: bool = False


class ChainStrategy:
    mode: InferenceMode
    current_chain_ids: List[str]
    num_current_chain: int
    # memory_multiplier: float = 1.0
    # batch_size_threshold: int = 5

    def __init__(self, current_chain_ids: List[str]):
        num_current_chain = len(current_chain_ids)

        self.mode = InferenceMode.AUTOREGRESSIVE if num_current_chain == 1 else InferenceMode.SPECULATIVE
        self.current_chain_ids = current_chain_ids
        self.num_current_chain = num_current_chain
        # self.worker_params = worker_params
        # self.memory_multiplier = memory_multiplier
    
    # def create(self, name: str, mode: InferenceMode, current_chain_ids: List[str], worker_params: Dict[str, SpeculativeParams], memory_multiplier: float = 1.0):
    def set_next_strategy(self, current_chain_ids: List[str]):
        new_strategy = None
        diff = ChainStrategyDiff()
        diff.non_diff = self.current_chain_ids==current_chain_ids
        if not diff.non_diff:
            new_strategy = ChainStrategy(current_chain_ids)
            if new_strategy.mode != self.mode:
                diff.switch_mode = True
                diff.new_mode = new_strategy.mode
            old_model_ids = set(self.current_chain_ids)
            new_model_ids = set(new_strategy.current_chain_ids)
            diff.unload_models = list(old_model_ids - new_model_ids)
            # diff.reload_models = list(new_model_ids - old_model_ids)
            diff.reload_models = [
                model_id 
                for model_id in new_strategy.current_chain_ids 
                if model_id not in old_model_ids
            ]

            diff.check_reload = len(diff.reload_models) > 0
        return new_strategy, diff
    
    def print_strategy(self):
        logger.info(f"current_chain_ids: {self.current_chain_ids}, mode: {self.mode}, num_current_chain: {self.num_current_chain}")

class ModelHistory:
    model_name: str
    request_idx_count: torch.Tensor
    seq_lens: torch.Tensor
    input_ids_for_draft_extend: Optional[torch.Tensor] = None
    out_cache_loc_for_draft_extend: Optional[torch.Tensor] = None
    accept_length_for_draft_extend: Optional[torch.Tensor] = None
    seq_lens_for_draft_extend: Optional[torch.Tensor] = None
    next_token_tensor_for_draft_extend: Optional[torch.Tensor] = None
    special_for_draft_extend: bool = False

    def __init__(self, model_name: str, max_req_num: int, device: torch.device, dtype: torch.dtype):
        self.model_name = model_name
        self.request_idx_count = torch.zeros(max_req_num, dtype=dtype, device=device)
        self.seq_lens = torch.zeros(max_req_num, dtype=dtype, device=device)
    
    def sync_history(self, 
                     target_model_history: 'ModelHistory', 
                     req_pool_indices: torch.Tensor, 
                     seq_lens: torch.Tensor, 
                     full_req_pool_indices: torch.Tensor):
        
        current_req_ids = self.request_idx_count[req_pool_indices]
        current_seq_lens = self.seq_lens[req_pool_indices]
        
        target_req_ids = target_model_history.request_idx_count[req_pool_indices].clone()
        full_target_req_ids = target_model_history.request_idx_count[full_req_pool_indices].clone()
        target_seq_lens = seq_lens.clone()
        
        is_same_req = (current_req_ids == target_req_ids)
        # has_new_req = not is_same_req.all().item()
        
        effective_start_lens = torch.where(is_same_req, current_seq_lens, torch.zeros_like(current_seq_lens))
        extend_lens = torch.where(is_same_req, target_seq_lens - current_seq_lens, target_seq_lens)
        
        # 见不到的直接视为已经结束。
        # self.seq_lens[:] = 0
        # self.request_idx_count = target_model_history.request_idx_count.clone()
        self.request_idx_count[full_req_pool_indices] = full_target_req_ids
        self.seq_lens[req_pool_indices] = target_seq_lens
        
        return effective_start_lens, extend_lens
    
    def add_request_idx_count(self, req_pool_indices: torch.Tensor):
        pool_size = self.request_idx_count.numel()
        if req_pool_indices.numel() > 0:
            rpi_max = req_pool_indices.max().item()
            if rpi_max >= pool_size:
                with open("/tmp/diag_batch_trace.log", "a") as f:
                    f.write(f"CHAIN_UTILS-OOB: add_request_idx_count rpi_max={rpi_max} >= pool_size={pool_size}, "
                            f"rpi={req_pool_indices.tolist()}\n")
                safe_mask = req_pool_indices < pool_size
                req_pool_indices = req_pool_indices[safe_mask]
                if req_pool_indices.numel() == 0:
                    return
        self.request_idx_count[req_pool_indices] += 1


@dataclass
class SubmitInputs:
    accepted_token: Optional[torch.Tensor] = None
    accepted_token_num: Optional[torch.Tensor] = None
    # out_loc_cache_for_token: Optional[torch.Tensor] = None
    # out_loc_cache_num: Optional[torch.Tensor] = None
    # drafted_id: Optional[torch.Tensor] = None
    src_req_pool_indices: Optional[torch.Tensor] = None
    spec_steps: int = 0
    spec_topk: int = 0

    @classmethod
    def create_submit_inputs(cls, max_req_num: int, spec_step_num: int, spec_topk: int, device: torch.device):
        
        # batch_size = req_pool_indices.shape[0]
        # TODO: 暂时只支持topk=1
        assert spec_topk == 1
        # device = req_pool_indices.device
        # accepted_token = torch.full((batch_size, spec_draft_num), 
        #                             13, 
        #                             dtype=torch.long, 
        #                             device=device)
        spec_verify_num = spec_step_num + 1
        src_req_pool_indices = torch.arange(max_req_num, device=device, dtype=torch.long)
        # accepted_token = verified_id.unsqueeze(1).expand(-1, spec_verify_num+1).clone().to(device, dtype=torch.long)
        accepted_token = torch.full((max_req_num, spec_verify_num+1), 13, dtype=torch.long, device=device)
        # accepted_token = torch.zeros((batch_size, spec_verify_num+1), device=device, dtype=torch.long)
        # out_loc_cache_for_token = torch.zeros((max_req_num, spec_verify_num), device=device, dtype=torch.long)
        # drafted_id = torch.zeros((max_req_num, spec_verify_num), device=device, dtype=torch.long)


        accepted_token_num = torch.zeros(max_req_num, dtype=torch.long, device=device)
        # out_loc_cache_num = torch.zeros(max_req_num, dtype=torch.long, device=device)

        # logger.info(f"update_from_verify_output, accepted_token: {accepted_token}, drafted_id: {drafted_id}")

        submit_inputs = cls(
            accepted_token=accepted_token,
            accepted_token_num=accepted_token_num,
            # out_loc_cache_for_token=out_loc_cache_for_token,
            # out_loc_cache_num=out_loc_cache_num,
            # drafted_id=drafted_id,    
            src_req_pool_indices=src_req_pool_indices,
            spec_steps=spec_step_num,
            spec_topk=spec_topk,
        )
        return submit_inputs
    
    def append_verified_id(self, verified_id_list: List[torch.Tensor], req_pool_indices: torch.Tensor):
        device = self.accepted_token.device
        for verified_id in verified_id_list:
            self.accepted_token[req_pool_indices, self.accepted_token_num[req_pool_indices]] = verified_id.to(device, dtype=torch.long).clone()
            self.accepted_token_num[req_pool_indices] += 1
    
    def fill_verified_id(self, verified_id: torch.Tensor, req_pool_indices: torch.Tensor):
        device = self.accepted_token.device
        val_to_fill = verified_id.to(device, dtype=torch.long).unsqueeze(1)
        # self.accepted_token[req_pool_indices, self.accepted_token_num[req_pool_indices]] = verified_id.to(device, dtype=torch.long).clone()
        self.accepted_token[req_pool_indices, :] = val_to_fill
        self.accepted_token_num[req_pool_indices] += 1
    
    def clear_accepted_token(self, req_pool_indices: torch.Tensor):
        self.accepted_token_num[req_pool_indices] = 0

    # @classmethod
    # def create_from_verify_output(cls, verified_id: torch.Tensor, req_pool_indices: torch.Tensor, spec_step_num: int, spec_topk: int, device: torch.device):
        
    #     batch_size = req_pool_indices.shape[0]
    #     # TODO: 暂时只支持topk=1
    #     assert spec_topk == 1
    #     # device = req_pool_indices.device
    #     # accepted_token = torch.full((batch_size, spec_draft_num), 
    #     #                             13, 
    #     #                             dtype=torch.long, 
    #     #                             device=device)
    #     spec_verify_num = spec_step_num + 1
    #     accepted_token = verified_id.unsqueeze(1).expand(-1, spec_verify_num+1).clone().to(device, dtype=torch.long)
    #     # accepted_token = torch.zeros((batch_size, spec_verify_num+1), device=device, dtype=torch.long)
    #     # out_loc_cache_for_token = torch.zeros((batch_size, spec_verify_num), device=device, dtype=torch.long)
    #     # drafted_id = torch.zeros((batch_size, spec_verify_num), device=device, dtype=torch.long)


    #     accepted_token_num = torch.ones(batch_size, dtype=torch.long, device=device)
    #     # out_loc_cache_num = torch.zeros(batch_size, dtype=torch.long, device=device)

    #     # logger.info(f"update_from_verify_output, accepted_token: {accepted_token}, drafted_id: {drafted_id}")

    #     submit_inputs = cls(
    #         accepted_token=accepted_token,
    #         accepted_token_num=accepted_token_num,
    #         # out_loc_cache_for_token=out_loc_cache_for_token,
    #         # out_loc_cache_num=out_loc_cache_num,
    #         # drafted_id=drafted_id,
    #         src_req_pool_indices=req_pool_indices,
    #         spec_steps=spec_step_num,
    #         spec_topk=spec_topk,
    #     )
    #     return submit_inputs


    # def update_from_verify_output(self, verify_output: EagleVerifyOutput, req_pool_indices: torch.Tensor):
    #     device = self.src_req_pool_indices.device
    #     max_len = self.spec_steps + 1

    #     accept_length = verify_output.accept_length
            
    #     total_accepted_this_step = accept_length.to(device, dtype=torch.long) + 1
    #     verified_tokens = verify_output.verified_id.to(device, dtype=torch.long)
    #     out_loc_cache_for_token = verify_output.out_loc_cache_for_token.to(device, dtype=torch.long)
        
    #     B_current = total_accepted_this_step.shape[0]
    #     K_total = verified_tokens.shape[0]
        
    #     if K_total == 0:
    #         return

    #     master = self.src_req_pool_indices.unsqueeze(1)
    #     current = req_pool_indices.unsqueeze(0)
    #     comparison = (master == current)
    #     row_indices, col_indices = comparison.nonzero(as_tuple=True)
    #     master_indices_for_current = torch.empty_like(row_indices)
    #     master_indices_for_current[col_indices] = row_indices

    #     start_cols = self.accepted_token_num[master_indices_for_current]
    #     end_cols_proposed = start_cols + total_accepted_this_step
    #     end_cols_final = torch.clamp(end_cols_proposed, max=max_len+1)
    #     num_to_add_final = torch.clamp(end_cols_final - start_cols, min=0)

    #     valid_forward_num = self.out_loc_cache_num[master_indices_for_current]
    #     valid_forward_num = valid_forward_num + total_accepted_this_step


    #     relative_col_idx = torch.cat([
    #         torch.arange(n, device=device) for n in total_accepted_this_step.tolist()
    #     ])
    #     group_id_for_each_token = torch.repeat_interleave(
    #         torch.arange(B_current, device=device), 
    #         total_accepted_this_step
    #     )
    #     limit_per_token = num_to_add_final[group_id_for_each_token]
    #     mask = (relative_col_idx < limit_per_token)
    #     filtered_tokens = verified_tokens[mask]


    #     filtered_out_loc_cache_for_token = out_loc_cache_for_token[mask]
        
    #     if filtered_tokens.shape[0] > 0:
    #         flat_row_indices_orig = torch.repeat_interleave(
    #             master_indices_for_current, 
    #             total_accepted_this_step
    #         )
    #         flat_row_indices = flat_row_indices_orig[mask]

    #         start_cols_orig = torch.repeat_interleave(start_cols, total_accepted_this_step)
    #         flat_col_indices_orig = start_cols_orig + relative_col_idx
    #         flat_col_indices = flat_col_indices_orig[mask]

    #         self.accepted_token[flat_row_indices, flat_col_indices] = filtered_tokens
    #         self.out_loc_cache_for_token[flat_row_indices, flat_col_indices-1] = filtered_out_loc_cache_for_token
        
    #     self.accepted_token_num[master_indices_for_current] = end_cols_final.clamp(max=max_len)
    #     self.out_loc_cache_num[master_indices_for_current] = valid_forward_num.clamp(max=max_len)

    
    def update_from_verify_output(self, verify_output: EagleVerifyOutput, req_pool_indices: torch.Tensor):
        device = self.src_req_pool_indices.device
        max_len = self.spec_steps + 1

        accept_length = verify_output.accept_length
            
        total_accepted_this_step = accept_length.to(device, dtype=torch.long) + 1
        verified_tokens = verify_output.verified_id.to(device, dtype=torch.long)
        # out_loc_cache_for_token = verify_output.out_loc_cache_for_token.to(device, dtype=torch.long)
        # drafted_id = verify_output.drafted_id.to(device, dtype=torch.long)

        
        B_current = total_accepted_this_step.shape[0]
        K_total = verified_tokens.shape[0]
        
        if K_total == 0:
            return

        # master = self.src_req_pool_indices.unsqueeze(1)
        # current = req_pool_indices.unsqueeze(0)
        # comparison = (master == current)
        # row_indices, col_indices = comparison.nonzero(as_tuple=True)
        # master_indices_for_current = torch.empty_like(row_indices)
        # master_indices_for_current[col_indices] = row_indices

        start_cols = self.accepted_token_num[req_pool_indices]
        end_cols_proposed = start_cols + total_accepted_this_step
        end_cols_final = torch.clamp(end_cols_proposed, max=max_len+1)
        num_to_add_final = torch.clamp(end_cols_final - start_cols, min=0)

        # valid_forward_num = self.out_loc_cache_num[master_indices_for_current]
        # valid_forward_num = valid_forward_num + total_accepted_this_step


        relative_col_idx = torch.cat([
            torch.arange(n, device=device) for n in total_accepted_this_step.tolist()
        ])
        group_id_for_each_token = torch.repeat_interleave(
            torch.arange(B_current, device=device), 
            total_accepted_this_step
        )
        limit_per_token = num_to_add_final[group_id_for_each_token]
        mask = (relative_col_idx < limit_per_token)
        filtered_tokens = verified_tokens[mask]


        # filtered_out_loc_cache_for_token = out_loc_cache_for_token[mask]
        # filtered_drafted_id = drafted_id[mask]
        
        if filtered_tokens.shape[0] > 0:
            flat_row_indices_orig = torch.repeat_interleave(
                req_pool_indices, 
                total_accepted_this_step
            )
            flat_row_indices = flat_row_indices_orig[mask]

            start_cols_orig = torch.repeat_interleave(start_cols, total_accepted_this_step)
            flat_col_indices_orig = start_cols_orig + relative_col_idx
            flat_col_indices = flat_col_indices_orig[mask]

            self.accepted_token[flat_row_indices, flat_col_indices] = filtered_tokens
            # self.out_loc_cache_for_token[flat_row_indices, flat_col_indices-1] = filtered_out_loc_cache_for_token
            # self.drafted_id[flat_row_indices, flat_col_indices-1] = filtered_drafted_id
        
        self.accepted_token_num[req_pool_indices] = end_cols_final.clamp(max=max_len+1)
        # logger.info(f"update_from_verify_output, accepted_token_num: {self.accepted_token_num[master_indices_for_current]}")
        # self.out_loc_cache_num[master_indices_for_current] = valid_forward_num.clamp(max=max_len)
        # logger.info(f"update_from_verify_output, accepted_token: {self.accepted_token}, drafted_id: {self.drafted_id}, total_accepted_this_step: {total_accepted_this_step}, out_loc_cache_for_token: {self.out_loc_cache_for_token}")
        # return self.out_loc_cache_for_token.clone(), self.out_loc_cache_num.clone(), self.accepted_token.clone()
        # logger.info(f"update_from_verify_output, accepted_token: {self.accepted_token[master_indices_for_current, :]}")
        
        # logger.info(f"update_from_verify_output, accepted_token: {self.accepted_token}, drafted_id: {self.drafted_id}, total_accepted_this_step: {total_accepted_this_step}, out_loc_cache_num: {self.out_loc_cache_num}")
        # logger.info(f" accepted_token_num: {self.accepted_token_num}, out_loc_cache_num: {self.out_loc_cache_num}")
    
    def process_finished_req(self, finished_req_pool_indices: List[int], req_pool_indices: torch.Tensor):
        max_len = self.spec_steps + 1
        update_mask = torch.tensor(False, device=self.src_req_pool_indices.device)
        if finished_req_pool_indices:
            finished_ids_tensor = torch.tensor(
                finished_req_pool_indices,
                device=self.src_req_pool_indices.device,
                dtype=self.src_req_pool_indices.dtype
            )
            update_mask = torch.isin(self.src_req_pool_indices, finished_ids_tensor)
            self.accepted_token_num[update_mask] = max_len

        # all_finished_mask = (self.accepted_token_num == max_len)
        all_finished_mask = update_mask
        if not torch.any(all_finished_mask):
            return req_pool_indices, None
            
        all_finished_req_ids = self.src_req_pool_indices[all_finished_mask]
        current_pool_remove_mask = torch.isin(req_pool_indices, all_finished_req_ids)
        keep_reqs_index = torch.where(~current_pool_remove_mask)[0]
        keep_req_pool_indices = req_pool_indices[keep_reqs_index]

        return keep_req_pool_indices, keep_reqs_index
    
    def get_extra_out_loc_cache_num(self) -> int:
        max_len = self.spec_steps + 1
        return (max_len - self.out_loc_cache_num).sum().item()
        
    
    def update_out_loc_cache(self, extra_out_loc_cache: torch.Tensor) -> torch.Tensor:
        max_len = self.spec_steps + 1
        col_indices = torch.arange(max_len, device=self.out_loc_cache_for_token.device)
        mask = col_indices[None, :] >= self.out_loc_cache_num[:, None]
        self.out_loc_cache_for_token[mask] = extra_out_loc_cache
        # print("\n")
        # logger.info(f"update_out_loc_cache, out_loc_cache_for_token: {self.out_loc_cache_for_token}")
        return self.out_loc_cache_for_token.flatten()

    
    def check_submit_flag(self, req_pool_indices: torch.Tensor, spec_steps: int) -> bool:
        # logger.info(f"check_submit_flag, out_loc_cache_for_token: {self.out_loc_cache_for_token}, accepted_token_num: {self.accepted_token_num}")
        # submit_thresh = min(0.9*(self.spec_steps+1), self.spec_steps)
        # submit_thresh = 0.5*(self.spec_steps+1)
        logger.info(f"check_submit_flag, spec_steps: {spec_steps}")
        submit_thresh = 0.5*(spec_steps+1)
        # submit_thresh = 0.3*(spec_steps+1)
        # submit_thresh = 1
        current_accepted_token_num = self.accepted_token_num[req_pool_indices]
        accepted_token_num_mean = current_accepted_token_num.clamp(max=spec_steps+1).float().mean()
        # logger.info(f"accepted_token_num_mean: {accepted_token_num_mean}, submit_thresh: {submit_thresh}")
        submit_flag = accepted_token_num_mean >= submit_thresh
        logger.info(f"check_submit_flag, accepted_token_num_mean: {accepted_token_num_mean}, submit_thresh: {submit_thresh}, submit_flag: {submit_flag}")
        return submit_flag

    def clone(self) -> 'SubmitInputs':
        return SubmitInputs(
            accepted_token=self.accepted_token.clone(),
            accepted_token_num=self.accepted_token_num.clone(),
            out_loc_cache_for_token=self.out_loc_cache_for_token.clone(),
            out_loc_cache_num=self.out_loc_cache_num.clone(),
            src_req_pool_indices=self.src_req_pool_indices.clone(),
            drafted_id=self.drafted_id.clone(),
            spec_steps=self.spec_steps,
            spec_topk=self.spec_topk,
        )
    
    def build_token_list(self, req_pool_indices: torch.Tensor, num_verify_tokens: int) -> List[torch.Tensor]:
        current_accepted_token = self.accepted_token[req_pool_indices, :(num_verify_tokens+1)].clone()
        current_batch_size= current_accepted_token.shape[0]
        # spec_verify_num = spec_verify_num_add1 - 1
        # print(f"accepted_token: {self.accepted_token[:, 1:].shape}")
        verified_id = current_accepted_token[:, 0].contiguous()
        # verified_id = current_accepted_token[:, 0].contiguous()
        # token_list = list(torch.split(current_accepted_token[:, 1:-1], 1, dim=1))
        token_list = list(torch.split(current_accepted_token[:, 1:], 1, dim=1))
        # logger.info(token_list)
        # for t in token_list:
        #     logger.info(f"contiguous  {t.is_contiguous()}")
        #     logger.info(t)
        # token_list = [t.contiguous() for t in raw_split]
        return verified_id, token_list, current_batch_size
        

@dataclass
class BatchSnapshot:
    """保存进入某个模型前的关键状态，用于回滚 / 复用。"""

    model_name: str
    reqs: List[Req]
    seq_lens: Optional[torch.Tensor]
    seq_lens_sum: int
    req_pool_indices: Optional[torch.Tensor]
    # out_cache_loc: Optional[torch.Tensor]
    cache_alloc_state: Optional[torch.Tensor]
    # input_ids: Optional[torch.Tensor]
    # verified_id: Optional[torch.Tensor]
    # forward_mode: Any
    # spec_info: Optional[Any]
    # tree_payload: Optional[TreePayload] = None
    sampling_info_temperatures: Optional[torch.Tensor]
    sampling_info_top_ps: Optional[torch.Tensor]
    sampling_info_top_ks: Optional[torch.Tensor]
    sampling_info_min_ps: Optional[torch.Tensor]
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_batch(
        cls,
        model_name: str,
        batch: ScheduleBatch,
        # tree_payload: Optional[TreePayload] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "BatchSnapshot":
        snapshot = cls(
            model_name=model_name,
            reqs = getattr(batch, "reqs", []),
            seq_lens=_clone_tensor(getattr(batch, "seq_lens", None)),
            seq_lens_sum=getattr(batch, "seq_lens_sum", 0),
            req_pool_indices=_clone_tensor(getattr(batch, "req_pool_indices", None)),
            # out_cache_loc=_clone_tensor(getattr(batch, "out_cache_loc", None)),
            cache_alloc_state = getattr(batch, "token_to_kv_pool_allocator", None).backup_state(),
            # input_ids=_clone_tensor(getattr(batch, "input_ids", None)),
            # verified_id=_clone_tensor(getattr(batch.spec_info, "verified_id", None)),
            # forward_mode=getattr(batch, "forward_mode", None),
            # spec_info=copy.deepcopy(getattr(batch, "spec_info", None)),
            # tree_payload=tree_payload.clone() if tree_payload else None,
            sampling_info_temperatures=_clone_tensor(getattr(batch.sampling_info, "temperatures", None)),
            sampling_info_top_ps=_clone_tensor(getattr(batch.sampling_info, "top_ps", None)),
            sampling_info_top_ks=_clone_tensor(getattr(batch.sampling_info, "top_ks", None)),
            sampling_info_min_ps=_clone_tensor(getattr(batch.sampling_info, "min_ps", None)),
            extra=extra.copy() if extra else {},
        )
        return snapshot
    
    def clone(self, model_name=None, reqs_clone=False) -> "BatchSnapshot":
        if model_name is None:
            model_name = self.model_name
        return BatchSnapshot(
            model_name=model_name,
            reqs=self.reqs if not reqs_clone else [i for i in self.reqs],
            seq_lens=_clone_tensor(self.seq_lens),
            seq_lens_sum=self.seq_lens_sum,
            req_pool_indices=_clone_tensor(self.req_pool_indices),
            cache_alloc_state=_clone_tensor(self.cache_alloc_state),
            # input_ids=_clone_tensor(self.input_ids),
            # verified_id=_clone_tensor(self.verified_id),
            sampling_info_temperatures=_clone_tensor(self.sampling_info_temperatures),
            sampling_info_top_ps=_clone_tensor(self.sampling_info_top_ps),
            sampling_info_top_ks=_clone_tensor(self.sampling_info_top_ks),
            sampling_info_min_ps=_clone_tensor(self.sampling_info_min_ps),
            extra=self.extra.copy(),
        )

    # def apply_to_batch(self, batch: ScheduleBatch) -> None:
    #     if self.seq_lens is not None:
    #         batch.seq_lens = self.seq_lens.clone()
    #     if self.req_pool_indices is not None:
    #         batch.req_pool_indices = self.req_pool_indices.clone()
    #     if self.out_cache_loc is not None:
    #         batch.out_cache_loc = self.out_cache_loc.clone()
    #     if self.input_ids is not None:
    #         batch.input_ids = self.input_ids.clone()
    #     batch.seq_lens_sum = self.seq_lens_sum
    #     batch.forward_mode = self.forward_mode
    #     # batch.spec_info = copy.deepcopy(self.spec_info)
    
    def update_state(self, verify_output: EagleVerifyOutput):
        pass


    def printk(self) -> str:
        
        return (
            f"BatchSnapshot(model={self.model_name}, "
            f"seq_lens={self.seq_lens}, seq_lens_sum={self.seq_lens_sum}, "
            f"req_pool_indices={self.req_pool_indices}, "
            f"reqs={len(self.reqs)}, "
            # f"input_ids={self.input_ids}, "
            # f"verified_id={self.verified_id}, "
            f"sampling_info_temperatures={self.sampling_info_temperatures.shape}, "
            f"extra={self.extra})"
        )


class StateManager:
    """管理各模型的 BatchSnapshot，并辅助构造下一层 verify 输入。"""

    def __init__(self) -> None:
        self._submit_inputs: Dict[str, SubmitInputs] = {}
        self._spec_params: Dict[str, SpeculativeParams] = {}
        self._model_history: Dict[str, ModelHistory] = {}
        self.model_verify_windows: Dict[str, int] = {}
        self.max_req_num: int = 0
        self.cache_verified_id = None
        self.cache_token_list = []
        self.reset_all_state()
    
    
    def reset_all_state(self):
        self._snapshots: Dict[str, BatchSnapshot] = {}
        self.base_snapshot: Optional[BatchSnapshot] = None
        self.cached_score_template: Optional[torch.Tensor] = None
        self.cached_parents_list: List[torch.Tensor] = []
        # self.out_loc_cache_for_token: Optional[torch.Tensor] = None
        # self.out_loc_cache_num: Optional[torch.Tensor] = None
        # self.accept_token: Optional[torch.Tensor] = None
    
    def set_attr(self, max_req_num: int):
        # ReqToTokenPool is allocated with size=max_num_reqs+1, so pool indices
        # can reach max_num_reqs. Match that here to avoid off-by-one OOB.
        self.max_req_num = max_req_num + 1

    def take_snapshot(
        self,
        batch: ScheduleBatch,
        model_name_list: Sequence[str],
        # tree_payloads: Optional[Dict[str, TreePayload]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        for model_name in model_name_list:
            # payload = None
            # if tree_payloads is not None:
            #     payload = tree_payloads.get(model_name)
            snapshot = BatchSnapshot.from_batch(
                model_name=model_name,
                batch=batch,
                # tree_payload=payload,
                extra=extra,
            )
            self._snapshots[model_name] = snapshot
    
    def take_base_snapshot(self, batch: ScheduleBatch):
        self.base_snapshot = BatchSnapshot.from_batch(
            model_name="base",
            batch=batch,
            extra={},
        )

        # logger.info(f"take_base_snapshot: {self.base_snapshot.printk()}")

    # snapshot

    def get_snapshot(self, model_name: str) -> Optional[BatchSnapshot]:
        return self._snapshots.get(model_name, self.base_snapshot.clone(model_name))

    def clear_snapshot(self, model_name: str) -> None:
        self._snapshots.pop(model_name, None)
    
    # submit_inputs

    def get_submit_inputs(self, model_name: str) -> Optional[SubmitInputs]:
        return self._submit_inputs.get(model_name)
    
    def create_submit_input(self, exec_worker_name: str, spec_steps: int, spec_topk: int, device: torch.device):
        # submit_inputs = SubmitInputs.create_from_verify_output(verified_id, req_pool_indices, spec_steps, spec_topk, req_pool_indices.device)
        # self._submit_inputs[exec_worker_name] = submit_inputs
        submit_inputs = self._submit_inputs.get(exec_worker_name)
        if submit_inputs is None:
            # logger.info(f"create ")
            submit_inputs = SubmitInputs.create_submit_inputs(self.max_req_num, spec_steps, spec_topk, device)
            self._submit_inputs[exec_worker_name] = submit_inputs
    
    def append_submit_inputs(self, exec_worker_name: str, verified_id_list: List[torch.Tensor], req_pool_indices: torch.Tensor):
        submit_inputs = self._submit_inputs.get(exec_worker_name)
        if submit_inputs is None:
            raise ValueError(f"submit_inputs is None for model {exec_worker_name}")
        submit_inputs.append_verified_id(verified_id_list, req_pool_indices)
        # logger.info(f"append_submit_inputs, {submit_inputs.accepted_token[req_pool_indices]}")
    
    def fill_submit_inputs(self, exec_worker_name: str, verified_id: torch.Tensor, req_pool_indices: torch.Tensor):
        submit_inputs = self._submit_inputs.get(exec_worker_name)
        if submit_inputs is None:
            raise ValueError(f"submit_inputs is None for model {exec_worker_name}")
        submit_inputs.fill_verified_id(verified_id, req_pool_indices)
        # logger.info(f"append_submit_inputs, {submit_inputs.accepted_token[req_pool_indices]}")
    
    def clear_submit_inputs(self, exec_worker_name: str, req_pool_indices: torch.Tensor):
        submit_inputs = self._submit_inputs.get(exec_worker_name)
        if submit_inputs is None:
            raise ValueError(f"submit_inputs is None for model {exec_worker_name}")
        submit_inputs.clear_accepted_token(req_pool_indices)

    def update_submit_inputs(self, verify_output: EagleVerifyOutput, exec_worker_name: str, req_pool_indices: torch.Tensor, spec_steps: int, spec_topk: int) -> bool:
        submit_inputs = self._submit_inputs.get(exec_worker_name)
        if submit_inputs is None:
            # submit_inputs = SubmitInputs.create_from_verify_output(self.base_snapshot.verified_id, self.base_snapshot.req_pool_indices, spec_steps, spec_topk, req_pool_indices.device)
            # # submit_inputs = SubmitInputs.create_submit_inputs(self.max_req_num, spec_steps, spec_topk, req_pool_indices.device)
            # self._submit_inputs[exec_worker_name] = submit_inputs
            raise ValueError(f"submit_inputs is None for model {exec_worker_name}")
        # else:
        # submit_inputs.update_from_verify_output(verify_output, req_pool_indices)
        # keep_req_pool_indices, keep_reqs_index = submit_inputs.process_finished_req(verify_output.finished_req_pool_indices, req_pool_indices)
        # submit_flag = submit_inputs.check_submit_flag()
        submit_inputs.update_from_verify_output(verify_output, req_pool_indices)
        keep_req_pool_indices, keep_reqs_index = submit_inputs.process_finished_req(verify_output.finished_req_pool_indices, req_pool_indices)
        submit_flag = submit_inputs.check_submit_flag(req_pool_indices, spec_steps)
        # logger.info(f"update_submit_inputs: {submit_inputs.accepted_token[req_pool_indices]}")
        # logger.info(f"submit_flag: {submit_flag}")

        return submit_flag, keep_req_pool_indices, keep_reqs_index
    
    def fit_seq_lens(self, batch: ScheduleBatch, model_name: str):
        # logger.info(f"fit_seq_lens, batch: {batch.seq_lens}")
        submit_inputs = self._submit_inputs.get(model_name)
        if submit_inputs is None:
            # self._submit_inputs[exec_worker_name] = submit_inputs
            raise ValueError(f"submit_inputs is None for model {model_name}")
        accepted_token_num = submit_inputs.accepted_token_num[batch.req_pool_indices]
        # logger.info(f"accepted_token_num: {accepted_token_num} \
        #             fit_seq_lens \
        #             batch.seq_lens: {batch.seq_lens} \
        #             batch.seq_lens_sum: {batch.seq_lens_sum} \
        #             batch.seq_lens.dtype: {batch.seq_lens.dtype} \
        #             ")
        
        batch.seq_lens = batch.seq_lens + accepted_token_num-1
        batch.seq_lens_sum = sum(batch.seq_lens.tolist())
        
        # logger.info(f"accepted_token_num: {accepted_token_num} \
        #             fit_seq_lens \
        #             batch.seq_lens: {batch.seq_lens} \
        #             batch.seq_lens_sum: {batch.seq_lens_sum} \
        #             batch.seq_lens.dtype: {batch.seq_lens.dtype} \
        #             batch.seq_lens_src: {self.base_snapshot.seq_lens} \
        #             batch.seq_lens_src.dtype: {self.base_snapshot.seq_lens.dtype} \
        #             ")
        # logger.info(f"after fit_seq_lens, batch: {batch.seq_lens}")

    def update_reqs(self, batch: ScheduleBatch, keep_req_pool_indices: torch.Tensor, keep_reqs_index: torch.Tensor):
        if keep_req_pool_indices.shape[0] != batch.req_pool_indices.shape[0]:
            batch.reqs = [batch.reqs[i] for i in keep_reqs_index]
            batch.req_pool_indices = batch.req_pool_indices[keep_reqs_index]
            batch.seq_lens = batch.seq_lens[keep_reqs_index]
            batch.seq_lens_sum = batch.seq_lens.sum().item()
            batch.sampling_info.temperatures = batch.sampling_info.temperatures[keep_reqs_index]
            batch.sampling_info.top_ps = batch.sampling_info.top_ps[keep_reqs_index]
            batch.sampling_info.top_ks = batch.sampling_info.top_ks[keep_reqs_index]
            batch.sampling_info.min_ps = batch.sampling_info.min_ps[keep_reqs_index]

    
    def create_max_draft_list(self, max_batch_size, max_spec_steps: int, device: torch.device, dtype: torch.dtype):
        if self.cached_score_template is not None and self.cached_score_template.shape[0] == max_batch_size and len(self.cached_parents_list) >= max_spec_steps:
            return
        
        score_tensor_template = torch.full(
            (max_batch_size, 1), 
            1.0, 
            device=device, 
            dtype=dtype
        )
        # self.cached_score_list = [score_tensor_template] * spec_steps
        self.cached_score_template = score_tensor_template

        self.cached_parents_list = []
        step_0_parent = torch.tensor(
            [[-1, 0]], 
            device=device, 
            dtype=torch.long
        ).repeat(max_batch_size, 1)
        self.cached_parents_list.append(step_0_parent)

        for i in range(1, max_spec_steps):
            step_i_parent = torch.full(
                (max_batch_size, 1), 
                i, 
                device=device, 
                dtype=torch.long
            )
            self.cached_parents_list.append(step_i_parent)
        
    def build_verify_input(
        self,
        snapshot: BatchSnapshot,
        submit_inputs: SubmitInputs,
        req_pool_indices: torch.Tensor,
        num_verify_tokens: int,
        enabled_skip_extend: bool,
        is_top_verify: bool,
        spec_topk: int = 1,
    ) -> EagleVerifyInput:
        # verified_id, token_list, current_batch_size, spec_verify_num = submit_inputs.build_token_list(req_pool_indices)
        # spec_steps = spec_verify_num - 1
        # score_view = self.cached_score_template[:(current_batch_size+1)].contiguous()
        # # score_list = [score_view] * spec_steps
        # score_list = [score_view for _ in range(spec_steps)]
        # parents_list = []
        # for cached_parent_tensor in self.cached_parents_list:
        #     parent_view = cached_parent_tensor[:(current_batch_size+1)].contiguous()
        #     parents_list.append(parent_view)
        # logger.info(f"cxxxxxx")
        assert spec_topk == 1
        verified_id, token_list, current_batch_size = submit_inputs.build_token_list(req_pool_indices, num_verify_tokens)
        # if self.cache_verified_id == None or self.cache_verified_id.shape[0] != verified_id.shape[0]:
        #     self.cache_verified_id = verified_id.clone()
        # if self.cache_token_list == [] or self.cache_token_list[0].shape[0] != token_list[0].shape[0]:
        #     self.cache_token_list = [t.clone() for t in token_list]
        # if self.cache_verified_id == None or self.cache_verified_id.shape[0] != snapshot.seq_lens.shape[0]:
        #     self.cache_verified_id = snapshot.seq_lens.clone()
        # spec_steps = submit_inputs.spec_steps
        spec_steps = num_verify_tokens-1
        # only support tok==1
        # assert spec_steps == num_verify_tokens-1
        score_view = self.cached_score_template[:(current_batch_size+1)]
        score_list = [score_view] * spec_steps
        # score_list = [score_view.clone() for _ in range(spec_steps)]
        parents_list = []

        for cached_parent_tensor in self.cached_parents_list[:spec_steps]:
            parent_view = cached_parent_tensor[:(current_batch_size+1)]
            parents_list.append(parent_view)
        
        # if torch.distributed.get_rank() == 0:
            # logger.info(f" input  \
            #             verified_id:{verified_id}, verify_contiguous {verified_id.is_contiguous()}, \
            #             seq_lens {snapshot.seq_lens}, seq_lens {snapshot.seq_lens.is_contiguous()}, \
            #             token_list {token_list}, \
            #             score_list {score_list}, \
            #             parents_list {parents_list}, \
            #             ")
        # logger.info(f"build_tree_efficient num_verify_tokens: {num_verify_tokens}, parents_list: {len(parents_list)}, \
        # p_shape: {parents_list[0].shape}, score_list: {len(score_list)}, s_shape, {score_list[0].shape}, \
        # submit_inputs.spec_steps: {submit_inputs.spec_steps}, \
        # verified_id: {verified_id.shape}, token_list: {len(token_list)}, t_shape, {token_list[0].shape}")

        # # snapshot.seq_lens_sum, snapshot.seq_lens.sum().item()
        # logger.info(f"num_verify_tokens: {num_verify_tokens}, spec_num: {submit_inputs.spec_steps}")


        return EagleVerifyInput.create(
            verified_id=verified_id,
            token_list=token_list,
            # verified_id=self.cache_verified_id,
            # token_list=self.cache_token_list,
            score_list=score_list,
            parents_list=parents_list,
            seq_lens=_clone_tensor(snapshot.seq_lens),
            # seq_lens=self.cache_verified_id,
            seq_lens_sum=snapshot.seq_lens_sum,
            topk=spec_topk,
            spec_steps=spec_steps,
            num_verify_tokens=num_verify_tokens,
            enabled_skip_extend=enabled_skip_extend,
            is_top_verify=is_top_verify,
        )
    
    # def restore_batch_state(self, batch: ScheduleBatch, snapshot: BatchSnapshot, draft_spec_info: EagleVerifyInput):
    #     batch.seq_lens = snapshot.seq_lens
    #     batch.seq_lens_sum = snapshot.seq_lens_sum
    #     batch.token_to_kv_pool_allocator.restore_state(snapshot.cache_alloc_state)
    #     batch.spec_info = draft_spec_info

    # spec_params
    def init_spec_params(self, server_args_list: List[ServerArgs]):
        target_model_server_args = server_args_list[-1]
        self._spec_params['base_params'] = SpeculativeParams(
            num_steps=target_model_server_args.speculative_num_steps,
            eagle_topk=target_model_server_args.speculative_eagle_topk,
            num_draft_tokens=target_model_server_args.speculative_num_draft_tokens,
        )
        max_speculative_num_steps = 0
        max_speculative_num_topk = 0
        max_speculative_num_draft_tokens = 0

        for server_args in server_args_list:
            self._spec_params[server_args.model_path] = SpeculativeParams(
                num_steps=server_args.speculative_num_steps,
                eagle_topk=server_args.speculative_eagle_topk,
                num_draft_tokens=server_args.speculative_num_draft_tokens,
            )
            self.model_verify_windows[server_args.model_path] = server_args.speculative_num_draft_tokens
            max_speculative_num_steps = max(max_speculative_num_steps, server_args.speculative_num_steps)
            max_speculative_num_topk = max(max_speculative_num_topk, server_args.speculative_eagle_topk)
            max_speculative_num_draft_tokens = max(max_speculative_num_draft_tokens, server_args.speculative_num_draft_tokens)

        return max_speculative_num_steps, max_speculative_num_topk, max_speculative_num_draft_tokens

    def get_spec_params(self, model_name: str) -> SpeculativeParams:
        if model_name not in self._spec_params:
            raise ValueError(f"Model name {model_name} not found in spec_params")
        return self._spec_params[model_name]
    
    def update_spec_params(self, model_name: str, num_steps: int, eagle_topk: int, num_draft_tokens: int):
        model_spec_params = self._spec_params[model_name]
        model_spec_params.num_steps = num_steps
        model_spec_params.eagle_topk = eagle_topk
        model_spec_params.num_draft_tokens = num_draft_tokens
        if model_name != 'base_params':
            self.model_verify_windows[model_name] = num_draft_tokens
    
        
    def add_request_idx_count_from_prefill(self, model_name: str, req_pool_indices: torch.Tensor):
        model_history = self._model_history.get(model_name)

        if model_history is None:
            model_history = ModelHistory(model_name, self.max_req_num, req_pool_indices.device, req_pool_indices.dtype)
        model_history.add_request_idx_count(req_pool_indices)
        self._model_history[model_name] = model_history
    
    def save_model_history(self, model_name_list: List[str], req_pool_indices: torch.Tensor, seq_lens: torch.Tensor, accept_length: torch.Tensor, stage_name: str):
        for model_name in model_name_list:
            model_history = self._model_history[model_name]
            # if accept_length is not None:
            #     model_history.seq_lens[req_pool_indices] = seq_lens.clone() - accept_length.clone() - 1
            #     logger.info(f"save_model_history, accept_length is not None, seq_lens: {seq_lens}, accept_length: {accept_length}")
            # else:
            #     # from prefill
            #     model_history.seq_lens[req_pool_indices] = seq_lens.clone()
            #     logger.info(f"save_model_history, accept_length is None, seq_lens: {seq_lens}, accept_length: {accept_length}")
            if stage_name == "unload":
                model_history.seq_lens[req_pool_indices] = seq_lens.clone()
            else:
                model_history.seq_lens[req_pool_indices] = seq_lens.clone() - accept_length.clone() - 1
            # logger.info(f"save_model_history, accept_length is None, seq_lens: {seq_lens}, accept_length: {accept_length}")
            # model_history.seq_lens[req_pool_indices] = seq_lens.clone() - accept_length.clone() - 1
        pass

    def load_model_history(self, 
                           model_name_list: List[str], 
                           batch: 'ScheduleBatch', 
                           target_model_name: str, 
                           req_pool_indices_for_draft: torch.Tensor, 
                           seq_lens: torch.Tensor, 
                           full_req_pool_indices: torch.Tensor,
                           next_draft_model_name: str, 
                           verified_id: torch.Tensor,
                           drafted_ids: torch.Tensor,
                           draft_lens: torch.Tensor,
                           drafted_out_cache_loc: torch.Tensor,
                           next_out_cache_loc: torch.Tensor,
                        #    worker_map,
                        #    model_hidden_states_pools
                           ):
        
        target_model_history = self._model_history[target_model_name]
        
        active_indices_list = req_pool_indices_for_draft.tolist()
        full_indices_list = full_req_pool_indices.tolist()
        pool_id_map = {pid: i for i, pid in enumerate(full_indices_list)}
        
        verified_end_indices = torch.cumsum(draft_lens, dim=0) - 1
        next_token_tensor = verified_id[verified_end_indices]
        drafted_ids = drafted_ids.to(dtype=verified_id.dtype)
        # logger.info(f"load_model_history, drafted_ids: {drafted_ids.dtype}, verified_id: {verified_id.dtype}")
        
        # drafted_ids_list = drafted_ids.tolist()
        # next_token_list = next_token_tensor.tolist()
        # draft_lens_list = draft_lens.tolist()
        # logger.info(f"drafted_ids_list: {drafted_ids.shape}, next_token_list: {next_token_tensor.shape}, draft_lens_list: {draft_lens}")
        
        for model_name in model_name_list:
            reload_model_history = self._model_history[model_name]
            # draft_worker = worker_map[model_name]
            
            current_seq_lens, extend_lens = reload_model_history.sync_history(
                target_model_history, req_pool_indices_for_draft, seq_lens, full_req_pool_indices
            )

            # if torch.distributed.get_rank() == 0:
            #     logger.info(f"load_model_history, current_seq_lens: {current_seq_lens}, extend_lens: {extend_lens}")
            

            is_next_draft = (model_name == next_draft_model_name)
            
            reload_inputs, reload_out_cache_loc, accept_length = self.gen_reload_inputs_optimized(
                batch=batch,
                active_indices_list=active_indices_list,
                pool_id_map=pool_id_map,
                drafted_ids=drafted_ids,
                next_token_tensor=next_token_tensor,
                draft_lens=draft_lens,
                current_seq_lens=current_seq_lens,
                extend_lens=extend_lens,
                is_next_draft=is_next_draft,
                req_pool_indices_tensor=req_pool_indices_for_draft,
                drafted_out_cache_loc=drafted_out_cache_loc,
                next_out_cache_loc=next_out_cache_loc,
            )
            # if torch.distributed.get_rank() == 0:
            #     logger.info(f"reload_inputs,  {reload_inputs.shape}, reload_out_cache_loc, {reload_out_cache_loc.shape},accept_length:  {accept_length.shape}")
            #     logger.info(f"reload_inputs: {reload_inputs}, reload_out_cache_loc: {reload_out_cache_loc}")
                
            # if draft_worker.speculative_algorithm.is_eagle():
            #     if has_new_req:
            #         hidden_states_pool = model_hidden_states_pools[draft_worker.base_model_name]
            #         reload_model_history.hidden_states_for_draft_extend = hidden_states_pool.read_batch(reload_out_cache_loc)
            #         logger.info(f"reload_model_history. reload_out_cache_loc: {reload_out_cache_loc}")
                
            #         reload_model_history.input_ids_for_draft_extend = reload_inputs
            #         reload_model_history.out_cache_loc_for_draft_extend = reload_out_cache_loc
            #         reload_model_history.accept_length_for_draft_extend = accept_length
            #         reload_model_history.seq_lens_for_draft_extend = current_seq_lens+accept_length+1
            #         reload_model_history.next_token_tensor_for_draft_extend = next_token_tensor
            #         reload_model_history.special_for_draft_extend = True
            #     else:
            #         reload_model_history.special_for_draft_extend = False
            # else:
            #     reload_model_history.input_ids_for_draft_extend = reload_inputs
            #     reload_model_history.out_cache_loc_for_draft_extend = reload_out_cache_loc
            #     reload_model_history.accept_length_for_draft_extend = accept_length
            #     reload_model_history.seq_lens_for_draft_extend = current_seq_lens+accept_length+1
            #     reload_model_history.next_token_tensor_for_draft_extend = next_token_tensor
            #     reload_model_history.hidden_states_for_draft_extend = None
            #     reload_model_history.special_for_draft_extend = True
            reload_model_history.input_ids_for_draft_extend = reload_inputs
            reload_model_history.out_cache_loc_for_draft_extend = reload_out_cache_loc
            reload_model_history.accept_length_for_draft_extend = accept_length
            reload_model_history.seq_lens_for_draft_extend = current_seq_lens+accept_length+1
            reload_model_history.next_token_tensor_for_draft_extend = next_token_tensor
            # reload_model_history.hidden_states_for_draft_extend = None
            reload_model_history.special_for_draft_extend = True

    def gen_reload_inputs_optimized(self, 
                                    batch: 'ScheduleBatch',
                                    active_indices_list: List[int],
                                    pool_id_map: dict,
                                    drafted_ids: torch.Tensor,      # [num_draft_tokens]
                                    next_token_tensor: torch.Tensor,# [batch_size]
                                    draft_lens: torch.Tensor,       # [batch_size]
                                    current_seq_lens: torch.Tensor, # [batch_size]
                                    extend_lens: torch.Tensor,      # [batch_size]
                                    is_next_draft: bool,
                                    req_pool_indices_tensor: torch.Tensor,
                                    drafted_out_cache_loc: torch.Tensor,
                                    next_out_cache_loc: torch.Tensor):
        """
        Optimized input assembly for speculative decoding verification step.
        
        This function assembles the flat input tensor and corresponding cache locations 
        by vectorizing operations on GPU and minimizing CPU-GPU synchronization overhead.
        """
        
        device = req_pool_indices_tensor.device
        num_reqs = len(active_indices_list)

        # -------------------------------------------------------------------------
        # 1. Compute Memory Layout (GPU)
        # -------------------------------------------------------------------------
        # Determine the structure of the output tensor: [Extend | Draft | Next] per request.
        next_lens = torch.ones_like(draft_lens) if is_next_draft else torch.zeros_like(draft_lens)
        
        # Calculate offsets for each request in the flattened output buffer.
        total_per_req_lens = extend_lens + draft_lens + next_lens
        out_ends = torch.cumsum(total_per_req_lens, dim=0)
        out_starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), out_ends[:-1]])
        
        total_output_len = out_ends[-1].item()
        
        # Pre-allocate output buffers on GPU.
        full_input_ids = torch.empty(total_output_len, dtype=drafted_ids.dtype, device=device)
        full_out_cache_loc = torch.empty(total_output_len, dtype=drafted_out_cache_loc.dtype, device=device)

        # -------------------------------------------------------------------------
        # 2. Vectorized Scatter: Draft Tokens (GPU)
        # -------------------------------------------------------------------------
        if drafted_ids.numel() > 0:
            # Map flat drafted_ids to their interleaved positions in full_input_ids.
            # Destination start for draft part: global_req_start + extend_len
            draft_dest_starts = out_starts + extend_lens
            
            # Expand start indices to match the number of tokens in each draft.
            repeated_dest_starts = torch.repeat_interleave(draft_dest_starts, draft_lens)
            
            # Calculate local offsets within each draft sequence (0, 1, 2, ...).
            draft_src_ends = torch.cumsum(draft_lens, dim=0)
            draft_src_starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), draft_src_ends[:-1]])
            repeated_src_starts = torch.repeat_interleave(draft_src_starts, draft_lens)
            
            inner_offsets = torch.arange(drafted_ids.numel(), device=device) - repeated_src_starts
            
            # Scatter data.
            dest_indices = repeated_dest_starts + inner_offsets
            full_input_ids[dest_indices] = drafted_ids
            full_out_cache_loc[dest_indices] = drafted_out_cache_loc

        # -------------------------------------------------------------------------
        # 3. Vectorized Scatter: Next Tokens (GPU)
        # -------------------------------------------------------------------------
        if is_next_draft and next_token_tensor.numel() > 0:
            # Destination position: right after the draft part.
            next_dest_indices = out_starts + extend_lens + draft_lens
            
            # Scatter data.
            full_input_ids[next_dest_indices] = next_token_tensor
            full_out_cache_loc[next_dest_indices] = next_out_cache_loc

        # -------------------------------------------------------------------------
        # 4. Input Marshalling: Extend Tokens (CPU -> GPU)
        # -------------------------------------------------------------------------
        # Process history tokens. CPU intervention is unavoidable here as data resides in 
        # Python objects, but we optimize by pre-allocating a single Numpy buffer.
        total_extend_len = extend_lens.sum().item()
        
        if total_extend_len > 0:
            cpu_extend_ids = np.empty(total_extend_len, dtype=np.int64)
            
            # Use lists for faster iteration in Python
            extend_lens_cpu = extend_lens.tolist()
            current_lens_cpu = current_seq_lens.tolist()
            
            cursor = 0
            for i, pid in enumerate(active_indices_list):
                length = extend_lens_cpu[i]
                if length == 0:
                    continue
                    
                batch_idx = pool_id_map.get(pid)
                if batch_idx is None:
                    cursor += length
                    continue

                req = batch.reqs[batch_idx]
                start = current_lens_cpu[i]
                input_len = len(req.origin_input_ids)
                
                # Efficient slicing to handle boundary between prompt and generated output.
                if start >= input_len:
                    out_start = start - input_len
                    cpu_extend_ids[cursor : cursor + length] = req.output_ids[out_start : out_start + length]
                elif start + length <= input_len:
                    cpu_extend_ids[cursor : cursor + length] = req.origin_input_ids[start : start + length]
                else:
                    part1_len = input_len - start
                    cpu_extend_ids[cursor : cursor + part1_len] = req.origin_input_ids[start:]
                    cpu_extend_ids[cursor + part1_len : cursor + length] = req.output_ids[:length - part1_len]
                
                cursor += length
                
            # Non-blocking transfer to GPU
            gpu_extend_ids = torch.from_numpy(cpu_extend_ids).to(device, dtype=drafted_ids.dtype, non_blocking=True)
            
            # Calculate scatter indices for extend tokens
            extend_dest_starts = out_starts
            repeated_extend_starts = torch.repeat_interleave(extend_dest_starts, extend_lens)
            
            extend_src_ends = torch.cumsum(extend_lens, dim=0)
            extend_src_starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), extend_src_ends[:-1]])
            repeated_src_starts = torch.repeat_interleave(extend_src_starts, extend_lens)
            
            inner_offsets = torch.arange(total_extend_len, device=device) - repeated_src_starts
            extend_dest_indices = repeated_extend_starts + inner_offsets
            
            full_input_ids[extend_dest_indices] = gpu_extend_ids

            # ---------------------------------------------------------------------
            # 5. Cache Lookup: Extend Tokens (GPU)
            # ---------------------------------------------------------------------
            # Retrieve physical cache locations from the block table using advanced indexing.
            req_to_token_matrix = batch.req_to_token_pool.req_to_token
            max_extend = extend_lens.max().item()
            
            # Create a grid to index the 2D block table
            grid_reload = torch.arange(max_extend, device=device, dtype=torch.long)
            mask_reload = grid_reload[None, :] < extend_lens[:, None]
            
            col_indices = current_seq_lens[:, None] + grid_reload[None, :]
            row_indices = req_pool_indices_tensor[:, None]
            
            # Extract valid cache locations and flatten them.
            # Note: masked_select flattens in row-major order, which aligns with our 
            # extend_dest_indices derived from req-by-req iteration.
            reload_vals = req_to_token_matrix[row_indices, col_indices]
            flat_reload_vals = torch.masked_select(reload_vals, mask_reload)
            
            full_out_cache_loc[extend_dest_indices] = flat_reload_vals.to(dtype=drafted_out_cache_loc.dtype)

        return full_input_ids, full_out_cache_loc, total_per_req_lens-1

    # def gen_reload_inputs(self, 
    #                     batch: ScheduleBatch, 
    #                     req_pool_indices: torch.Tensor, 
    #                     current_seq_lens: torch.Tensor, 
    #                     extend_lens: torch.Tensor,
    #                     full_req_pool_indices: torch.Tensor):
        
    #     active_indices = req_pool_indices.tolist()
    #     current_lens = current_seq_lens.tolist()
    #     extend_lens_list = extend_lens.tolist()
    #     full_indices = full_req_pool_indices.tolist()
        
    #     pool_id_map = {pid: i for i, pid in enumerate(full_indices)}
        
    #     flat_tokens = []
        
    #     for i, pid in enumerate(active_indices):
    #         batch_idx = pool_id_map.get(pid)
    #         if batch_idx is None: 
    #             continue
                
    #         req = batch.reqs[batch_idx]
            
    #         start = current_lens[i]
    #         length = extend_lens_list[i]
            
    #         if length <= 0:
    #             continue

    #         input_len = len(req.origin_input_ids)
            
    #         if start >= input_len:
    #             out_start = start - input_len
    #             segment = req.output_ids[out_start : out_start + length]
    #             flat_tokens.extend(segment)
                
    #         elif start + length <= input_len:
    #             segment = req.input_ids[start : start + length]
    #             flat_tokens.extend(segment)
                
    #         else:
    #             part1 = req.input_ids[start:]
    #             remain_len = length - len(part1)
    #             part2 = req.output_ids[:remain_len]
                
    #             flat_tokens.extend(part1)
    #             flat_tokens.extend(part2)

    #     device = req_pool_indices.device
    #     if not flat_tokens:
    #         reload_input_ids = torch.empty(0, dtype=torch.long, device=device)
    #     else:
    #         reload_input_ids = torch.tensor(flat_tokens, dtype=torch.long, device=device)

    #     if extend_lens.numel() == 0 or extend_lens.max() == 0:
    #         reload_out_cache_loc = torch.empty(0, dtype=torch.long, device=device)
    #     else:
    #         req_to_token_matrix = batch.req_to_token_pool.req_to_token
    #         max_len = extend_lens.max().item()
    #         range_tensor = torch.arange(max_len, device=device, dtype=torch.long)
            
    #         mask = range_tensor[None, :] < extend_lens[:, None]
    #         col_indices = current_seq_lens[:, None] + range_tensor[None, :]
    #         row_indices = req_pool_indices[:, None]
            
    #         reload_out_cache_loc = req_to_token_matrix[row_indices, col_indices][mask]

    #     return reload_input_ids, reload_out_cache_loc

    def printk(self) -> str:
        if not self._snapshots:
            return "StateManager(empty)"
        infos = ", ".join(
            snapshot.printk() for snapshot in self._snapshots.values()
        )
        return f"StateManager({infos})"


class HiddenStatesPool:
    __slots__ = ('_data', 'max_num_tokens', 'hidden_size', 'dtype', 'device', 'model_name', 'base_model_name')

    def __init__(
        self,
        model_name: str,
        base_model_name: str,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        
        self._data = torch.empty(
            (max_num_tokens, hidden_size),
            dtype=dtype,
            device=device,
        )
    
    @classmethod
    def create_from_draft_worker(cls, draft_worker):
        if draft_worker.speculative_algorithm.is_eagle():
            runner = draft_worker.model_runner
            hidden_size = getattr(runner.model_config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(
                    getattr(runner.model_config, "hf_config", None), "hidden_size", 0
                )
            hidden_size = hidden_size or 0
            # logger.info(f"hidden_size: {hidden_size}, model_name: {worker.model_name}")

            dtype = getattr(runner, "dtype", torch.bfloat16)
            draft_model_name = draft_worker.model_name
            base_model_name = draft_worker.base_model_name
            max_num_tokens = runner.max_total_num_tokens
            device = runner.device
            return cls(draft_model_name, base_model_name, max_num_tokens, hidden_size, dtype, device)
        else:
            return None

    @property
    def shape(self):
        return self._data.shape

    def write_batch(
        self, 
        hidden_states: torch.Tensor, 
        out_loc: torch.Tensor
    ) -> None:
        self._data[out_loc] = hidden_states

    def read_batch(
        self, 
        out_loc: torch.Tensor
    ) -> torch.Tensor:
        return self._data[out_loc]

    def get_pool_shape(self) -> Tuple[int, int]:
        return self._data.shape

    def reset_device(self, new_device: str) -> None:
        if self.device != new_device:
            self.device = new_device
            self._data = self._data.to(new_device)


class ExtendHistory:
    def __init__(self, current_seq_lens: torch.Tensor, extend_lens: torch.Tensor):
        self.current_seq_lens = current_seq_lens
        self.extend_lens = extend_lens
    
    def gen_reload_inputs(self, batch: ScheduleBatch):
        pass



