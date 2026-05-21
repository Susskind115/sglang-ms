# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A tensor parallel worker."""

import logging
import threading
from typing import Optional, Tuple, Union, List
from contextlib import contextmanager
import time
import os
import torch
from huggingface_hub import snapshot_download

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, global_server_args_dict
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunner
# from sglang_lib.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed, empty_context, fast_topk, get_available_gpu_memory, is_cuda
from sglang.srt.distributed import GroupCoordinator, patch_tensor_parallel_group
from sglang.srt.layers.dp_attention import disable_dp_size
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.model_hub.chain_utils import ModelHistory

# from sglang_lib.spec_info import SpeculativeAlgorithm


from sglang.srt.managers.schedule_batch import (
    ScheduleBatch,
    get_last_loc,
    global_server_args_dict,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
# from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
#     EAGLEDraftCudaGraphRunner,
# )
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
    assign_draft_cache_locs,
    generate_token_bitmask,
    select_top_k_tokens,
)
from sglang.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.utils import empty_context, fast_topk, get_available_gpu_memory, is_cuda

from sglang.srt.model_hub.utils.nvtx_marker import nvtx_profile

# from sglang.srt.model_hub.test.scan_arrays import find_diff_indices

logger = logging.getLogger(__name__)

@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    with disable_dp_size(), patch_tensor_parallel_group(tp_group):
        yield


class TpModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        spec_flag: bool = False,
        model_config: ModelConfig = None,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocator] = None,
        defer_memory_init: bool = False,
        enabled_skip_extend: bool = False,
    ):
        # Parse args
        # self.server_args = server_args
        self.tp_size = server_args.tp_size
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.gpu_id = gpu_id
        self.model_name = server_args.model_path
        self.base_model_name = server_args.tokenizer_path
        self.spec_flag = spec_flag
        self.enabled_skip_extend = enabled_skip_extend
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
        self.first_dst_k_cache = None

        # Init model and tokenizer
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=(
                server_args.model_path
            ),
            is_draft_model=is_draft_worker,
        ) if model_config is None else model_config

        # if is_draft_worker:
        #     server_args.disable_cuda_graph = True # 感觉可能是这里导致的。无情封死cuda graph
        
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.page_size = server_args.page_size
        self.enable_nan_detection = server_args.enable_nan_detection

        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            pp_rank=pp_rank,
            pp_size=server_args.pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
            spec_flag=spec_flag,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            defer_memory_init=defer_memory_init,
        )

        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # self.post_init_model_runner(server_args, defer_memory_init)

        # A reference make this class has the same member as TpModelWorkerClient
        self.worker = self
        self.device = self.model_runner.device

    def set_speculative_args(self,num_steps: int,  topk: int, num_draft_tokens: int):
        self.speculative_num_steps = num_steps
        self.topk = topk
        self.speculative_num_draft_tokens = num_draft_tokens
        if self.speculative_num_steps is not None:
            self.padded_static_len = self.speculative_num_steps + 1
    
    def update_speculative_args(self, num_steps: int, topk: int, num_draft_tokens: int):
        if num_steps == self.speculative_num_steps and topk == self.topk and num_draft_tokens == self.speculative_num_draft_tokens:
            return
        self.set_speculative_args(num_steps, topk, num_draft_tokens)
        self.model_runner.update_speculative_args(num_steps, topk, num_draft_tokens)
        if self.draft_attn_backend is not None:
            self.draft_attn_backend.set_speculative_args(num_steps, topk, num_draft_tokens)
        if self.draft_cuda_graph_runner is not None:
            self.draft_cuda_graph_runner.set_speculative_args(num_steps, topk, num_draft_tokens)

    def post_init_model_runner(self, defer_memory_init: bool, 
                               visible_gpu_memory: float,
                               req_to_token_pool: Optional[ReqToTokenPool] = None, 
                               token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocator] = None,
                               draft_runner_cache_size: Optional[int] = None,
                               max_num_reqs: Optional[int] = None):
        server_args = self.model_runner.server_args
        # print("self.model_config: ", vars(self.model_config))
        if self.model_runner.is_draft_worker:
            self.model_runner.server_args.draft_runner_cache_size = draft_runner_cache_size
            self.model_runner.server_args.max_num_reqs = max_num_reqs
        if defer_memory_init: 
            self.model_runner.init_memory_and_backend(self.model_runner.pending_memory_init["min_per_gpu_memory"], 
                                                      visible_gpu_memory=visible_gpu_memory,
                                                      req_to_token_pool=req_to_token_pool, 
                                                      token_to_kv_pool_allocator=token_to_kv_pool_allocator, 
                                                      force_disable_cuda_graph=self.model_runner.is_draft_worker)
        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.token_to_kv_pool_allocator = self.model_runner.token_to_kv_pool_allocator

        # Init nccl groups
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()
        
        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
                // (server_args.dp_size if server_args.enable_dp_attention else 1)
            ),
            self.model_runner.req_to_token_pool.size,
        )
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + self.tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]

        self.draft_attn_backend = None
        self.draft_cuda_graph_runner = None

        if self.model_runner.is_draft_worker:
            self.init_draft_attributes()
            self.init_draft_backend_and_cuda_graphs()
        else:
            self.set_speculative_args(server_args.speculative_num_steps, server_args.speculative_eagle_topk, server_args.speculative_num_draft_tokens)
        set_random_seed(self.random_seed)

        
    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            global_server_args_dict,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_tp_group(self):
        return self.model_runner.tp_group

    def get_attention_tp_group(self):
        return self.model_runner.attention_tp_group

    def get_attention_tp_cpu_group(self):
        return getattr(self.model_runner.attention_tp_group, "cpu_group", None)

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    @nvtx_profile
    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

        # if self.model_name == "/data/huggingface/Llama-2-7b-chat-hf":
        #     logger.info(f"forward : input_ids:{forward_batch.input_ids}, out_cache_loc: {forward_batch.out_cache_loc}, positions: {forward_batch.positions}")

        # if forward_batch.forward_mode == ForwardMode.TARGET_VERIFY and torch.distributed.get_rank() == 0:
        #     # logger.info(f"forward_batch_generation verify, forward_batch.input_ids: {forward_batch.input_ids.shape}, forward_batch.out_cache_loc: {len(forward_batch.out_cache_loc)}")
        #     logger.info(f"batch vars: {vars(forward_batch)}")

        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self.pp_group.recv_tensor_dict(
                    all_gather_group=self.get_attention_tp_group()
                )
            )

        if self.pp_group.is_last_rank:
            logits_output, can_run_cuda_graph = self.model_runner.forward(
                forward_batch, pp_proxy_tensors=pp_proxy_tensors
            )
            if launch_done is not None:
                launch_done.set()

            if skip_sample:
                next_token_ids = None
            else:
                next_token_ids = self.model_runner.sample(
                    logits_output, model_worker_batch
                )
            return logits_output, next_token_ids, can_run_cuda_graph
        else:
            pp_proxy_tensors, can_run_cuda_graph = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return pp_proxy_tensors.tensors, None, can_run_cuda_graph

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output, _ = self.model_runner.forward(forward_batch)
        embeddings = logits_output.embeddings
        return embeddings

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path, recv_req.load_format
        )
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.model_runner.update_weights_from_distributed(
            recv_req.name, recv_req.dtype, recv_req.shape
        )
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_rank]
            ),
            load_format=recv_req.load_format,
        )
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.model_runner.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter
    
    # enable draft attr

    def init_draft_attributes(self):
        # Parse arguments
        # self.server_args = server_args
        server_args = self.model_runner.server_args
        # self.topk = server_args.speculative_eagle_topk
        # self.speculative_num_steps = server_args.speculative_num_steps
        self.set_speculative_args(server_args.speculative_num_steps, server_args.speculative_eagle_topk, server_args.speculative_num_draft_tokens)
        self.device = server_args.device
        # self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self.enabled_skip_extend = self.enabled_skip_extend and self.topk == 1 and (self.speculative_num_steps+1 == self.speculative_num_draft_tokens)

    def init_draft_backend_and_cuda_graphs(self):
        # Init draft attention backend and cuda graph runner
        server_args = self.model_runner.server_args
        attention_backend = server_args.attention_backend
        disable_cuda_graph = server_args.disable_cuda_graph
        with self.draft_tp_context(self.model_runner.tp_group):
            self.init_draft_attention_backend(attention_backend)
            self.init_draft_cuda_graphs(disable_cuda_graph)
    
    def init_draft_attention_backend(self, attention_backend):
        # Create multi-step attn backends and cuda graph runners
        if attention_backend == "flashinfer":
            if not global_server_args_dict["use_mla_backend"]:
                from sglang.srt.layers.attention.flashinfer_backend import (
                    FlashInferMultiStepDraftBackend,
                )

                self.draft_attn_backend = FlashInferMultiStepDraftBackend(
                    self.model_runner,
                    self.topk,
                    self.speculative_num_steps,
                )
            else:
                from sglang.srt.layers.attention.flashinfer_mla_backend import (
                    FlashInferMLAMultiStepDraftBackend,
                )

                self.draft_attn_backend = FlashInferMLAMultiStepDraftBackend(
                    self.model_runner,
                    self.topk,
                    self.speculative_num_steps,
                )
            self.draft_extend_attn_backend = None
            self.padded_static_len = self.speculative_num_steps + 1
            self.has_prefill_wrapper_verify = True
        elif attention_backend == "triton":
            from sglang.srt.layers.attention.triton_backend import (
                TritonMultiStepDraftBackend,
            )

            self.draft_attn_backend = TritonMultiStepDraftBackend(
                self.model_runner,
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend = None
            self.padded_static_len = self.speculative_num_steps + 1
            self.has_prefill_wrapper_verify = False
        elif attention_backend == "fa3":
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionMultiStepBackend,
            )

            self.draft_attn_backend = FlashAttentionMultiStepBackend(
                self.model_runner,
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend = None
            self.padded_static_len = self.speculative_num_steps + 1
            self.has_prefill_wrapper_verify = False
        elif attention_backend == "flashmla":
            from sglang.srt.layers.attention.flashmla_backend import (
                FlashMLAMultiStepDraftBackend,
            )

            self.draft_attn_backend = FlashMLAMultiStepDraftBackend(
                self.model_runner,
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend = None
            self.padded_static_len = self.speculative_num_steps + 1
            self.has_prefill_wrapper_verify = False
        else:
            raise ValueError(
                f"EAGLE is not supported in attention backend {attention_backend}"
            )

        self.model_runner.draft_attn_backend = self.draft_attn_backend
    
    
    def set_eagle_embed_and_head(self, target_worker: "TpModelWorker"):
        assert hasattr(target_worker, "model_runner")
        self.model_runner.model_config.context_length = target_worker.model_runner.model_config.context_len

        if self.speculative_algorithm.is_eagle3():
            embed, head = target_worker.model_runner.model.get_embed_and_head()
            # EAGLE3 models don't share lm_head
            self.model_runner.model.set_embed(embed)

            # grab hot token ids
            self.hot_token_id = self.model_runner.model.get_hot_token_id().to(
                embed.device
            )
        elif self.speculative_algorithm.is_eagle():
            embed, head = target_worker.model_runner.model.get_embed_and_head()
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.model_runner.model.set_embed_and_head(embed, head)
        else:
            self.hot_token_id = self.model_runner.model.get_hot_token_id().to(
                embed.device
            )
    
    def init_draft_cuda_graphs(self, disable_cuda_graph):
        """Capture cuda graphs."""
        self.draft_cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if disable_cuda_graph:
            return

        # Capture draft
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        from sglang.srt.speculative.eagle_draft_cuda_graph_runner import EAGLEDraftCudaGraphRunner
        self.draft_cuda_graph_runner = EAGLEDraftCudaGraphRunner(self)
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. avail mem={after_mem:.2f} GB. mem usage={(before_mem - after_mem):.2f} GB."
        )

        # Capture extend
        if self.draft_extend_attn_backend:
            raise NotImplementedError()
    
    ##################### draft forward zone ######################

    @nvtx_profile
    def draft(self, batch: ScheduleBatch, speculative_num_draft_tokens: int=0):
        # Parse args
        num_seqs = batch.batch_size()
        assert batch.batch_size() == batch.spec_info.verified_id.shape[0]
        spec_info = batch.spec_info
        if speculative_num_draft_tokens == 0:
            speculative_num_draft_tokens = self.speculative_num_draft_tokens
        # TODO 记得修改这个变量名 skip_forward_draft_extend_after_decode_flag
        # self.skip_forward_draft_extend_after_decode_flag = (self.topk == 1 and self.speculative_num_steps+1 == self.model_runner.server_args.speculative_num_draft_tokens)
        # print(f"spec_info draft, {spec_info}")
        # if torch.distributed.get_rank() == 0:
        #     # logger.info(f"req_to_token: {self.model_runner.req_to_token_pool.req_to_token}, {self.model_runner.req_to_token_pool.free_slots}, {self.model_runner.req_to_token_pool.req_to_token.shape}")
        #     logger.info(f": {self.model_runner.req_to_token_pool}")

        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )
    
        # Allocate cache locations
        if self.page_size == 1:
            out_cache_loc, token_to_kv_pool_state_backup = batch.alloc_token_slots(
                num_seqs * self.topk * (self.speculative_num_steps), backup_state=True
                # num_seqs * self.topk * (self.speculative_num_steps+1), backup_state=True
            )
        else:
            if self.topk == 1:
                prefix_lens = batch.seq_lens
                seq_lens = prefix_lens + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    tok-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                #  "x" means speculative draft tokens
                #  "." means padded tokens

                # TODO: fuse these ops
                prefix_lens = batch.seq_lens
                last_page_lens = prefix_lens % self.page_size
                num_new_pages = (
                    last_page_lens + self.speculative_num_steps + self.page_size - 1
                ) // self.page_size
                seq_lens = (
                    prefix_lens // self.page_size * self.page_size
                    + num_new_pages * (self.page_size * self.topk)
                )
                extend_num_tokens = torch.sum(seq_lens - prefix_lens).item()
                raise NotImplementedError(
                    "page_size > 1 and top_k > 1 are not supported."
                )
                # TODO: Support page_size > 1 and top_k > 1
                # 1. Duplicate the KV cache in the last partial page for all top-k segments
                # 2. Modify generate_draft_decode_kv_indices accordingly

            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            out_cache_loc, token_to_kv_pool_state_backup = (
                batch.alloc_paged_token_slots_extend(
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
        )
        batch.out_cache_loc = out_cache_loc
        # if torch.distributed.get_rank() == 0:
        #     logger.info(f"draft start out_cache_loc: {batch.out_cache_loc}")
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

        # Get forward batch
        # spec_info.capture_hidden_mode = self.check_capture_hidden_mode(self.speculative_algorithm)
        # spec_info.capture_hidden_mode = self.model_runner.server_args.capture_hidden_mode

        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.model_runner
        )
        # print("forward_batch.spec_info", forward_batch.spec_info)
        # # TODO 强行试一下
        # score_list, token_list, parents_list = self.draft_cuda_graph_runner.replay(
        #         forward_batch
        #     )
        can_draft_cuda_graph = self.draft_cuda_graph_runner and self.draft_cuda_graph_runner.can_run(
            forward_batch
        )
        if can_draft_cuda_graph:
            score_list, token_list, parents_list = self.draft_cuda_graph_runner.replay(
                forward_batch
            )
        else:
            # Initialize attention backend
            self.draft_attn_backend.init_forward_metadata(forward_batch)
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.model_runner
            )
            # Run forward steps
            score_list, token_list, parents_list = self.draft_forward(forward_batch)

        # if torch.distributed.get_rank() == 0:
        #     logger.info(f"token_to_kv_pool_state_backup: {token_to_kv_pool_state_backup}")
        #     logger.info(f"out_cache_loc: {out_cache_loc}")
        
        if self.enabled_skip_extend:
            # 准备跳过forward_draft_extend_after_decode
            # logger.info(f"enabled_skip_extend: {self.enabled_skip_extend}")
            pass
        else:
            self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)
            if spec_info.filtered_out_cache_loc is not None:
                self.token_to_kv_pool_allocator.free(spec_info.filtered_out_cache_loc)

        # score_list: {score_list},\
        # token_list: {token_list},\
        # parents_list: {parents_list},\
        # logger.info(f"draft output, score_list: {score_list},\
        #     token_list: {token_list},\
        #     parents_list: {parents_list},\
        #     batch.seq_lens: {batch.seq_lens},")

        # logger.info(f"draft output, verified_id: {spec_info.verified_id},\
        #     batch.seq_lens: {batch.seq_lens},\
        #     batch.seq_lens_sum: {batch.seq_lens_sum},\
        #     self.topk: {self.topk},\
        #     self.speculative_num_steps: {self.speculative_num_steps},\
        #     self.speculative_num_draft_tokens: {self.speculative_num_draft_tokens},\
        #     self.enabled_skip_extend: {self.enabled_skip_extend}")

        ret = EagleVerifyInput.create(
            spec_info.verified_id,
            score_list,
            token_list,
            parents_list,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            speculative_num_draft_tokens,
            self.enabled_skip_extend,
        )
        # print("draft ret", score_list, token_list, parents_list)
        # logger.info(f"verify spec_info.draft_token: {ret.draft_token}, token_list:{token_list}")
        return ret

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        
        # speculative_num_steps+1
        # out_cache_loc_backup = out_cache_loc.view(forward_batch.batch_size, -1).clone()
        # out_cache_loc =  out_cache_loc_backup.clone()[:, :-1]

        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        use_compiled_topk = not self.speculative_algorithm.is_eagle3()
        # print("hidden_states draft_forward", hidden_states)
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i,
                topk_p,
                topk_index,
                hidden_states,
                scores,
                self.topk,
                use_compiled=use_compiled_topk,
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            out_cache_loc = out_cache_loc.view(forward_batch.batch_size, -1)
            # logger.info(f"out_cache_loc: {out_cache_loc.shape}, {forward_batch.batch_size}")
            forward_batch.out_cache_loc = out_cache_loc[
                :, self.topk * i : self.topk * (i + 1)
            ].flatten()
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.model_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            self._detect_nan_if_needed(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states
        

        return score_list, token_list, parents_list
    
    @nvtx_profile
    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        prefill_logits_num: int = 0,
        # hidden_states: torch.Tensor,
        # next_token_ids: List[int],

    ):
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # logger.info(f"going from TPWorker.forward_draft_extend")
        
        batch.spec_info.prepare_for_extend(batch)
        # batch.spec_info.hidden_states = hidden_states_pool.read_batch(batch.out_cache_loc)
        # batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        # batch.spec_info.capture_hidden_mode = self.check_capture_hidden_mode(self.speculative_algorithm)
        model_worker_batch = batch.get_model_worker_batch()

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.model_runner
        )
        
        forward_batch.return_logprob = False
        if prefill_logits_num > 0:
            forward_batch.return_logprob = True
            seq_lens = forward_batch.extend_seq_lens_cpu
            # logger.info(f"seq_lens: {seq_lens}")
            batch_size = len(seq_lens)
            forward_batch.top_logprobs_nums = [0] * batch_size
            forward_batch.token_ids_logprobs = [None] * batch_size
            start_lens = [max(0, L - prefill_logits_num) for L in seq_lens]
            # start_lens = [max(0, L - prefill_logits_num - 1) for L in seq_lens]
            # logger.info(f"start_lens: {start_lens}")
            forward_batch.extend_logprob_start_lens_cpu = start_lens
            
        
        logits_output, _ = self.model_runner.forward(forward_batch)
        
        # logger.info(f"logits_output.input_token_logprobs: {logits_output.input_token_logprobs.shape}")
        self._detect_nan_if_needed(logits_output)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        # assert forward_batch.spec_info is batch.spec_info
        # self.capture_for_decode(logits_output, forward_batch.spec_info)
        return logits_output, None, model_worker_batch.bid

    @nvtx_profile
    def forward_target_extend_after_decode(self, batch: ScheduleBatch):
        # logger.info("forward_target_extend_after_decode")
        if not batch.spec_info.enabled_skip_extend:
            # logger.info("can't skip extend")
            # logger.info(f"forward_target, batch: {vars(batch)}")
            # logger.info(f"forward_target_extend, {batch.seq_lens} ")
            seq_lens_backup = batch.seq_lens.clone()
            seq_lens_sum_backup = batch.seq_lens_sum
            req_pool_indices_backup = batch.req_pool_indices.clone()
            accept_length_backup = batch.spec_info.accept_length.clone()
            return_logprob_backup = batch.return_logprob

            if len(req_pool_indices_backup) != len(batch.spec_info.req_pool_indices_for_draft_extend):
                restore_temperatures = batch.sampling_info.temperatures
                restore_top_ps = batch.sampling_info.top_ps
                restore_top_ks = batch.sampling_info.top_ks
                restore_min_ps = batch.sampling_info.min_ps

            # Prepare metadata
            batch.forward_mode = ForwardMode.DRAFT_EXTEND
            new_verified_id, token_to_kv_pool_state_backup = batch.spec_info.prepare_target_extend_after_decode(
                batch,
                self.speculative_num_steps,
            )

            # batch.spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
            # batch.spec_info.capture_hidden_mode = self.model_router.get_model_capture_hidden_mode_from_table(self.model_name, "full")
            batch.return_logprob = False
            model_worker_batch = batch.get_model_worker_batch()

            # forward_batch = ForwardBatch.init_new(
            #     model_worker_batch, self.model_runner
            #     )
            # # Run
            # logits_output, _ = self.model_runner.forward(forward_batch)
            logits_output, next_token_ids, model_worker_batch_bid = self.forward_batch_generation(model_worker_batch)

            self._detect_nan_if_needed(logits_output)
            # self.capture_for_decode(logits_output, forward_batch.spec_info)
            
            # self.state_manager.create_submit_input(self.model_name, new_verified_id, batch.req_pool_indices, self.speculative_num_steps, self.topk)

            # Restore backup.
            # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
            # batch.forward_mode = ForwardMode.DECODE
            # batch.seq_lens = seq_lens_backup
            # batch.req_pool_indices = req_pool_indices_backup
            # batch.spec_info.accept_length = accept_length_backup
            # batch.return_logprob = return_logprob_backup
            # batch.spec_info.hidden_states = logits_output.hidden_states

            batch.forward_mode = ForwardMode.DECODE
            batch.seq_lens = seq_lens_backup
            batch.seq_lens_sum = seq_lens_sum_backup
            batch.req_pool_indices = req_pool_indices_backup
            batch.spec_info.accept_length = accept_length_backup
            batch.return_logprob = return_logprob_backup
            batch.spec_info.hidden_states = logits_output.hidden_states
            if len(req_pool_indices_backup) != len(batch.spec_info.req_pool_indices_for_draft_extend):
                batch.sampling_info.temperatures = restore_temperatures
                batch.sampling_info.top_ps = restore_top_ps
                batch.sampling_info.top_ks = restore_top_ks
                batch.sampling_info.min_ps = restore_min_ps

            batch.spec_info.verified_id = batch.spec_info.append_by_accept_lengths(batch.spec_info.verified_id, next_token_ids, batch.spec_info.accept_length)
            batch.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)
            # logger.info(f"after forward_target_extend, {batch.seq_lens} ,accept_length:{batch.spec_info.accept_length}")

            # logger.info(f"after forward_target, batch: {vars(batch)}")

            return new_verified_id, logits_output, next_token_ids, model_worker_batch_bid

        else:
            logger.info(f"enabled_skip_extend! {self.model_name}")
            pass

    @nvtx_profile
    def forward_target_extend_after_decode_src(self, batch: ScheduleBatch, reload_model_history: Optional[ModelHistory] = None):
        # logger.info("\n")
        if not batch.spec_info.enabled_skip_extend:
            # logger.info("can't skip extend")
            # logger.info(f"forward_target_extend, {batch.seq_lens} ,accept_length:{batch.spec_info.accept_length}")
            seq_lens_backup = batch.seq_lens.clone()
            seq_lens_sum_backup = batch.seq_lens_sum
            req_pool_indices_backup = batch.req_pool_indices.clone()
            reqs_backup = list(batch.reqs)
            accept_length_backup = batch.spec_info.accept_length.clone()
            out_cache_loc_backup = batch.out_cache_loc.clone()
            return_logprob_backup = batch.return_logprob
            # Prepare metadata
            batch.forward_mode = ForwardMode.DRAFT_EXTEND
            if reload_model_history is not None:
                new_verified_id = batch.spec_info.prepare_target_extend_after_decode_history(
                    batch,
                    # self.speculative_num_steps,
                    reload_model_history
                )
                # torch.distributed.barrier()
                
                # logger.info(f"history out self.positions {batch.spec_info.positions}, batch.seq_lens {batch.seq_lens},")
            else:
                new_verified_id = batch.spec_info.prepare_target_extend_after_decode_src(
                    batch,
                    self.speculative_num_steps
                )

            # batch.spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
            # batch.spec_info.capture_hidden_mode = self.model_router.get_model_capture_hidden_mode_from_table(self.model_name, "full")
            batch.return_logprob = False
            model_worker_batch = batch.get_model_worker_batch()

            # forward_batch = ForwardBatch.init_new(
            #     model_worker_batch, self.model_runner
            #     )
            # # Run
            # logits_output, _ = self.model_runner.forward(forward_batch)
            logits_output, next_token_ids, model_worker_batch_bid = self.forward_batch_generation(model_worker_batch, skip_sample=True)

            self._detect_nan_if_needed(logits_output)
            # self.capture_for_decode(logits_output, forward_batch.spec_info)
            
            # self.state_manager.create_submit_input(self.model_name, new_verified_id, batch.req_pool_indices, self.speculative_num_steps, self.topk)

            # Restore backup.
            # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
            # batch.forward_mode = ForwardMode.DECODE
            # batch.seq_lens = seq_lens_backup
            # batch.req_pool_indices = req_pool_indices_backup
            # batch.spec_info.accept_length = accept_length_backup
            # batch.return_logprob = return_logprob_backup
            # batch.spec_info.hidden_states = logits_output.hidden_states

            batch.forward_mode = ForwardMode.DECODE
            batch.seq_lens = seq_lens_backup
            batch.seq_lens_sum = seq_lens_sum_backup
            batch.req_pool_indices = req_pool_indices_backup
            batch.reqs = reqs_backup
            batch.spec_info.accept_length = accept_length_backup
            batch.return_logprob = return_logprob_backup
            batch.spec_info.hidden_states = logits_output.hidden_states
            batch.spec_info.accept_length_cpu = accept_length_backup.tolist()
            batch.out_cache_loc = out_cache_loc_backup
            # logger.info(f"after forward_target_extend_after_decode_src, {batch.seq_lens} ,accept_length:{batch.spec_info.accept_length}")

            # batch.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)

            return new_verified_id, logits_output, next_token_ids, model_worker_batch_bid

        else:
            logger.info(f"enabled_skip_extend! {self.model_name}")
            pass

    @nvtx_profile
    def forward_draft_extend_after_decode(self, batch: ScheduleBatch, reload_model_history: Optional[ModelHistory] = None):
        # if self.skip_forward_draft_extend_after_decode_flag:
        # if not self.enabled_skip_extend:
        # default
        # Backup fields that will be modified in-place
        # logger.info(f"enabled_skip_extend: {self.enabled_skip_extend}")
        # logger.info("forward_draft_extend_after_decode")
        seq_lens_backup = batch.seq_lens.clone()
        seq_lens_sum_backup = batch.seq_lens_sum
        req_pool_indices_backup = batch.req_pool_indices
        reqs_backup = list(batch.reqs)
        accept_length_backup = batch.spec_info.accept_length.clone()
        out_cache_loc_backup = batch.out_cache_loc.clone()
        return_logprob_backup = batch.return_logprob
        # hidden_states_backup = batch.spec_info.hidden_states.clone()


        # Prepare metadata
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        if reload_model_history is not None:
            batch.spec_info.prepare_extend_after_decode_history(
                batch,
                reload_model_history
            )
            # torch.distributed.barrier()
        else:
            batch.spec_info.prepare_extend_after_decode(
                batch,
                self.speculative_num_steps,
            )
        # logger.info(f"batch.spec_info.hidden_states: {batch.spec_info.hidden_states.shape}, {reload_model_history is None}")

        # batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        # batch.spec_info.capture_hidden_mode = self.model_router.get_model_capture_hidden_mode_from_table(self.model_name, "last")
        # batch.spec_info.capture_hidden_mode = self.check_capture_hidden_mode(self.speculative_algorithm)
        # batch.spec_info.capture_hidden_mode = capture_hidden_mode
        batch.return_logprob = False
        model_worker_batch = batch.get_model_worker_batch()

        # logits_output, _, _ = self.forward_batch_generation(
        #     model_worker_batch, skip_sample=True
        # )

        # self._detect_nan_if_needed(logits_output)
        # self.capture_for_decode(logits_output, model_worker_batch.spec_info)
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.model_runner
            )

        # Run
        logits_output, _ = self.model_runner.forward(forward_batch)

        self._detect_nan_if_needed(logits_output)
        self.capture_for_decode(logits_output, forward_batch.spec_info)

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = ForwardMode.DECODE
        batch.seq_lens = seq_lens_backup
        batch.seq_lens_sum = seq_lens_sum_backup
        batch.req_pool_indices = req_pool_indices_backup
        batch.reqs = reqs_backup
        batch.spec_info.accept_length = accept_length_backup
        batch.return_logprob = return_logprob_backup
        batch.spec_info.accept_length_cpu = accept_length_backup.tolist()
        batch.out_cache_loc = out_cache_loc_backup
        # batch.spec_info.hidden_states = hidden_states_backup

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        draft_input.hidden_states = logits_output.hidden_states

    def _detect_nan_if_needed(self, logits_output: LogitsProcessorOutput):
        if self.enable_nan_detection:
            logits = logits_output.next_token_logits
            if torch.any(torch.isnan(logits)):
                logger.error("Detected errors during sampling! NaN in the logits.")
                raise ValueError("Detected errors during sampling! NaN in the logits.")
    
    # def check_capture_hidden_mode(self, speculative_algorithm: SpeculativeAlgorithm):
    #     capture_hidden_mode = CaptureHiddenMode.LAST
    #     if speculative_algorithm.is_eagle():
    #         capture_hidden_mode = CaptureHiddenMode.LAST
    #     elif speculative_algorithm.is_not_eagle():
    #         capture_hidden_mode = CaptureHiddenMode.LAST
    #         # capture_hidden_mode = CaptureHiddenMode.NULL
    #     else:
    #         raise ValueError(f"Invalid speculative algorithm: {speculative_algorithm}")
    #     return capture_hidden_mode

    # def check_binary_capture_hidden_mode(self, draft_speculative_algorithm: SpeculativeAlgorithm, target_speculative_algorithm: SpeculativeAlgorithm):
    #     capture_hidden_mode = CaptureHiddenMode.NULL
    #     if draft_speculative_algorithm.is_eagle3() or target_speculative_algorithm.is_eagle3():
    #         capture_hidden_mode = CaptureHiddenMode.FULL
    #     elif draft_speculative_algorithm.is_eagle() or target_speculative_algorithm.is_eagle():
    #         capture_hidden_mode = CaptureHiddenMode.LAST
    #     else:
    #         capture_hidden_mode = CaptureHiddenMode.NULL
    #     return capture_hidden_mode


##################### target forward zone ######################
        
    @nvtx_profile
    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        # logger.info(f"model name: {self.model_name}")
        spec_info.prepare_for_verify(batch, self.page_size)
        # if torch.distributed.get_rank() == 0:
        #     logger.info(f"verify start out_cache_loc: {batch.out_cache_loc}")
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec_info
        model_worker_batch = batch.get_model_worker_batch()
        target_req_bs = int(model_worker_batch.req_pool_indices.numel())
        target_token_bs = int(model_worker_batch.input_ids.shape[0])

        # forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        
        # if torch.distributed.get_rank() == 0:
            
        #     logger.info(f"verify forward_batch.token_to_kv_pool: {forward_batch.token_to_kv_pool}")
        #     logger.info(f"verify_model self.model_runner.token_to_kv_pool_allocator.get_kvcache(): {self.model_runner.token_to_kv_pool_allocator.get_kvcache()}")

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrive_next_token.shape
            ).cpu()
        # Forward
        logits_output, _, can_run_cuda_graph = (
            self.forward_batch_generation(
                model_worker_batch, skip_sample=True
            )
        )
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # Overlap the CPU operations for bitmask generation with the forward pass.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrive_next_token.device)
                # otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        self._detect_nan_if_needed(logits_output)
        # logger.info(f"verify, model_worker_batch.input_ids: {batch.seq_lens},")
        spec_info.hidden_states = logits_output.hidden_states
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        # logger.info(f"before hidden_states: {logits_output.hidden_states}")
        if spec_info.capture_hidden_mode != CaptureHiddenMode.NULL:
            logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]
        
        # logger.info(f" after , model_worker_batch.input_ids: {batch.seq_lens}, accept_length: {res.accept_length}")
        # logger.info(f"inputs: {batch.input_ids}, draft_id: {res.drafted_id}, verified_id: {res.verified_id}")
        # Prepare the batch for the next draft forwards.
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = res.draft_input
        emitted_tokens = int(sum(res.accept_length_per_req_cpu) + target_req_bs)
        batch.runtime_step_telemetry = {
            "target_step_kind": "target_verify",
            "target_step_req_bs": target_req_bs,
            "target_step_token_bs": target_token_bs,
            "accepted_tokens_per_target_step": emitted_tokens,
            "verify_tokens_per_emitted_token": (
                float(target_token_bs) / float(emitted_tokens)
                if emitted_tokens > 0
                else 0.0
            ),
        }
        # batch.spec_info.verified_id = res.verified_id
        # logger.info(f"after verify, draft_input: {res.draft_input.accept_length}")

        if batch.return_logprob:
            self.add_logprob_values(batch, res, logits_output)

        return logits_output, res, model_worker_batch.bid, can_run_cuda_graph


    def add_logprob_values(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
    ):
        # Extract args
        logits_output = res.logits_output
        top_logprobs_nums = batch.top_logprobs_nums
        token_ids_logprobs = batch.token_ids_logprobs
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits, dim=-1
        )
        batch_next_token_ids = res.verified_id
        num_tokens_per_req = [accept + 1 for accept in res.accept_length_per_req_cpu]

        # We should repeat top_logprobs_nums to match num_tokens_per_req.
        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)

        # Extract logprobs
        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums_repeat_interleaved)

        if any(x is not None for x in token_ids_logprobs):
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(logprobs, token_ids_logprobs_repeat_interleaved)

        logits_output.next_token_logprobs = logprobs[
            torch.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
            batch_next_token_ids,
        ]

        # Add output logprobs to the request
        pt = 0
        next_token_logprobs = logits_output.next_token_logprobs.tolist()
        verified_ids = batch_next_token_ids.tolist()
        for req, num_tokens in zip(batch.reqs, num_tokens_per_req):
            for _ in range(num_tokens):
                if req.return_logprob:
                    req.output_token_logprobs_val.append(next_token_logprobs[pt])
                    req.output_token_logprobs_idx.append(verified_ids[pt])
                    if req.top_logprobs_num > 0:
                        req.output_top_logprobs_val.append(
                            res.logits_output.next_token_top_logprobs_val[pt]
                        )
                        req.output_top_logprobs_idx.append(
                            res.logits_output.next_token_top_logprobs_idx[pt]
                        )
                pt += 1
    
    @nvtx_profile
    def forward_target_extend(
        self, batch: ScheduleBatch, 
        prefill_logits_num: int = 0,
    ) -> Tuple[LogitsProcessorOutput, List[int], int]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
            bid: The model batch ID. Used for overlap schedule.
        """
        # # Forward with the target model and get hidden states.
        # # We need the full hidden states to prefill the KV cache of the draft model.
        # model_worker_batch = batch.get_model_worker_batch()
        # # model_worker_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        # # TODO: one static config
        # if batch.spec_info.capture_hidden_mode.need_capture():
        #     model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        # # model_worker_batch.capture_hidden_mode = capture_hidden_mode
        # logits_output, next_token_ids, _ = self.forward_batch_generation(
        #     model_worker_batch
        # )
        # return logits_output, next_token_ids, model_worker_batch.bid
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.

        # [DIAG] Check batch state before get_model_worker_batch in forward_target_extend
        _diag_bs = len(batch.reqs) if batch.reqs else 0
        _diag_rpi = batch.req_pool_indices
        _diag_sl = batch.seq_lens
        if _diag_rpi is not None and _diag_sl is not None:
            _rpi_n = _diag_rpi.numel()
            _sl_n = _diag_sl.numel()
            if _rpi_n != _diag_bs or _sl_n != _diag_bs:
                import time as _t
                _msg = f"[{_t.strftime('%H:%M:%S')}] EXTEND-INCON: bs={_diag_bs}, rpi={_rpi_n}, sl={_sl_n}, mode={batch.forward_mode}\n"
                try:
                    with open("/tmp/diag_batch_trace.log", "a") as _f:
                        _f.write(_msg)
                except:
                    pass
            if _diag_rpi.numel() > 0:
                _pool_sz = batch.req_to_token_pool.size if batch.req_to_token_pool else -1
                _rmax = _diag_rpi.max().item()
                if _pool_sz > 0 and _rmax >= _pool_sz:
                    import time as _t
                    _msg = f"[{_t.strftime('%H:%M:%S')}] EXTEND-POOL-OOB: max={_rmax} vs pool={_pool_sz}, bs={_diag_bs}\n"
                    try:
                        with open("/tmp/diag_batch_trace.log", "a") as _f:
                            _f.write(_msg)
                    except:
                        pass
        # [/DIAG]

        model_worker_batch = batch.get_model_worker_batch()
        # model_worker_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        # TODO: one static config
        # if batch.spec_info.capture_hidden_mode.need_capture():
        #     model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        if prefill_logits_num > 0:
            model_worker_batch.return_logprob = True
            seq_lens = model_worker_batch.extend_seq_lens
            batch_size = len(seq_lens)
            model_worker_batch.top_logprobs_nums = [0] * batch_size
            model_worker_batch.token_ids_logprobs = [None] * batch_size
            start_lens = [max(0, L - prefill_logits_num) for L in seq_lens]
            model_worker_batch.extend_logprob_start_lens = start_lens

        # model_worker_batch.capture_hidden_mode = capture_hidden_mode
        logits_output, next_token_ids, _ = self.forward_batch_generation(
            model_worker_batch
        )
        return logits_output, next_token_ids, model_worker_batch.bid
    
    
def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        cache_dir = snapshot_download(
            os.path.dirname(token_map_path),
            ignore_patterns=["*.bin", "*.safetensors"],
        )
        token_map_path = os.path.join(cache_dir, os.path.basename(token_map_path))
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int32)
