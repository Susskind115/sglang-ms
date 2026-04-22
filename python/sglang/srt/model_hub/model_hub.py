"""Model Hub of the SGLang."""

import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict

import torch
import random
import math
from huggingface_hub import snapshot_download
from torch.distributed import get_rank
import torch.distributed as dist

# from sglang.srt.distributed import GroupCoordinator, patch_tensor_parallel_group
# from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator
# from sglang.srt.layers.dp_attention import disable_dp_size
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyOutput

from sglang.srt.managers.schedule_batch import (
    ScheduleBatch,
    get_last_loc,
    global_server_args_dict,
    Req,
)
from sglang.srt.distributed import (
    get_tp_group,
    get_world_group,
)
from sglang.srt.managers.tp_worker_overlap_thread import TpModelWorkerClient
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
# from sglang_lib.tp_worker import TpModelWorker
# from sglang.srt.model_executor.forward_batch_info import (
#     CaptureHiddenMode,
#     ForwardBatch,
#     ForwardMode,
# )
from sglang.srt.server_args import ServerArgs
from sglang.srt.sampling.sampling_params import SamplingParams
# from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
#     EAGLEDraftCudaGraphRunner,
# )
# from sglang.srt.speculative.eagle_utils import (
#     EagleDraftInput,
#     EagleVerifyInput,
#     EagleVerifyOutput,
#     assign_draft_cache_locs,
#     generate_token_bitmask,
#     select_top_k_tokens,
# )
# # from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
# from sglang_lib.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import empty_context, fast_topk, get_available_gpu_memory, is_cuda
from sglang.srt.model_hub.utils.nvtx_marker import nvtx_profile, count_time
from sglang.srt.model_hub.utils.specrouter_debug import (
    describe_batch,
    describe_chain_diff,
    describe_spec_info,
    describe_strategy,
    specrouter_debug_log,
)

from .chain_profiler import ChainPerformanceProfiler
from .chain_scheduler import ChainScheduler
from .model_router import ModelRouter
from .chain_utils import StateManager, ChainStrategy, ChainStrategyDiff, InferenceMode, HiddenStatesPool

# global first_warmup_done

# if is_cuda():
#     from sgl_kernel import segment_packbits

logger = logging.getLogger(__name__)

class ModelHub:
    def __init__(self, server_args_list: List[ServerArgs], router_args: dict,
                 main_worker_kargs: dict,
                 is_generation: bool = False, **kwargs):
        self.main_model_worker = None
        self.model_workers = []
        self.full_model_ids = []

        self.spec_worker = None
        self.defer_memory_init = True
        self.router_args = router_args
        self.model_router = ModelRouter()
        self.state_manager = StateManager()

        self.chain_profiler: Optional[ChainPerformanceProfiler] = None
        self.chain_scheduler: Optional[ChainScheduler] = None
        self.enable_chain_profiler = router_args.get("enable_chain_profiler", True)
        self.enable_chain_scheduler = router_args.get("enable_chain_scheduler", True)
        self.chain_sampling_mode = router_args.get("chain_sampling_mode", "greedy")
        self.chain_sampling_temperature = float(
            router_args.get("chain_sampling_temperature", 1.0)
        )
        self.chain_sampling_topk = int(router_args.get("chain_sampling_topk", 0))
        topp = router_args.get("chain_sampling_topp", [0.7, 0.9])
        if isinstance(topp, (list, tuple)) and len(topp) >= 2:
            self.chain_sampling_topp = (float(topp[0]), float(topp[1]))
        else:
            self.chain_sampling_topp = (0.7, 0.9)
        self._chain_stats_ready = False

        self.worker_map: Dict[str, TpModelWorker] = {}
        self.server_args_map: Dict[str, ServerArgs] = {}
        self._chain_last_applied: Tuple[str, ...] = tuple()
        self._similarity_initialized = False
        self.req_probs_latest: Optional[torch.Tensor] = None
        self.req_probs_count: Optional[torch.Tensor] = None
        self.req_probs_active: Optional[torch.Tensor] = None
        self.compatibility_matrix: Optional[torch.Tensor] = None

        self.current_batch_probs: Optional[float] = None
        self.current_batch_size: Optional[int] = None
        self.req_prob_model_ids: List[str] = []
        self.req_prob_model_id_to_index: Dict[str, int] = {}

        # 动态切换配置：(autoregressive_limit, speculative_limit)
        # self.running_requests_limit = (4097, 256)
        self.running_requests_limit = (4097, 48)
        self.prob_alpha = 0.05 # 动量权重
        # TODO 要改的。
        self.server_args = server_args_list[-1] # target server_args
        self.restart_done = False

        # 保存原始的max_running_requests配置
        self.original_max_running_requests = self.server_args.max_running_requests

        self.is_generation = is_generation
        self.enable_overlap = not self.server_args.disable_overlap_schedule

        self.spec_flag = router_args.get("spec_flag", False)
        self.init_speculative_args(self.server_args.speculative_num_steps, 
                                            self.server_args.speculative_eagle_topk, 
                                            self.server_args.speculative_num_draft_tokens)
        self.current_mode = "autoregressive" if not self.spec_flag else "speculative"
        self.src_mode = self.current_mode

        # 动态切换相关状态
        self.mode_switch_enabled = router_args["mode_switch_enabled"]
        self.enabled_skip_extend = router_args["enabled_skip_extend"]
        self.switch_thresh = router_args["switch_thresh"]
        self.first_prefill_pass = False
        # exploration mechanism to break positive-feedback deadlock
        self._explore_interval = 30
        self._explore_burst = 2
        self._explore_count = 0
        self._explore_remaining = 0
        # logger.info(f"switch_thresh: {self.switch_thresh}")
        # self.mode_switch_enabled = router_args.get("mode_switch_enabled", True)
        # 默认配置
        self.effective_max_running_requests = self.running_requests_limit[0]  # 默认使用自回归的限制
        # self.effective_max_running_requests = 256  # 默认使用自回归的限制

        # Check whether overlap can be enabled
        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        if self.enable_overlap:
            TpWorkerClass = TpModelWorkerClient
        else:
            TpWorkerClass = TpModelWorker
        # print("来！ modelhub")

        # if orchestration_strategy == "speculative":
        #     # # params
        #     main_worker_kargs.update(self.router_args)

        #     # model hub
        #     self.model_workers = self.init_workers(server_args_list, main_worker_kargs)
        #     print("初始化完成！ modelhub")
            # self.model_workers_ids = [i for i in range(len(self.model_workers))]
            # self.model_workers_ids[-1] = "target"
            # self.spec_worker = SpeculativeModelsWorker(target_worker=target_worker, draft_workers=draft_workers, 
            #                                            target_server_args=target_server_args, draft_server_args_list=draft_server_args_list, 
            #                                            workers_ids=self.model_workers_ids, **kwargs)
        # params
        main_worker_kargs.update({"spec_flag": self.spec_flag})
        main_worker_kargs.update({"enabled_skip_extend": self.enabled_skip_extend})
        # main_worker_kargs.update(self.router_args)

        # 动态切换策略：始终按最大需求分配内存
        # 使用自回归模式的内存需求作为基准（更大的并发数）
        # if self.mode_switch_enabled:
        #     # 为了支持动态切换，总是分配足够的内存池
        #     server_args_list = server_args_list[-1:]
        #     server_args_list[-1].max_running_requests = self.running_requests_limit[0]  # 使用更大的值
        #     self.max_running_requests = self.running_requests_limit[0]
        #     logger.info(f"Dynamic mode switching enabled. Memory allocated for max_running_requests={self.max_running_requests}")
        # else:
        # 传统模式：根据spec_flag决定
        if not self.spec_flag:
            server_args_list = server_args_list[-1:]
            logger.info("spec_flag is False, only use target model")
        else:
            logger.info("spec_flag is True, use target model and draft model")

        # model hub
        selected_server_args_list = server_args_list[:]
        model_algorithm = {server_args.model_path: server_args.speculative_algorithm for server_args in selected_server_args_list}
        target_model_name = server_args_list[-1].model_path
        model_chain = [server_args.model_path for server_args in selected_server_args_list]
        # model_attr_list = [(server_args.model_path, server_args.speculative_algorithm) for server_args in selected_server_args_list_mock]

        # TODO
        # selected_server_args_list_mock = self.model_router.init_model_capture_hidden_mode(selected_server_args_list_mock)
        self.init_workers(selected_server_args_list, main_worker_kargs, TpWorkerClass)

        # self.init_hidden_states_pool()


        # set model router
        self.model_router.set_router_chain(model_chain)
        self.model_router.set_target_model_name(target_model_name)
        self.model_router.update_model_algorithm_table(model_algorithm)
        self.model_router.update_capture_hidden_mode_table()

        # set state manager
        # self.state_manager.set_attr(max_req_num=self.max_running_requests)
        self.state_manager.set_attr(max_req_num=self.max_running_requests)
        self.max_speculative_num_steps, self.max_speculative_num_topk, self.max_speculative_num_draft_tokens = self.state_manager.init_spec_params(server_args_list)
        self.current_chain_strategy = ChainStrategy(model_chain)
        self.lazy_switch_strategy = None
        self.lazy_switch_diff = None
        self.process_lazy_switch = False
        self._chain_selected_counts = defaultdict(int)
        self._chain_applied_counts = defaultdict(int)
        self._chain_observed_counts = defaultdict(int)
        self._chain_search_count = 0
        self._chain_switch_count = 0
        self._chain_lazy_switch_count = 0
        self._last_selected_chain: Optional[Tuple[str, ...]] = None

        # self.worker_map = {worker.model_name: worker for worker in self.model_workers}
        self.server_args_map = {
            server_args.model_path: server_args for server_args in selected_server_args_list
        }

        self._chain_last_applied = self._current_chain_ids()
        self._record_applied_chain(self._chain_last_applied)
        self._init_chain_components(selected_server_args_list)
        self.warmup_model_hub()


    def init_model_configs(self, server_args_list: List[ServerArgs]):
        model_configs = []
        for server_args in server_args_list[:-1]:
            model_configs.append(ModelConfig.from_server_args(
                server_args,
                model_path=(
                    server_args.model_path
                ),
                is_draft_model=True,
            ))
        model_configs.append(ModelConfig.from_server_args(
            server_args_list[-1],
            model_path=(
                server_args_list[-1].model_path
            ),
            is_draft_model=False,
        ))
        
        # for model_config in model_configs:
        #     print("model_config", vars(model_config))
        return model_configs
    
    # def get_cell_size(self, model_config: ModelConfig, is_draft_worker: bool):
    #     if is_draft_worker:
    #         num_layers = getattr(
    #             model_config.hf_config,
    #             "num_nextn_predict_layers",
    #             self.num_effective_layers,
    #         )
    #     else:
    #         num_layers = self.num_effective_layers
    #     if self.use_mla_backend:
    #         # FIXME: pipeline parallelism is not compatible with mla backend
    #         assert self.pp_size == 1
    #         cell_size = (
    #             (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
    #             * num_layers
    #             * torch._utils._element_size(self.kv_cache_dtype)
    #         )
    #     else:
    #         cell_size = (
    #             self.model_config.get_num_kv_heads(get_attention_tp_size())
    #             * self.model_config.head_dim
    #             * num_layers
    #             * 2
    #             * torch._utils._element_size(self.kv_cache_dtype)
    #         )
    #     # max_num_token = int(visible_memory * (1 << 30) // cell_size)
        
    #     return num_layers, cell_size

    def profile_model_memory_rate(self, workers, available_gpu_memory):
        cell_sizes = []
        for worker in workers:
            num_layer, cell_size = worker.model_runner.get_model_kvcache_info()
            cell_sizes.append(cell_size)
        sum_cell_size = sum(cell_sizes)
        worker_rates = [cell_size / sum_cell_size for cell_size in cell_sizes]
        worker_memorys = [worker_rate * available_gpu_memory for worker_rate in worker_rates]
        return worker_rates, worker_memorys
        

    def init_workers(self, server_args_list: List[ServerArgs], main_worker_kargs: dict, TpWorkerClass: Union[TpModelWorker, TpModelWorkerClient]):
        model_configs = self.init_model_configs(server_args_list)
        # model_mem_rates = self.profile_model_memory_rate(model_configs=model_configs)
        main_model_worker = TpWorkerClass(server_args=server_args_list[-1], model_config=model_configs[-1], defer_memory_init=self.defer_memory_init, **main_worker_kargs)
        max_cos_sin_cache_lens = main_model_worker.model_runner.model.config.max_position_embeddings
        draft_worker_kargs = main_worker_kargs.copy()
        draft_worker_kargs["is_draft_worker"] = True
        workers = []
        
        for i, server_args in enumerate(server_args_list[:-1]):
            # TODO 暂时测试性能，不追求优雅设计
            server_args.context_length = main_model_worker.model_runner.model_config.context_len
            draft_worker = TpModelWorker(server_args=server_args, **draft_worker_kargs, 
                                         model_config=model_configs[i],
                                         defer_memory_init=self.defer_memory_init)
            
            # if draft_worker.speculative_algorithm.is_eagle():
            #     # TODO 需要手动设定一种指向eagle base模型的方式，现在先粗糙实现
            #     draft_worker.set_eagle_embed_and_head(main_model_worker)
            draft_worker.model_runner.model.update_max_cos_sin_cache(max_cos_sin_cache_lens)
            workers.append(draft_worker)
            self.full_model_ids.append(draft_worker.model_name)
        workers.append(main_model_worker)
        self.full_model_ids.append(main_model_worker.model_name)

        device = main_model_worker.model_runner.device
        dtype = main_model_worker.model_runner.dtype
        self.dtype = dtype

        self.model_workers = workers
        self.worker_map = {worker.model_name: worker for worker in workers}
        logger.info(f"full_model_ids: {self.full_model_ids}")


        # compatibility matrix
        num_models = len(self.full_model_ids)
        name_to_idx = {name: idx for idx, name in enumerate(self.full_model_ids)}
        compatibility_matrix = torch.ones(
            (num_models, num_models), 
            dtype=torch.bool, device=device
        )
        compatibility_matrix[-1,-1] = False
        for i, draft_worker in enumerate(workers[:-1]):
            if draft_worker.speculative_algorithm.is_eagle():
                draft_worker.set_eagle_embed_and_head(self.worker_map[draft_worker.base_model_name])
                compatibility_matrix[i, :] = False
                compatibility_matrix[:, i] = False
                compatibility_matrix[i, name_to_idx[draft_worker.base_model_name]] = True

                # For EAGLE3 draft models, the target model must capture
                # intermediate-layer aux hidden states (3x hidden_size)
                # so the draft model's fc layer can transform them properly.
                if draft_worker.speculative_algorithm.is_eagle3():
                    target_worker = self.worker_map[draft_worker.base_model_name]
                    target_model = target_worker.model_runner.model
                    if hasattr(target_model, "set_eagle3_layers_to_capture"):
                        target_model.set_eagle3_layers_to_capture()
                        target_worker.model_runner.capture_aux_hidden_states = True
        self.compatibility_matrix = compatibility_matrix

        # 初始化main_model_worker的memory pool
        total_gpu_memory = main_model_worker.model_runner.pending_memory_init["min_per_gpu_memory"]
        mem_fraction_static = main_model_worker.model_runner.mem_fraction_static
        available_gpu_memory = get_available_gpu_memory(
            main_model_worker.model_runner.device,
            main_model_worker.model_runner.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - mem_fraction_static
        )
        # print(f"available_gpu_memory:{available_gpu_memory}, total_gpu_memory:{total_gpu_memory}, mem_fraction_static:{mem_fraction_static}, rest_memory:{rest_memory}")
        worker_rates, worker_memorys = self.profile_model_memory_rate(workers, rest_memory)

        main_model_worker.post_init_model_runner(self.defer_memory_init, visible_gpu_memory=worker_memorys[-1])
        req_to_token_pool, token_to_kv_pool_allocator = main_model_worker.get_memory_pool()
        req_to_token_pool.set_from_model_hub(True)
        token_to_kv_pool_allocator.set_from_model_hub(True)
        
        draft_runner_cache_size = main_model_worker.model_runner.server_args.draft_runner_cache_size
        max_num_reqs = main_model_worker.model_runner.server_args.max_num_reqs
        self.max_running_requests = max_num_reqs

        for i, worker in enumerate(workers[:-1]):
            worker.post_init_model_runner(self.defer_memory_init, 
                                          visible_gpu_memory=worker_memorys[i],
                                          req_to_token_pool=req_to_token_pool, 
                                          token_to_kv_pool_allocator=token_to_kv_pool_allocator, 
                                          draft_runner_cache_size=draft_runner_cache_size, 
                                          max_num_reqs=max_num_reqs, 
                                          )
            # if get_rank() == 0:
            #     logger.info(f"token_to_kv_pool_allocator: {token_to_kv_pool_allocator}")
            # logger.info(f"worker.kvcache == token_to_kv_pool_allocator.kvcache: {worker.model_runner.token_to_kv_pool == token_to_kv_pool_allocator.get_kvcache()}")
            # logger.info(f"worker.model_runner.token_to_kv_pool_allocator == token_to_kv_pool_allocator: {worker.model_runner.token_to_kv_pool_allocator.get_kvcache() == token_to_kv_pool_allocator.get_kvcache()}")
            # logger.info(f"main_model_worker.model_runner.token_to_kv_pool_allocator == token_to_kv_pool_allocator: {main_model_worker.model_runner.token_to_kv_pool_allocator.get_kvcache() == token_to_kv_pool_allocator.get_kvcache()}")

        
        self._init_request_prob_storage(
            main_model_worker.model_runner.device, 
            main_model_worker.model_runner.dtype, 
            max_num_reqs
        )


        # return workers
    
    def init_hidden_states_pool(self):
        self.model_hidden_states_pools = defaultdict(lambda: None)
        draft_worker = self.model_workers[0]
        self.model_hidden_states_pools[draft_worker.base_model_name] = HiddenStatesPool.create_from_draft_worker(draft_worker)

    def _current_chain_ids(self) -> List[str]:
        # [draft_model_small, draft_model_large, ..., target_model]
        return [worker.model_name for worker in self.model_workers]
        # return self.model_router.router_chain

    def _init_chain_components(self, server_args_list: Sequence[ServerArgs]) -> None:
        if self.enable_chain_profiler:
            self.chain_profiler = ChainPerformanceProfiler(
                enable_nvtx=self.router_args.get("enable_nvtx", False),
                log_to_console=self.router_args.get("profiler_log_to_console", False),
                summary_to_console=self.router_args.get("profiler_summary_to_console", False),
            )
        else:
            self.chain_profiler = None

        enable_scheduler = (
            self.enable_chain_scheduler
            and self.spec_flag
            and len(self.model_workers) > 1
        )
        if not enable_scheduler:
            self.chain_scheduler = None
            self._chain_stats_ready = False
            self._similarity_initialized = False
            return

        model_ids = self._current_chain_ids()
        self.chain_scheduler = ChainScheduler(
            model_ids,
            profiler=self.chain_profiler,
            strategy=self.router_args.get("chain_strategy", "greedy"),
            compatibility_matrix=self.compatibility_matrix,
            max_window_size=self.speculative_num_steps,
            dtype=self.dtype,
        )
        self.chain_scheduler.init_global_time_dict({mid: 0.0 for mid in model_ids})
        self.chain_scheduler.update_model_chain(model_ids)
        self._chain_stats_ready = False
        self._similarity_initialized = False

    @property
    def current_spec_flag(self):
        return len(self.model_workers) > 1

    def warmup_model_hub(self) -> None:
        # if not self.router_args.get("enable_model_warmup", True):
        #     return
        if not self.model_workers:
            return
        warmup_prompt_len1 = 128
        warmup_prompt_len2 = 256

        for worker in self.model_workers:
            if get_rank() == 0:
                logger.info(f"warmup_model_hub worker: {worker.model_name}")
            self._warmup_single_worker(worker, warmup_prompt_len1, warmup_prompt_len2, rounds=10)
            # try:
            #     self._warmup_single_worker(worker, rounds=10)
            # except Exception as exc:  # pragma: no cover - warmup best-effort
            #     if get_rank() == 0:
            #         logger.warning(
            #             "Warmup for model %s skipped due to error: %s",
            #             getattr(worker, "model_name", "unknown"),
            #             exc,
            #         )
        if self.chain_scheduler:
            self.chain_scheduler.sync_warmup_data(warmup_prompt_len1, warmup_prompt_len2)

    def _warmup_single_worker(self, worker: TpModelWorker, warmup_prompt_len1: int, warmup_prompt_len2: int, rounds: int = 1) -> None:
        runner = worker.model_runner
        req_pool = runner.req_to_token_pool
        kv_allocator = runner.token_to_kv_pool_allocator

        if req_pool is None or kv_allocator is None:
            return

        req_free_backup = list(getattr(req_pool, "free_slots", []))
        kv_free_backup = None
        if hasattr(kv_allocator, "backup_state"):
            kv_state = kv_allocator.backup_state()
            kv_free_backup = (
                kv_state.clone() if isinstance(kv_state, torch.Tensor) else kv_state
            )

        warmup_batch_size = 128
        # warmup_prompt_len = 2048
        # warmup_prompt_len2 = 4096
        batch, token_sequence = self._build_warmup_batch(worker, default_prompt_len=1, default_batch_size=warmup_batch_size)
        long_batch, long_token_sequence = self._build_warmup_batch(worker, default_prompt_len=warmup_prompt_len1, default_batch_size=1)
        long_batch2, long_token_sequence2 = self._build_warmup_batch(worker, default_prompt_len=warmup_prompt_len2, default_batch_size=1)
        solo_batch, solo_token_sequence = self._build_warmup_batch(worker, default_prompt_len=1, default_batch_size=1)

        batch_recompile, token_sequence_recompile = self._build_warmup_batch(worker, default_prompt_len=1, default_batch_size=warmup_batch_size)
        solo_batch_recompile, solo_token_sequence_recompile = self._build_warmup_batch(worker, default_prompt_len=1, default_batch_size=1)
        batch.prepare_for_extend()
        solo_batch.prepare_for_extend()
        long_batch.prepare_for_extend()
        long_batch2.prepare_for_extend()
        batch_recompile.prepare_for_extend()
        solo_batch_recompile.prepare_for_extend()
        self._maybe_init_eagle_spec_info(worker, batch, token_sequence[0], 'full')
        self._maybe_init_eagle_spec_info(worker, long_batch2, long_token_sequence2[0], 'full')
        self._maybe_init_eagle_spec_info(worker, solo_batch, solo_token_sequence[0], 'full')
        self._maybe_init_eagle_spec_info(worker, long_batch, long_token_sequence[0], 'full')
        self._maybe_init_eagle_spec_info(worker, batch_recompile, token_sequence_recompile[0], 'full')
        self._maybe_init_eagle_spec_info(worker, solo_batch_recompile, solo_token_sequence_recompile[0], 'full')
        if solo_batch is None:
            return

        try:
            with torch.no_grad():
                if worker.model_name != self.model_router.target_model_name:
                    # warm up draft compile
                    # bs = 1
                    # model_worker_batch = solo_batch.get_model_worker_batch()
                    worker.draft(solo_batch_recompile, worker.speculative_num_steps+1)

                    # bs = 128
                    # model_worker_batch = batch.get_model_worker_batch()
                    worker.draft(batch_recompile, worker.speculative_num_steps+1)

                model_worker_batch = long_batch.get_model_worker_batch()
                for i in range(rounds):
                    self._profile_call(
                        ["warmup_prefill1", "prefill"],
                        # "prefill",
                        1, # forward_count
                        1.0, # verify_K
                        worker,
                        worker.forward_batch_generation,
                        model_worker_batch,
                        None,
                        True,
                    )
                
                model_worker_batch = long_batch2.get_model_worker_batch()
                for i in range(rounds):
                    self._profile_call(
                        ["warmup_prefill2"],
                        # "prefill",
                        1, # forward_count
                        1.0, # verify_K
                        worker,
                        worker.forward_batch_generation,
                        model_worker_batch,
                        None,
                        True,
                    )

                # dummy_next_token = (
                #     int(batch.input_ids[-1].item()) if batch.input_ids.numel() > 0 else 0
                # )
                # self._maybe_init_eagle_spec_info(worker, batch, [dummy_next_token], 'last')

                # warmup_token = torch.tensor(
                #     [dummy_next_token]*warmup_batch_size, dtype=torch.int64, device=batch.device
                # )
                # batch.output_ids = warmup_token
                # for req in batch.reqs:
                #     req.output_ids = [int(warmup_token[0].item())]
                #     req.fill_ids = list(req.origin_input_ids) + req.output_ids
                #     req.extend_input_len = 1

                # batch.prepare_for_decode()

                decode_worker_batch = batch.get_model_worker_batch()
                for i in range(rounds):
                    self._profile_call(
                        ["draft", "warmup_decode_bs"],
                        # "warmup_decode",
                        1, # forward_count
                        warmup_batch_size, # verify_K
                        worker,
                        worker.forward_batch_generation,
                        decode_worker_batch,
                        None,
                        True,
                    )
                
                model_worker_batch = solo_batch.get_model_worker_batch()
                for i in range(rounds):
                    self._profile_call(
                        ["warmup_decode", "draft"],
                        # "prefill",
                        1, # forward_count
                        1.0, # verify_K
                        worker,
                        worker.forward_batch_generation,
                        model_worker_batch,
                        None,
                        True,
                    )
                
                

            if is_cuda():
                torch.cuda.synchronize(worker.model_runner.gpu_id)
        finally:
            if kv_free_backup is not None and hasattr(kv_allocator, "restore_state"):
                kv_allocator.restore_state(kv_free_backup)
            if req_free_backup is not None:
                req_pool.free_slots = list(req_free_backup)

    def _build_warmup_batch(self, worker: TpModelWorker, default_prompt_len: int = 128, default_batch_size: int = 1) -> Optional[Tuple[ScheduleBatch, List[List[int]]]]:
        runner = worker.model_runner
        req_pool = runner.req_to_token_pool
        kv_allocator = runner.token_to_kv_pool_allocator
        if req_pool is None or kv_allocator is None:
            return None

        model_config = runner.model_config
        context_len = getattr(model_config, "context_len", default_prompt_len)
        prompt_len = max(1, min(default_prompt_len, context_len))

        vocab_size = getattr(model_config, "vocab_size", 32000) or 32000
        token_base = max(vocab_size - 1, 1)

        reqs = []
        all_token_sequences = []

        for i in range(default_batch_size):
            token_sequence = [random.randint(1, token_base) for _ in range(prompt_len)]
            
            req_id = f"__warmup__{worker.model_name}_{i}"
            
            sampling_params = SamplingParams(max_new_tokens=1)
            req = Req(
                rid=req_id,
                origin_input_text="",
                origin_input_ids=tuple(token_sequence),
                sampling_params=sampling_params,
            )
            req.fill_ids = list(token_sequence)
            req.extend_input_len = len(req.fill_ids)
            req.output_ids = []
            
            reqs.append(req)
            all_token_sequences.append(token_sequence)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=kv_allocator,
            tree_cache=None,
            model_config=model_config,
            enable_overlap=self.enable_overlap,
            spec_flag=False,
            enable_custom_logit_processor=False,
        )

        return batch, all_token_sequences

    def _maybe_init_eagle_spec_info(
        self,
        worker: TpModelWorker,
        batch: ScheduleBatch,
        token_sequence: Union[List[int], torch.Tensor],
        stage: str="last",
    ) -> None:
        if worker.model_name == self.model_router.target_model_name:
            return

        runner = worker.model_runner
        device = batch.device
        batch_size = batch.batch_size()
        if batch_size == 0:
            return

        capture_mode = self.model_router.get_model_capture_hidden_mode_from_table(worker.model_name, stage)
        hidden_size = getattr(runner.model_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(
                getattr(runner.model_config, "hf_config", None), "hidden_size", 0
            )
        hidden_size = hidden_size or 0
        # logger.info(f"hidden_size: {hidden_size}, model_name: {worker.model_name}")

        dtype = getattr(runner, "dtype", torch.bfloat16)
        hidden_states = None
        if worker.speculative_algorithm.is_eagle():
            if hidden_size > 0 and capture_mode != CaptureHiddenMode.NULL:
                if capture_mode == CaptureHiddenMode.FULL:
                    length = batch.extend_num_tokens or (batch_size * 1)
                    hidden_shape = (length, hidden_size)
                else:
                    hidden_shape = (batch_size, hidden_size)
                hidden_states = torch.randn(hidden_shape, device=device, dtype=dtype)

        if worker.speculative_algorithm.is_eagle3():
            vocab_size = getattr(runner.model_config, "draft_vocab_size", None)
        else:
            vocab_size = getattr(runner.model_config, "vocab_size", None)
        if not vocab_size:
            vocab_size = 32000
        logits_dtype = torch.float32
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=logits_dtype)
        topk = getattr(worker, "topk", None)
        if topk is None:
            topk = max(1, runner.server_args.speculative_eagle_topk)
        topk = min(max(1, topk), vocab_size)
        topk_p, topk_index = torch.topk(torch.softmax(logits, dim=-1), k=topk, dim=-1)
        topk_index = topk_index.to(torch.long)
        
        if isinstance(token_sequence, torch.Tensor):
            verified_id = token_sequence
        else:
            last_token = token_sequence[-1] if token_sequence else 0
            verified_id = torch.full(
                (batch_size,),
                int(last_token),
                dtype=torch.int32,
                device=device,
            )

        accept_len = torch.ones(batch_size, dtype=torch.int32, device=device)
        draft_input = EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            capture_hidden_mode=capture_mode,
            verified_id=verified_id,
            accept_length=accept_len,
            accept_length_cpu=[int(v) for v in accept_len.cpu().tolist()],
        )
        batch.spec_info = draft_input

    def _default_set_req_prob(self):
        self.req_probs_latest = None
        self.req_probs_count = None
        self.req_probs_active = None
        self.req_prob_model_ids = []
        self.req_prob_model_id_to_index = {}
        self.current_batch_probs = None

    def _init_request_prob_storage(self, device: torch.device, dtype: torch.dtype, size: int) -> None:
        if size <= 0:
            self._default_set_req_prob()
            return
        model_ids = [worker.model_name for worker in self.model_workers]
        num_models = len(model_ids)
        # print("init_request_prob_storage", num_models, size)
        if num_models == 0:
            self._default_set_req_prob()
            return

        self.req_prob_model_ids = model_ids
        self.req_prob_model_id_to_index = {
            model_id: idx for idx, model_id in enumerate(model_ids)
        } # same as time vec
        # self.req_probs_latest = torch.(
        #     size, num_models, num_models, device=device, dtype=torch.float32
        # )
        # self.req_probs_count_protect_threshold = 0.0
        self.req_probs_count_protect_threshold = 5.0
        self.req_probs_init = 0.9
        # self.req_probs_init = 0.6
        self.req_probs_latest = torch.full((size, num_models, num_models), self.req_probs_init, device=device, dtype=dtype)
        # self.req_probs_latest = (
        #     torch.eye(num_models, device=device, dtype=dtype)
        #     .unsqueeze(0)
        #     .repeat(size, 1, 1)
        # )
        self.req_probs_count = (
            torch.eye(num_models, device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(size, 1, 1)
        )
        # self.req_probs_count = torch.zeros(size, device=device, dtype=torch.float32)
        self.req_probs_active = torch.zeros(size, device=device, dtype=torch.bool)
        self.current_batch_probs = None

    # def _ensure_prob_storage_matches_models(self) -> None:
    #     if not self.model_workers:
    #         self._default_set_req_prob()
    #         return

    #     if self.req_probs_latest is None:
    #         self._init_request_prob_storage(
    #             self.model_workers[-1].model_runner.device, self.max_running_requests
    #         )
    #         return

    #     new_ids = [worker.model_name for worker in self.model_workers]
    #     old_ids = getattr(self, "req_prob_model_ids", [])

    #     if new_ids == old_ids:
    #         return

    #     if not old_ids:
    #         self._init_request_prob_storage(
    #             self.model_workers[-1].model_runner.device, self.max_running_requests
    #         )
    #         return

    #     device = self.req_probs_latest.device
    #     try:
    #         perm_indices = [old_ids.index(model_id) for model_id in new_ids]
    #     except ValueError:
    #         self._init_request_prob_storage(device, self.max_running_requests)
    #         return

    #     perm_tensor = torch.tensor(perm_indices, device=device, dtype=torch.long)
    #     # TODO 重排是否可以换成逻辑重排。
    #     logger.info(f"perm_tensor: {self.req_probs_latest.shape}")
    #     self.req_probs_latest = self.req_probs_latest.index_select(1, perm_tensor)
    #     self.req_probs_latest = self.req_probs_latest.index_select(2, perm_tensor)
    #     logger.info(f"perm_tensor: {self.req_probs_latest.shape}")
    #     self.req_prob_model_ids = new_ids
    #     self.req_prob_model_id_to_index = {
    #         model_id: idx for idx, model_id in enumerate(new_ids)
    #     }

    def _profile_call(self, name: Union[str, List[str]], forward_count: int, verify_K: float, worker, func, *args, **kwargs):
        if self.chain_profiler is None or worker is None:
            return func(*args, **kwargs)
        with self.chain_profiler.range(
            name, forward_count, model_id=worker.model_name, ignore_level=True, verify_K=verify_K
        ):
            return func(*args, **kwargs)

    def print_chain_scheduler(self):
        if get_rank() == 0:
            # logger.info(self.chain_scheduler.get_models_chain())
            logger.info(self.chain_scheduler.get_global_time_dict())
            logger.info(self.chain_scheduler.get_global_similarity_matrix())  


    # def _update_chain_time_stats(self, mode='draft') -> None:
    #     if not self.chain_scheduler or not self.chain_profiler:
    #         return
    #     # model_ids_tuple = [self.chain_scheduler.model_ids[0], "draft"]
    #     if mode == 'draft':
    #         model_ids_tuple = [(mid, "verify") for mid in self.chain_scheduler.model_ids[1:]]
    #         model_ids_tuple.append((self.chain_scheduler.model_ids[0], "draft"))
    #     elif mode == 'prefill':
    #         model_ids_tuple = [(mid, "prefill") for mid in self.chain_scheduler.model_ids]
    #     else:
    #         raise ValueError(f"Invalid mode: {mode}")
    #     time_dict = self.chain_profiler.gen_model_time_dict(
    #         model_ids_tuple
    #     )
    #     if any(value is not None for value in time_dict.values()):
    #         self.chain_scheduler.update_global_time_dict(
    #             time_dict
    #         )
    #         self._chain_stats_ready = True
    #     self.chain_scheduler.update_decode_time_dict(self.chain_profiler._model_decode_timers)

    def _attach_batch_probs(self, batch: ScheduleBatch) -> Optional[float]:
        if (
            self.req_probs_latest is None
            or self.req_probs_active is None
            or getattr(batch, "req_pool_indices", None) is None
            or not self.req_prob_model_ids
        ):
            # batch.batch_probs_of_draft = None
            self.current_batch_probs = None
            return None

        req_indices = batch.req_pool_indices
        req_indices = req_indices.to(self.req_probs_latest.device, dtype=torch.long)
        if req_indices.numel() == 0:
            self.current_batch_probs = None
            return None
        if torch.any(req_indices < 0) or torch.any(
            req_indices >= self.req_probs_latest.shape[0]
        ):
            self.current_batch_probs = None
            return None
        # probs = self.req_probs_latest.index_select(0, req_indices)
        # value = probs.mean(dim=0)
        probs_value = self.req_probs_latest.index_select(0, req_indices).mean(dim=0)
        # value = probs
        count_value = self.req_probs_count.index_select(0, req_indices).mean(dim=0)
        mixed_probs = torch.where(count_value < self.req_probs_count_protect_threshold, self.req_probs_init, probs_value)
        # mixed_probs = probs_value
        # batch.batch_probs_of_draft = probs_value
        self.current_batch_probs = mixed_probs
        self.current_batch_size = batch.batch_size()
        
        # if get_rank() == 0:
        #     logger.info(f"self.current_batch_probs: {self.current_batch_probs}")
        # logger.info(f"count_value: {count_value}")
        # return count_value

    def _update_req_probs_from_verify(
        self,
        batch: ScheduleBatch,
        probs_of_draft: Optional[torch.Tensor],
        probs_count: Optional[torch.Tensor],
        target_model_name: str,
        draft_model_name: str,
    ) -> None:
        if (
            self.chain_scheduler is None
            or probs_of_draft is None
            or probs_count is None
            or self.req_probs_latest is None
            or self.req_probs_count is None
            or self.req_probs_active is None
        ):
            return
        
        device = self.req_probs_latest.device
        dtype = self.req_probs_latest.dtype
        probs_tensor = probs_of_draft.detach().to(
            device, dtype=dtype
        )
        # probs_count = self.speculative_num_draft_tokens-1
        # probs_count_safefy = (probs_count+1).clamp(max=self.speculative_num_draft_tokens-1)
        
        req_indices = batch.req_pool_indices.to(device, dtype=torch.long)
        # req_indices = req_indices
        req_slice = req_indices
        # probs_slice = probs_tensor
        probs_slice = probs_tensor
        # self.req_prob_model_id_to_index 得对齐
        draft_idx = self.req_prob_model_id_to_index.get(draft_model_name)
        # target_model_name = self.model_workers[-1].model_name
        target_idx = self.req_prob_model_id_to_index.get(target_model_name)

        # 关键作用
        old_probs_slice = self.req_probs_latest[req_slice, draft_idx, target_idx]
        old_count_slice = self.req_probs_count[req_slice, draft_idx, target_idx]

        # # 公式: new_value = (1 - alpha) * old_value + alpha * new_observation
        # new_probs_slice = (
        #     (1 - self.prob_alpha) * old_probs_slice + self.prob_alpha * probs_slice
        # )

        # first_update_mask = ~self.req_probs_active[req_slice]
        # final_probs = torch.where(first_update_mask, probs_slice, new_probs_slice)
        # # 动量更新



        final_probs = (old_probs_slice*old_count_slice + probs_slice*probs_count)/(old_count_slice+probs_count)
        # if get_rank() == 0:
        #     logger.info(f"old_count_slice: {old_count_slice}")
        #     logger.info(f"probs_count: {probs_count}")
        # final_probs = (old_probs_slice*old_count_slice + probs_slice)/(old_count_slice+1)

        # uodate
        self.req_probs_latest[req_slice, draft_idx, target_idx] = final_probs
        self.req_probs_latest[req_slice, target_idx, draft_idx] = final_probs
        self.req_probs_count[req_slice, draft_idx, target_idx] += probs_count
        self.req_probs_count[req_slice, target_idx, draft_idx] += probs_count
        # self.req_probs_count[req_slice] += 1
        self.req_probs_active.index_fill_(0, req_slice, True)

        if target_model_name == self.model_router.target_model_name:
            self._check_finished_reqs(batch)


    def _check_finished_reqs(self, batch: ScheduleBatch) -> None:
        # 清理批中已完成但未在循环中覆盖到的请求
        req_indices = batch.req_pool_indices.to(self.req_probs_latest.device, dtype=torch.long)
        finished_indices: List[int] = []
        for offset, req in enumerate(batch.reqs):
            if req.finished():
                finished_indices.append(int(req_indices[offset].item()))
        if finished_indices:
            finished_tensor = torch.tensor(
                finished_indices, device=self.req_probs_latest.device, dtype=torch.long
            )
            self._reset_req_probs(finished_tensor)
            # self.req_probs_active.index_fill_(0, finished_tensor, False)
            # self.req_probs_count.index_fill_(0, finished_tensor, 0)

            # self.req_probs_latest.index_fill_(0, finished_tensor, 0.0)
            # if get_rank() == 0:
            #     logger.info(f"req_probs_latest: {self.req_probs_latest[finished_tensor, draft_idx, target_idx]}")

    def _reset_req_probs(self, finished_tensor: torch.Tensor) -> None:
        self.req_probs_active.index_fill_(0, finished_tensor, False)
        self.req_probs_count.index_fill_(0, finished_tensor, 0)
        # TODO 确认效果
        self.req_probs_latest.index_fill_(0, finished_tensor, self.req_probs_init)
    
    def _update_req_probs_from_prefill(
        self,
        batch: ScheduleBatch,
        similarity_matrix: Optional[torch.Tensor],
        model_names: List[str],
        probs_count: Optional[Union[torch.Tensor, int]],
        # draft_model_name: str,
    ) -> None:
        if (
            self.chain_scheduler is None
            or similarity_matrix is None
            or probs_count is None
            or self.req_probs_latest is None
            or self.req_probs_count is None
            or self.req_probs_active is None
        ):
            return

        device = self.req_probs_latest.device
        probs_tensor = similarity_matrix.detach().to(
            device, dtype=self.req_probs_latest.dtype
        )
        B, N = similarity_matrix.shape[:2]
        # N index proj
        global_model_indices = torch.tensor(
            [self.req_prob_model_id_to_index[name] for name in model_names],
            device=device,
            dtype=torch.long
        )
        idx_model_i = global_model_indices.view(1, N, 1)
        idx_model_j = global_model_indices.view(1, 1, N)
        
        # # B index proj
        req_indices = batch.req_pool_indices.to(device, dtype=torch.long)
        idx_req = req_indices.view(B, 1, 1)

        self.req_probs_latest[idx_req, idx_model_i, idx_model_j] = probs_tensor
        self.req_probs_count[idx_req, idx_model_i, idx_model_j] = probs_count
        self.req_probs_active.index_fill_(0, req_indices, True)
    
    
    def _batch_compute_and_update_similarity(
        self, 
        stacked_logits: torch.Tensor,  # Shape: [1, B, S, V]
        weights_list: List[float],  # Shape: [N]
        ) -> None:

        if not self.chain_scheduler:
            return
            
        device = self.chain_scheduler.device if self.chain_scheduler else stacked_logits.device
        # logits_target = logits_target.to(device)
        stacked_logits = stacked_logits.to(device)
        # stacked_logits: [N, B, S, V]
        N, B, S, V = stacked_logits.shape
        # N = N_draft + 1
        # stacked_probs: [N, B, S, V]
        stacked_probs = torch.softmax(stacked_logits, dim=-1)
        # stacked_probs = torch.cat([logits_target, stacked_probs], dim=0)
        # stacked_candidates: [N, B, S]
        stacked_candidates = torch.argmax(stacked_logits, dim=-1)
        # probs_at_steps: [N, B, S, V]
        # probs_at_steps = logits_target.expand(N, B, V)
        probs_at_steps = stacked_probs
        # candidates_at_steps: [N, B, S]
        candidates_at_steps = stacked_candidates
        # 扩展 probs (Target i): [N, 1, B, S, V]
        target_probs_expanded = probs_at_steps.unsqueeze(1)
        # 扩展 candidates (Draft j): [1, N, B, S, 1]
        draft_candidates_expanded = candidates_at_steps.unsqueeze(0).unsqueeze(-1)
        # broadcasted_probs: [N, N, B, S, V]
        broadcasted_probs = target_probs_expanded.expand(
            N, N, B, S, V
        )
        # broadcasted_candidates: [N, N, B, S, 1]
        broadcasted_candidates = draft_candidates_expanded.expand(
            N, N, B, S, 1
        )

        # all_scores: [N, N, B, S, 1]
        all_scores = torch.gather(
            broadcasted_probs,         # [N, N, B, S, V]
            4,                         # 在 V 维度 (dim=3) 上 gather
            broadcasted_candidates     # [N, N, B, S, 1]
        )
        # score_matrix: [B, N, N, S]
        score_matrix = all_scores.squeeze(-1).permute(2, 0, 1, 3).mean(dim=-1)
        # logger.info(f"score_matrix: {score_matrix.mean(dim=0)}")
        symmetric_matrix = (score_matrix + score_matrix.permute(0, 2, 1)) / 2.0

        weights = torch.tensor(weights_list, device=device)
        # weights_i: [N, 1]
        weights_i = weights.unsqueeze(1)
        # weights_j: [1, N]
        weights_j = weights.unsqueeze(0)
        weights_i_j = weights_i * weights_j

        symmetric_matrix = symmetric_matrix * weights_i_j.unsqueeze(0)

        # clamp
        non_diag_mask = ~torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        eye_matrix = torch.eye(N, device=device).unsqueeze(0)
        final_matrix = (symmetric_matrix * non_diag_mask) + eye_matrix

        return final_matrix

    # def _compute_distribution_from_logits(
    #     self, logits_output: Optional[LogitsProcessorOutput]
    # ) -> Optional[torch.Tensor]:
    #     if logits_output is None or getattr(logits_output, "next_token_logits", None) is None:
    #         return None
    #     logits = logits_output.next_token_logits.detach().to(torch.float32)
    #     #  [B, V]
    #     logger.info(f"logits: {logits.shape}")
    #     if logits.ndim == 1:
    #         logits = logits.unsqueeze(0)
    #     probs = torch.softmax(logits, dim=-1)
    #     logger.info(f"probs: {probs.max(dim=-1)}")
    #     probs = probs.mean(dim=0, keepdim=True)
    #     device = self.chain_scheduler.device if self.chain_scheduler else probs.device
    #     return probs.to(device)
    
    # def _compute_probs_and_candidates(
    #     self, logits_output: Optional[LogitsProcessorOutput]
    # ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    #     if logits_output is None or getattr(logits_output, "next_token_logits", None) is None:
    #         return None
        
    #     logits = logits_output.next_token_logits.detach().to(torch.float32)
    #     if logits.ndim == 1:
    #         logits = logits.unsqueeze(0)
    #     probs = torch.softmax(logits, dim=-1)
    #     candidates = torch.argmax(logits, dim=-1)
    #     device = self.chain_scheduler.device if self.chain_scheduler else probs.device
        
    #     return probs.to(device), candidates.to(device)




    # def _update_similarity_with_probabilities(self, dist_map: Dict[str, torch.Tensor]) -> None:
    #     if not self.chain_scheduler or len(dist_map) < 2:
    #         return
    #     chain = self.chain_scheduler.get_models_chain()
    #     ordered_pairs = self.chain_scheduler.model_product_unique(chain)
    #     similarity_dict: Dict[Tuple[str, str], Tuple[float, float]] = {}
    #     for i, j in ordered_pairs:
    #         if i not in dist_map or j not in dist_map:
    #             continue
    #         similarity_dict[(i, j)] = self.chain_scheduler.compute_similarity(
    #             dist_map[i], dist_map[j]
    #         )
    #     if not similarity_dict:
    #         return
    #     # if not getattr(self, "_similarity_initialized", False):
    #     #     for (i, j), (value, count) in similarity_dict.items():
    #     #         self.chain_scheduler.init_global_similarity(i, j, value, count)
    #     #     # self._similarity_initialized = True
    #     # else:
    #     #     self.chain_scheduler.update_similarity_batch(
    #     #         similarity_dict, realtime_syn=True
    #     #     )
    #     for (i, j), (value, count) in similarity_dict.items():
    #             self.chain_scheduler.init_global_similarity(i, j, value, count)

    def _search_chain(self, min_accept_length: int) -> None:
        # use self.current_batch_probs
        if not self.chain_scheduler:
            return None, None, None

        # with count_time("sync_stats_from_profiler"):
        self.chain_scheduler.sync_stats_from_profiler(
            self.current_batch_size, self.state_manager.model_verify_windows
        )
        target_worker = self.model_workers[-1]
        # window_size = getattr(
        #     self.model_workers[0],
        #     "speculative_num_steps",
        #     self.server_args.speculative_num_steps,
        # )

        # with count_time("predict_sublists_time"):
        # if count_value < 10:
        #     return None, None, None
        cascade_list = self.chain_scheduler.predict_sublists_time(
            target_worker.model_name, self.current_batch_size, self.current_batch_probs, min_accept_length
        )
        if not cascade_list:
            return
        # selected_chain = self.chain_scheduler.sampling_model_chain(
        #     cascade_list,
        #     mode=self.chain_sampling_mode,
        #     temperature=self.chain_sampling_temperature,
        #     k=self.chain_sampling_topk,
        #     p=self.chain_sampling_topp,
        #     generated_rate=0.0,
        # )
        # if len(selected_chain) < 2:
        #     return
        # if tuple(selected_chain) != self._chain_last_applied:
        #     self._apply_model_chain(selected_chain)

        # [EAGLE3_DEBUG] chain scheduler decision diagnostic
        if os.path.exists("/tmp/EAGLE3_DEBUG_CHAIN") and get_rank() == 0:
            opt = self.chain_scheduler.optimizer
            tm = self.chain_scheduler.time_manager
            logger.info(
                f"[EAGLE3_DEBUG] _search_chain | "
                f"bs={self.current_batch_size} min_accept={min_accept_length} "
                f"alpha_matrix={opt.alpha_matrix.tolist()} "
                f"dp_cost={opt.dp_cost.tolist()} "
                f"dp_prev={opt.dp_prev.tolist()} "
                f"dp_gamma={opt.dp_gamma.tolist()} "
                f"t_decode_base={tm.warmup_decode_base.tolist()} "
                f"t_decode_slope={tm.warmup_decode_slope.tolist()} "
                f"t_decode_runtime={tm.runtime_decode_unit.tolist()} "
                f"t_verify_slope={tm.warmup_verify_slope.tolist()} "
                f"t_verify_runtime={tm.runtime_verify_unit.tolist()} "
                f"batch_probs={self.current_batch_probs.tolist() if self.current_batch_probs is not None else None} "
                f"selected_chain={cascade_list['chain']} "
                f"dp_details={cascade_list['dp_details']}"
            )

        return cascade_list, cascade_list['chain'], cascade_list['dp_details']

    def _maybe_apply_model_chain(
        self, chain: Sequence[str], *, update_scheduler: bool = True
    ) -> None:
        if not chain:
            return
        if any(model_id not in self.worker_map for model_id in chain):
            return
        if tuple(chain) == self._chain_last_applied:
            return

        ordered_workers = [self.worker_map[model_id] for model_id in chain]
        self.model_workers = ordered_workers
        # self._ensure_prob_storage_matches_models()
        # router_chain = [
        #     (worker.model_name, worker.speculative_algorithm)
        #     for worker in self.model_workers
        # ]
        self.model_router.set_router_chain(chain)
        self.model_router.update_capture_hidden_mode_table()
        # TODO：_refresh_capture_modes看看啥时候需要refresh。
        # self._refresh_capture_modes(chain)
        if update_scheduler and self.chain_scheduler:
            self.chain_scheduler.update_model_chain(chain)
        if self._chain_last_applied and self._chain_last_applied != tuple(chain):
            self._chain_switch_count += 1
        self._chain_last_applied = tuple(chain)
        self._record_applied_chain(self._chain_last_applied)

    # def _refresh_capture_modes(self, chain: Sequence[str]) -> None:
    #     if not chain:
    #         return
    #     ordered_server_args = [
    #         self.server_args_map.get(model_id) for model_id in chain
    #     ]
    #     ordered_server_args = [sa for sa in ordered_server_args if sa is not None]
    #     if ordered_server_args:
    #         self.model_router.init_model_capture_hidden_mode(ordered_server_args)
    def init_speculative_args(self, num_steps: int, topk: int, num_draft_tokens: int):
        self.speculative_num_steps = num_steps
        self.speculative_eagle_topk = topk
        self.speculative_num_draft_tokens = num_draft_tokens
        self.decode_mem_cache_buf_multiplier = (
            1
            # if self.spec_algorithm.is_none()
            if not self.spec_flag
            else (
                self.speculative_num_draft_tokens
                + (
                    self.speculative_eagle_topk
                    * self.speculative_num_steps
                )
            )
        )

    def update_speculative_args(self, num_steps: int, topk: int, num_draft_tokens: int):
        if num_steps == self.speculative_num_steps and topk == self.speculative_eagle_topk and num_draft_tokens == self.speculative_num_draft_tokens:
            return
        self.init_speculative_args(num_steps, topk, num_draft_tokens)

    def get_worker_info(self):
        (
            max_total_num_tokens,
            max_prefill_tokens,
            max_running_requests,
            max_req_len,
            max_req_input_len,
            random_seed,
            device,
            global_server_args_dict,
            req_to_token_pool_size,
            req_to_token_pool_max_context_len,
            token_to_kv_pool_size,
        ) = self.model_workers[-1].get_worker_info()

        # 返回实际分配的max_running_requests（用于内存分配）
        # 但调度器会通过max_micro_batch_size来控制实际并发
        return (
            max_total_num_tokens,
            max_prefill_tokens,
            max_running_requests,  # 这是内存分配的上限
            max_req_len,
            max_req_input_len,
            random_seed,
            device,
            global_server_args_dict,
            req_to_token_pool_size,
            req_to_token_pool_max_context_len,
            token_to_kv_pool_size,
            self.enable_overlap,
            self.spec_flag,
        )

    def get_effective_max_running_requests(self):
        """获取当前模式下的有效最大并发请求数"""
        return self.effective_max_running_requests
    
    def get_worker(self, worker_id: int):
        return self.model_workers[worker_id]

    def memory_pool_clear(self):
        for worker in self.model_workers:
            worker.model_runner.req_to_token_pool.clear()
            worker.model_runner.token_to_kv_pool_allocator.clear()
    
    def get_worker_by_id(self, worker_id: int):
        if worker_id == -1:
            return self.model_workers[-1]
        else:
            return self.model_workers[worker_id]
    
    def extract_first_output_tokens(self, batch: ScheduleBatch):
        seq_lens = batch.seq_lens
        extend_lens = batch.extend_lens
        output_ids = batch.output_ids
        batch_size = len(seq_lens)
        # logger.info(f"extract_first_output_tokens, seq_lens: {seq_lens}, extend_lens: {extend_lens}, output_ids: {output_ids}")
        # logger.info(f"extract_first_output_tokens, batchsize: {len(seq_lens)}, extend_lens: {len(extend_lens)}, output_ids: {len(output_ids)}")

        if len(output_ids) > batch_size:
            # 说明之前是投机推理
            padded_extend = [0] + extend_lens + [1] * (batch_size - len(extend_lens))
            start_indices = torch.cumsum(torch.tensor(padded_extend[:-1], device=output_ids.device), dim=0)
            batch.output_ids = output_ids[start_indices]
    
    def post_spec_restart(self, batch: ScheduleBatch):
        self.restart_done = False
        batch.spec_flag = True
        # self.decode_mem_cache_buf_multiplier = (
        #     self.speculative_num_draft_tokens
        #     + (
        #         self.speculative_eagle_topk
        #         * self.speculative_num_steps
        #     )
        # )
        # assert False
    
    def get_current_chain_strategy(self):
        if self.lazy_switch_strategy is not None:
            return self.lazy_switch_strategy
        return self.current_chain_strategy

    @staticmethod
    def _chain_key(chain: Optional[Sequence[str]]) -> str:
        if not chain:
            return "EMPTY"
        return " -> ".join(chain)

    def _record_selected_chain(self, chain: Optional[Sequence[str]]) -> None:
        if not chain:
            return
        chain_tuple = tuple(chain)
        self._last_selected_chain = chain_tuple
        self._chain_selected_counts[self._chain_key(chain_tuple)] += 1

    def _record_applied_chain(self, chain: Optional[Sequence[str]]) -> None:
        if not chain:
            return
        self._chain_applied_counts[self._chain_key(tuple(chain))] += 1

    def _record_observed_chain(self, chain: Optional[Sequence[str]]) -> None:
        if not chain:
            return
        self._chain_observed_counts[self._chain_key(tuple(chain))] += 1

    def get_runtime_stats(self) -> Dict[str, object]:
        current_strategy = self.get_current_chain_strategy()
        return {
            "current_chain_ids": list(current_strategy.current_chain_ids),
            "current_mode": current_strategy.mode.value,
            "process_lazy_switch": self.process_lazy_switch,
            "lazy_chain_ids": (
                list(self.lazy_switch_strategy.current_chain_ids)
                if self.lazy_switch_strategy is not None
                else None
            ),
            "last_selected_chain": (
                list(self._last_selected_chain)
                if self._last_selected_chain is not None
                else None
            ),
            "search_count": self._chain_search_count,
            "switch_count": self._chain_switch_count,
            "lazy_switch_count": self._chain_lazy_switch_count,
            "selected_chain_counts": dict(sorted(self._chain_selected_counts.items())),
            "applied_chain_counts": dict(sorted(self._chain_applied_counts.items())),
            "observed_chain_counts": dict(sorted(self._chain_observed_counts.items())),
        }
    
    
    def check_switch_mode(self, batch: ScheduleBatch):
        # [EAGLE3_DEBUG] unconditional entry trace
        if get_rank() == 0:
            logger.info(f"[EAGLE3_DEBUG] check_switch_mode ENTRY | chain_scheduler={self.chain_scheduler is not None} first_prefill={self.first_prefill_pass} flag_file={os.path.exists('/tmp/EAGLE3_DEBUG_CHAIN')}")
        # if get_rank() == 0:
        #     logger.info(f"check_switch_mode, ")
        self._record_observed_chain(self.get_current_chain_strategy().current_chain_ids)
        if self.chain_scheduler is None:
            self.current_batch_probs = None
            return

        self._attach_batch_probs(batch)
        # global_similarity_matrix = self.chain_scheduler.global_similarity_matrix
        # self.chain_scheduler.global_similarity_matrix = 0.9 * global_similarity_matrix + 0.1 * self.current_batch_probs
        # if self.chain_scheduler:
        #     # source ar
        #     self.chain_scheduler.global_similarity_matrix = self.current_batch_probs

        # start_time = time.perf_counter()
        # logger.info(f"check_switch_mode, batch.spec_info:{batch.spec_info.accept_length.mean() if batch.spec_info.accept_length is not None else 1}")
        if batch.spec_info is not None and batch.spec_info.accept_length is not None:
            min_accept_length = math.ceil(batch.spec_info.accept_length.float().mean()+1)
            min_accept_length = min(min_accept_length, self.max_speculative_num_draft_tokens)
        else:
            min_accept_length = -1
        # min_accept_length = -1
        # with count_time("search_chain"):
        objects_to_sync = [None, None, None]
        self._chain_search_count += 1
        with count_time("search_chain"):
            if get_rank() == 0:
                if self.first_prefill_pass:
                    cascade_list, selected_chain, dp_details = self._search_chain(min_accept_length)
                    objects_to_sync = [cascade_list, selected_chain, dp_details]
        dist.broadcast_object_list(objects_to_sync, src=0)
        cascade_list, selected_chain, dp_details = objects_to_sync
        specrouter_debug_log(
            logger,
            "check_switch_mode_result",
            min_accept_length=min_accept_length,
            selected_chain=selected_chain,
            process_lazy_switch=self.process_lazy_switch,
            current_strategy=describe_strategy(self.current_chain_strategy),
            lazy_strategy=describe_strategy(self.lazy_switch_strategy),
            batch=describe_batch(batch),
        )
        # logger.info(f"cascade_list: {cascade_list}")
        # logger.info(f"cur: {self._current_chain_ids()} ")

        # [EAGLE3_DEBUG] force full chain override for control experiment
        if selected_chain is not None and os.path.exists("/tmp/EAGLE3_FORCE_FULL_CHAIN"):
            original_chain = selected_chain
            selected_chain = self.chain_scheduler.full_model_ids[:]
            if os.path.exists("/tmp/EAGLE3_DEBUG_CHAIN") and get_rank() == 0:
                logger.info(
                    f"[EAGLE3_DEBUG] FORCE_FULL_CHAIN | "
                    f"original={original_chain} forced={selected_chain}"
                )

        # exploration: periodically force full chain when solver chose shorter chain
        if selected_chain is not None and self.chain_scheduler is not None:
            full_ids = self.chain_scheduler.full_model_ids
            if self._explore_remaining > 0:
                self._explore_remaining -= 1
                selected_chain = full_ids[:]
            elif len(selected_chain) < len(full_ids):
                self._explore_count += 1
                if self._explore_count >= self._explore_interval:
                    self._explore_count = 0
                    self._explore_remaining = self._explore_burst - 1
                    selected_chain = full_ids[:]
                    if get_rank() == 0:
                        logger.info(
                            f"[EAGLE3_DEBUG] exploration triggered | "
                            f"forcing full chain for {self._explore_burst} steps"
                        )

        if selected_chain is not None:
            self._record_selected_chain(selected_chain)

            # [EAGLE3_DEBUG] check_switch_mode decision log
            if os.path.exists("/tmp/EAGLE3_DEBUG_CHAIN") and get_rank() == 0:
                current_ids = self._current_chain_ids()
                logger.info(
                    f"[EAGLE3_DEBUG] check_switch_mode | "
                    f"search_count={self._chain_search_count} "
                    f"current_chain={current_ids} "
                    f"selected_chain={selected_chain} "
                    f"same_as_current={current_ids == list(selected_chain)} "
                    f"batch_size={batch.batch_size()}"
                )

            # if len(selected_chain) > 1:
            #     self.update_spec_config_from_dp(dp_details)
            # self.switch_thresh = 8

            # if batch.batch_size() < self.switch_thresh:
            #     # selected_chain = [self.full_model_ids[1], self.full_model_ids[-1]]
            #     # selected_chain = [self.full_model_ids[0], self.full_model_ids[1], self.full_model_ids[-1]]
            #     selected_chain = [ self.full_model_ids[0], self.full_model_ids[-1]]
            # else:
            #     # selected_chain = [ self.full_model_ids[0], self.full_model_ids[-1]]
            #     selected_chain = [ self.full_model_ids[-1]]
            #     # selected_chain = [ self.full_model_ids[0], self.full_model_ids[-1]]
            
            # kkkk_thresh = 10
            # uuu_thresh = 12
            # if batch.batch_size() > uuu_thresh:
            #     # selected_chain = [self.full_model_ids[1], self.full_model_ids[-1]]
            #     selected_chain = self.chain_scheduler.full_model_ids
            #     # selected_chain = [self.full_model_ids[0], self.full_model_ids[1], self.full_model_ids[-1]]
            # elif batch.batch_size() > kkkk_thresh:
            #     selected_chain = [self.full_model_ids[-1]]
            #     # selected_chain = [self.full_model_ids[0], self.full_model_ids[1], self.full_model_ids[-1]]
            # else:
            #     # selected_chain = [self.full_model_ids[1], self.full_model_ids[-1]]
            #     # selected_chain = [ self.full_model_ids[1], self.full_model_ids[-1]]
            #     # selected_chain = [ self.full_model_ids[0], self.full_model_ids[-1]]
            #     selected_chain = self.chain_scheduler.full_model_ids
            #     # selected_chain = [self.full_model_ids[0], self.full_model_ids[1], self.full_model_ids[-1]]
            # # jingtai
            # selected_chain = self.full_model_ids[:]
            # selected_chain = [self.full_model_ids[0], self.full_model_ids[1], self.full_model_ids[-1]]
            # wechat
            # selected_chain = [self.full_model_ids[0],  self.full_model_ids[-1]]
            # self._maybe_apply_model_chain(selected_chain)
            # selected_chain = [self.full_model_ids[-1]]
            # selected_chain = [self.full_model_ids[1]]
            new_strategy, diff = self.get_current_chain_strategy().set_next_strategy(selected_chain)
            # self.current_chain_strategy.print_strategy()
            if not diff.non_diff or self.process_lazy_switch:
                # if diff.switch_mode:
                #     # self.current_chain_strategy = new_strategy
                #     self.current_chain_strategy.print_strategy()
                #     new_strategy.print_strategy()
                # end_time = time.perf_counter()
                # logger.info(f"self._maybe_update_chain, time: {end_time - start_time}")
                self.lazy_switch(new_strategy, diff)

        if self.restart_done:
            self.post_spec_restart(batch)

        # if get_rank() == 0:
        #     # print("\n")
        #     # logger.info(f"self.current_batch_probs: {self.current_batch_probs}")
        #     # logger.info(f"self.global_time_dict: {self.chain_scheduler.global_time_dict}, {self.chain_scheduler.decode_time_dict}")
        #     spec_al = "speculative" if len(selected_chain) > 1 else "autoregressive"
        #     logger.info(f"self.cascade_list: {cascade_list}, {spec_al}")
        #     # logger.info(f"self.cascade_list: {probs_value[0][1]}, {count_value}, {spec_al}")
        #     # logger.info(f"should: {spec_al}")
        #     # logger.info(f"self.selected_chain: {selected_chain}")
        #     # logger.info(f"self.req_probs_count: {self.req_probs_count}")
        # we set mode
        # switch_thresh = 1000
        # self.switch_thresh = 1

        # if batch.batch_size() < 5:
        #     draft_worker = self.model_workers[0]
        #     target_worker = self.model_workers[-1]
        #     logger.info(f"batch.batch_size(): {batch.batch_size()}")
        #     draft_worker.update_speculative_args(3,1,8)
        #     target_worker.update_speculative_args(6,1,4)
        # new_chain_strategy = ChainStrategy(
        #     name="new_chain_strategy",
        #     mode=InferenceMode.SPECULATIVE,
        #     current_chain_ids=selected_chain,
        #     worker_params={},
        #     memory_multiplier=1.0,
        # )
        # new_strategy = ChainStrategy.create(
        #     name="new_strategy",
        #     mode=InferenceMode.SPECULATIVE,
        #     current_chain_ids=selected_chain,
        #     worker_params={},
        #     memory_multiplier=1.0,
        # )
        
        # self._maybe_apply_model_chain(selected_chain)
        # self.switch_thresh = 8
        
        # if batch.batch_size() < self.switch_thresh:
        # if batch.batch_size() < self.switch_thresh:
            # new_ids = [self.full_model_ids[0], self.full_model_ids[1]]
            # self.lazy_reload(new_ids)
        
        # # if (len(selected_chain) > 1):
        #     # self._maybe_apply_model_chain([self.full_model_ids[0], self.full_model_ids[1]])
        #     self.set_inference_mode("speculative", batch)
        # else:
        #     # self._maybe_apply_model_chain([self.full_model_ids[1]])
        #     self.set_inference_mode("autoregressive", batch)

    def lazy_switch(self, new_strategy: ChainStrategy, diff: ChainStrategyDiff):
        specrouter_debug_log(
            logger,
            "lazy_switch_enter",
            process_lazy_switch=self.process_lazy_switch,
            current_strategy=describe_strategy(self.current_chain_strategy),
            lazy_strategy=describe_strategy(self.lazy_switch_strategy),
            new_strategy=describe_strategy(new_strategy),
            diff=describe_chain_diff(diff),
        )
        if self.process_lazy_switch:
            self.current_chain_strategy = self.lazy_switch_strategy
            self.lazy_switch_strategy = None
            self.lazy_switch_diff = None
            self.process_lazy_switch = False

        if not diff.non_diff:
            self._chain_lazy_switch_count += 1
            self.lazy_switch_strategy = new_strategy
            self.lazy_switch_diff = diff
            self.process_lazy_switch = True
        specrouter_debug_log(
            logger,
            "lazy_switch_exit",
            process_lazy_switch=self.process_lazy_switch,
            current_strategy=describe_strategy(self.current_chain_strategy),
            lazy_strategy=describe_strategy(self.lazy_switch_strategy),
            diff=describe_chain_diff(self.lazy_switch_diff),
        )
        
    def update_spec_config_from_dp(self, dp_details: List[Dict]):
        max_num_steps, max_eagle_topk, max_num_draft_tokens = 0, 0, 0
        for dp in dp_details:
            worker = self.worker_map[dp['model']]
            num_steps, eagle_topk, num_draft_tokens = dp['spec_config']
            worker.update_speculative_args(num_steps, eagle_topk, num_draft_tokens)
            max_num_steps = max(max_num_steps, num_steps)
            max_eagle_topk = max(max_eagle_topk, eagle_topk)
            max_num_draft_tokens = max(max_num_draft_tokens, num_draft_tokens)
            self.state_manager.update_spec_params(worker.model_name, num_steps, eagle_topk, num_draft_tokens)
        self.update_speculative_args(max_num_steps, max_eagle_topk, max_num_draft_tokens)
        self.state_manager.update_spec_params('base_params', max_num_steps, max_eagle_topk, max_num_draft_tokens)

    # def set_inference_mode(self, mode: str, batch: ScheduleBatch):
    #     """动态切换推理模式，支持'speculative'和'autoregressive'

    #     Args:
    #         mode: 推理模式 ('speculative' 或 'autoregressive')
    #         scheduler_ref: 调度器引用，用于动态调整max_micro_batch_size

    #     Returns:
    #         bool: 切换是否成功
    #     """
    #     if mode not in ["speculative", "autoregressive"]:
    #         raise ValueError(f"Unsupported inference mode: {mode}")
        
    #     can_skip = (self.current_mode == mode) and ((not batch.spec_flag and  (mode == "autoregressive")) or (batch.spec_flag and (mode == "speculative")))

    #     if can_skip:
    #         # logger.info("stop,skip")
    #         return True
    #     # else:
    #     #     logger.info("go through")

    #     old_mode = self.current_mode
    #     self.current_mode = mode

    #     if (old_mode == "speculative" or batch.spec_flag) and mode == "autoregressive":
    #         # # aux autoregressive
    #         # dkv_config = (1,1,2)
    #         # # dkv_config = (5,1,6)
    #         # self.update_speculative_args(*dkv_config)
    #         # for worker in self.model_workers:
    #         #     worker.update_speculative_args(*dkv_config)
    #         batch.spec_flag = False
    #         batch.spec_info = None
    #         self.extract_first_output_tokens(batch)
    #         self.decode_mem_cache_buf_multiplier = 1

    #     elif (old_mode == "autoregressive" or not batch.spec_flag) and mode == "speculative":
    #         # batch.spec_flag = True
    #         specParams = self.state_manager.get_spec_params('base_params')
    #         self.update_speculative_args(specParams.num_steps, 
    #                                         specParams.eagle_topk, 
    #                                         specParams.num_draft_tokens)
    #         for worker in self.model_workers:
    #             specParams = self.state_manager.get_spec_params(worker.model_name)
    #             worker.update_speculative_args(specParams.num_steps,
    #                                             specParams.eagle_topk,
    #                                             specParams.num_draft_tokens)
    #     if get_rank() == 0:     
    #         logger.info(f"Successfully switched inference mode from {old_mode} to {mode}, ")
    #         # logger.info(f"model_hub: speculative_num_steps:{self.speculative_num_steps}, speculative_eagle_topk:{self.speculative_eagle_topk}, speculative_num_draft_tokens:{self.speculative_num_draft_tokens}")
    #     return True
    
    def print_forward_batch_info(self, batch: ScheduleBatch, mode: str):
        if get_rank() == 0:
            if batch.out_cache_loc is not None:
                logger.info(f"forward_{mode}, input_ids:{batch.input_ids.shape}, batchsize:{batch.batch_size()}, out_cache_loc:{len(batch.out_cache_loc)}")
            else:
                logger.info(f"forward_{mode}, input_ids:{batch.input_ids.shape}, batchsize:{batch.batch_size()}, out_cache_loc:None")
        
    def forward_batch(self, batch: ScheduleBatch) -> Tuple[LogitsProcessorOutput, List[int], int, int, bool]:
        """统一的推理入口，根据当前模式选择执行路径
        Args:
            batch: 要处理的批次
            
        Returns:
            统一的返回格式: (logits_output, next_token_ids, batch_id, accepted_tokens, can_run_cuda_graph)
        """
        batch.runtime_step_telemetry = None
        # if self.current_mode == "speculative":
        #     return self.forward_batch_multi_speculative_generation_hfrouter(batch)
        # # elif self.current_mode == "autoregressive":
        # else:
        #     return self.forward_batch_autoregressive_generation(batch)
        return self.forward_batch_multi_speculative_generation_hfrouter(batch)
    
    def forward_batch_autoregressive_generation(self, batch: ScheduleBatch) -> Tuple[LogitsProcessorOutput, List[int], int, int, bool]:
        """Run autoregressive decoding forward.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        # target_worker = self.target_worker
        target_worker = self.model_workers[-1]
        model_worker_batch = batch.get_model_worker_batch()
        req_bs = int(model_worker_batch.req_pool_indices.numel())
        token_bs = int(model_worker_batch.input_ids.shape[0])
        batch.runtime_step_telemetry = {
            "target_step_kind": "target_autoregressive",
            "target_step_req_bs": req_bs,
            "target_step_token_bs": token_bs,
            "accepted_tokens_per_target_step": req_bs,
            "verify_tokens_per_emitted_token": (
                float(token_bs) / float(req_bs) if req_bs > 0 else 0.0
            ),
        }
        
        # 执行自回归推理
        logits_output, next_token_ids, can_run_cuda_graph = (
            # autoregressive
            self._profile_call(
                "draft",
                1, # forward_count
                1.0, # verify_K
                target_worker,
                target_worker.forward_batch_generation,
                model_worker_batch,
            )
        )
        # self._update_chain_time_stats()

        # 返回与投机推理相同格式的结果
        return logits_output, next_token_ids, model_worker_batch.bid, 0, can_run_cuda_graph

    
    def adaptive_spec_forward(self, batch: ScheduleBatch, next_state: str, exec_worker_name: str, last_worker_name:str, sync_model_name_list: List[str]):
        # logger.info(f"adaptive_spec_forward, next_state: {next_state}, last_worker_name: {last_worker_name}, exec_worker_name: {exec_worker_name}")
        if get_rank() == 0:
            logger.info(f"current {next_state}")
        specrouter_debug_log(
            logger,
            "adaptive_spec_forward_enter",
            next_state=next_state,
            exec_worker_name=exec_worker_name,
            last_worker_name=last_worker_name,
            sync_model_name_list=sync_model_name_list,
            submit_flag=self.submit_flag,
            process_lazy_switch=self.process_lazy_switch,
            current_chain_ids=self._current_chain_ids(),
            batch=describe_batch(batch),
        )
        bs = float(batch.batch_size())
        # if exec_worker_name == self.model_router.target_model_name and next_state in ["verify", "autoregressive"]:
        #     logger.info(f"adaptive_spec_forward, {next_state}, batch.seq_lens: {batch.seq_lens}, batch.req_pool_indices: {batch.req_pool_indices}")

        if next_state == "draft":
            draft_worker = self.worker_map[exec_worker_name]
            with draft_worker.draft_tp_context(draft_worker.model_runner.tp_group):
                    # name, forward_count, worker, func, *args, **kwargs
                    verify_input_spec_info = self._profile_call(
                        "draft", draft_worker.speculative_num_steps-1, 1.0, # forward_count, verify_K
                        draft_worker, draft_worker.draft, batch, draft_worker.speculative_num_steps+1
                    )
            # return verify_input_spec_info
            # logger.info(f"adaptive_spec_forward, draft, verify_input_spec_info: {vars(verify_input_spec_info)}")
            self.verify_input_spec_info = verify_input_spec_info
            specrouter_debug_log(
                logger,
                "draft_state_ready_for_verify",
                exec_worker_name=exec_worker_name,
                verify_input_spec_info=describe_spec_info(self.verify_input_spec_info),
            )

        elif next_state == "verify":
            # self._maybe_save_states(exec_worker_name, last_worker_name)
            target_worker = self.worker_map[exec_worker_name]
            # logger.info(f"last_worker_name: {last_worker_name}, self.submit_flag: {self.submit_flag}")
            # torch.distributed.barrier()
            # if get_rank() == 0:
            with count_time("snapshot_load"):
                self._maybe_load_state(batch, exec_worker_name, last_worker_name, target_worker.speculative_num_draft_tokens)
            # torch.distributed.barrier()
            # if exec_worker_name == self.model_router.target_model_name:
            #     logger.info(f"before , verify, verify_output:{batch.seq_lens},")
            # if exec_worker_name == self.model_router.target_model_name:
            #     with count_time("verify"):
            #         logits_output, verify_output, model_worker_batch_bid, can_run_cuda_graph = (
            #             self._profile_call(
            #                 "verify",
            #                 1, # forward_count
            #                 float(target_worker.speculative_num_draft_tokens)*bs, # verify_K
            #                 target_worker,
            #                 target_worker.verify,
            #                 batch,
            #                 self.verify_input_spec_info,
            #             )
            #         )
            # else:
            # with count_time("verify"):
            logits_output, verify_output, model_worker_batch_bid, can_run_cuda_graph = (
                self._profile_call(
                    "verify",
                    1, # forward_count
                    float(target_worker.speculative_num_draft_tokens)*bs, # verify_K
                    target_worker,
                    target_worker.verify,
                    batch,
                    self.verify_input_spec_info,
                )
            )
            self.chain_profiler.record("verify_bs_record", bs, model_id=target_worker.model_name, ignore_level=True)
                
            # if exec_worker_name == self.model_router.target_model_name:
            #     # logger.info(f"adaptive_spec_forward, verify, verify_output: {self.model_router.target_model_name}.")
            #     logger.info(f"adaptive_spec_forward, verify, input_ids: {batch.input_ids}, \
            #     seq_lens: {batch.seq_lens}, \
            #     verified_id: {batch.spec_info.verified_id}, \
            #     accept_length_per_req_cpu: {verify_output.accept_length_per_req_cpu}, \
            #     req_pool_indices: {batch.req_pool_indices}.")
            # logger.info(f"adaptive_spec_forward, verify, verify_output: {vars(verify_output)}")
            self._update_req_probs_from_verify(
                batch, verify_output.probs_of_draft, verify_output.safe_accept_length, last_worker_name, exec_worker_name
            )
            # if get_rank() == 0:
            self._maybe_save_state(batch, verify_output, exec_worker_name, last_worker_name, batch.req_pool_indices, target_worker.speculative_num_steps, target_worker.topk)
            # torch.distributed.barrier()
            # logger.info(f"last_worker_name: {last_worker_name}, self.submit_flag: {self.submit_flag}")
            # self.check_submit(verify_output, exec_worker_name)
            next_token_ids = verify_output.verified_id
            num_accepted_tokens = sum(verify_output.accept_length_per_req_cpu)

            # [EAGLE3_DEBUG] per-joint accept tracking for three-chain diagnosis
            if os.environ.get("EAGLE3_DEBUG_VERIFY", "0") == "1":
                is_target = (exec_worker_name == self.model_router.target_model_name)
                logger.info(
                    f"[EAGLE3_DEBUG] verify joint | verifier={exec_worker_name} "
                    f"| is_final_target={is_target} "
                    f"| drafter={last_worker_name} "
                    f"| batch_size={batch.batch_size()} "
                    f"| accept_per_req={verify_output.accept_length_per_req_cpu} "
                    f"| total_accepted={num_accepted_tokens} "
                    f"| mean_accept={num_accepted_tokens/max(batch.batch_size(),1):.2f}"
                )
            specrouter_debug_log(
                logger,
                "verify_state_complete",
                exec_worker_name=exec_worker_name,
                last_worker_name=last_worker_name,
                submit_flag=self.submit_flag,
                accepted_indices_len=len(verify_output.accepted_indices),
                finished_req_pool_indices_len=len(verify_output.finished_req_pool_indices),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                verify_output_accept_length=(
                    verify_output.accept_length.tolist()
                    if verify_output.accept_length is not None
                    else None
                ),
                batch=describe_batch(batch),
            )

            if self.first_prefill_pass == False:
                if target_worker.model_name == self.model_router.target_model_name and len(verify_output.finished_req_pool_indices) > 0:
                    self.first_prefill_pass = True

            return logits_output, next_token_ids, model_worker_batch_bid, num_accepted_tokens, can_run_cuda_graph

        elif next_state == "draft_extend":
            # 收取目标模型的状态last_worker_name
            # batch = self.state_manager.restore_snapshot(last_worker_name, batch)
            with count_time("target_extend"):
                sync_model_name_list = sync_model_name_list[::-1]
                for sync_model_name in sync_model_name_list[:-1]:
                    mid_target_worker = self.worker_map[sync_model_name]
                    batch.spec_info.capture_hidden_mode = self.model_router.get_model_capture_hidden_mode_from_table(mid_target_worker.model_name, "full")
                    if batch.spec_info.drafted_id is not None:
                        # start_time = time.perf_counter()
                        model_history = self.state_manager._model_history.get(mid_target_worker.model_name)
                        reload_flag = model_history is not None and model_history.special_for_draft_extend
                        reload_model_history = None
                        # with count_time(f"draft_extend_{mid_target_worker.model_name} reload_flag:{reload_flag}"):
                        if get_rank() == 0:
                            logger.info(f"draft_extend, reload_flag: {reload_flag}")
                        if reload_flag:
                            reload_model_history = model_history
                            batch.spec_info.capture_hidden_mode = self.model_router.get_model_capture_hidden_mode_from_table(mid_target_worker.model_name, "last")
                        with mid_target_worker.draft_tp_context(mid_target_worker.model_runner.tp_group):
                            new_verified_id, logits_output, next_token_ids, _ = self._profile_call(
                                "draft_extend", 1, bs, # forward_count, verify_K
                                # mid_target_worker, mid_target_worker.forward_target_extend_after_decode, batch
                                mid_target_worker, mid_target_worker.forward_target_extend_after_decode_src, batch, reload_model_history
                            )
                        # self.state_manager.append_submit_inputs(sync_model_name, [new_verified_id, next_token_ids], batch.spec_info.req_pool_indices_for_draft_extend)
                        self.state_manager.fill_submit_inputs(sync_model_name, new_verified_id, batch.spec_info.req_pool_indices_for_draft_extend)
                        # self.state_manager.append_submit_inputs(sync_model_name, [new_verified_id], batch.spec_info.req_pool_indices_for_draft_extend)
                        # logger.info(f"over, end. {batch.seq_lens}")
                        # logger.info(f"\n")
                        # end_time = time.perf_counter()
                        # logger.info(f"draft_extend time: {end_time - start_time}")
            with count_time("draft_extend"):
                    # self._maybe_save_state(batch, verify_output, exec_worker_name, last_worker_name, batch.req_pool_indices, mid_target_worker.speculative_num_steps, mid_target_worker.topk)
                # 执行草稿模型的extend
                if len(sync_model_name_list) > 0:
                    draft_worker = self.worker_map[sync_model_name_list[-1]] 
                    # logger.info(f"draft_extend, draft_worker: {draft_worker.model_name}, {batch.spec_info}")
                    if batch.spec_info.verified_id is not None:
                        model_history = self.state_manager._model_history.get(draft_worker.model_name)
                        # logger.info(f"draft_extend, reload_flag model_history: {draft_worker.model_name}, {model_history.special_for_draft_extend}")
                        reload_flag = model_history is not None and model_history.special_for_draft_extend
                        reload_model_history = model_history if reload_flag else None
                        if get_rank() == 0:
                            logger.info(f"draft_extend, reload_flag: {reload_flag}")

                        batch.spec_info.capture_hidden_mode = self.model_router.get_model_capture_hidden_mode_from_table(draft_worker.model_name, "last")
                        # logger.info(f"draft_extend, batch.spec_info.capture_hidden_mode: {draft_worker.model_name}, hidden_states: {batch.spec_info.hidden_states}")
                        # with count_time(f"draft_extend_{draft_worker.model_name} reload_flag:{reload_flag}"):
                        with draft_worker.draft_tp_context(draft_worker.model_runner.tp_group):
                            self._profile_call(
                                "draft_extend", 1, bs, # forward_count, verify_K
                                draft_worker, draft_worker.forward_draft_extend_after_decode, batch, reload_model_history
                            )
                    specrouter_debug_log(
                        logger,
                        "draft_extend_state_complete",
                        exec_worker_name=exec_worker_name,
                        draft_worker_name=draft_worker.model_name,
                        sync_model_name_list=sync_model_name_list,
                        batch=describe_batch(batch),
                    )
        
        elif next_state == "autoregressive":
            target_worker = self.worker_map[exec_worker_name]
            # logger.info(f"Rank {get_rank()}: ")
            # logger.info(f"batch.input_ids: {batch.input_ids}, batch.output_ids: {batch.output_ids}")
            # logger.info(f"Batch Size = {batch.batch_size()}, Input IDs Shape = {batch.input_ids.shape} Extend Seq Lens = {batch.extend_lens}")
            # 执行自回归推理
            # hidden_states_pool = self.model_hidden_states_pools[exec_worker_name]
            # hidden_states_pool = None
            # if batch.spec_info is None and hidden_states_pool is not None:
            #     batch.spec_info = EagleDraftInput(
            #         capture_hidden_mode=self.model_router.get_capture_hidden_mode_direct("last")
            #     )
            #     model_worker_batch = batch.get_model_worker_batch()

            #     logits_output, next_token_ids, can_run_cuda_graph = (
            #         # autoregressive
            #         self._profile_call(
            #             "draft",
            #             1, # forward_count
            #             1.0, # verify_K
            #             target_worker,
            #             target_worker.forward_batch_generation,
            #             model_worker_batch,
            #         )
            #     )
            #     hidden_states_pool.write_batch(logits_output.hidden_states, batch.out_cache_loc)
            #     batch.spec_info = None
            #     logits_output.hidden_states = None
            # else:
            model_worker_batch = batch.get_model_worker_batch()
            logger.info("hereh in")
            logits_output, next_token_ids, can_run_cuda_graph = (
                # autoregressive
                self._profile_call(
                    "draft",
                    1, # forward_count
                    1.0, # verify_K
                    target_worker,
                    target_worker.forward_batch_generation,
                    model_worker_batch,
                )
            )
            logger.info("hereh out")
            # # logger.info(f"batch.spec_info: {batch.spec_info}")
            if batch.spec_info is not None:
            #     # logger.info(f"rebuild_spec_info")
                logger.info("jkkakdkaksdkfa")
                # draft model come back
                self.rebuild_spec_info(batch, logits_output, next_token_ids)
                # if hidden_states_pool is not None:
                #     hidden_states_pool.write_batch(batch.spec_info.hidden_states, batch.out_cache_loc)

            return logits_output, next_token_ids, model_worker_batch.bid, 0, can_run_cuda_graph

        elif next_state == "unload":
            if get_rank() == 0:
                logger.info(f"unload reload_models:{self.lazy_switch_diff.reload_models}, unload_models: {self.lazy_switch_diff.unload_models}")
            draft_model = self.lazy_switch_strategy.current_chain_ids[0]
            exec_unload = False
            


            if self.lazy_switch_strategy.mode == InferenceMode.AUTOREGRESSIVE:

                exec_unload = True
                self.model_router.update_capture_hidden_mode_table(self.lazy_switch_strategy.current_chain_ids)
                batch.spec_flag = False
                batch.out_cache_loc = batch.spec_info.next_out_cache_loc
                # batch.req_to_token_pool.write(
                #     (batch.req_pool_indices, batch.seq_lens.clone()), batch.out_cache_loc.to(torch.int32)
                # )

                batch.input_ids = batch.spec_info.verified_id
                batch.output_ids = None
                # self.extract_first_output_tokens(batch)
                self.decode_mem_cache_buf_multiplier = 1

                self.state_manager.save_model_history(self.lazy_switch_diff.unload_models, batch.req_pool_indices, batch.seq_lens, batch.spec_info.accept_length, stage_name="unload")

                # prepare for this AUTOREGRESSIVE
                batch.spec_info = None
                # logger.info(f"batch.spec_info: {batch.spec_info}")
                batch.seq_lens.add_(1)
                # self.model_router.update_capture_hidden_mode_table(self.lazy_switch_strategy.current_chain_ids)
                self._maybe_apply_model_chain(self.lazy_switch_strategy.current_chain_ids)
                self.process_lazy_switch = False
                
            # elif draft_model not in self.lazy_switch_diff.reload_models and draft_model not in self.lazy_switch_diff.unload_models:
            #     exec_unload = True
            #     # batch.spec_info = EagleDraftInput(
            #     #     # capture_hidden_mode=self.model_router.model_capture_hidden_mode[target_worker.model_name]
            #     #     capture_hidden_mode=self.model_router.get_model_capture_hidden_mode_from_table(target_worker.model_name, "last")
            #     # )
            #     pass
            elif draft_model in self.lazy_switch_diff.reload_models:
                
                self.model_router.update_capture_hidden_mode_table(self.lazy_switch_strategy.current_chain_ids)
                if batch.spec_info is not None:
                    batch.spec_info.capture_hidden_mode = self.model_router.get_model_capture_hidden_mode_from_table(exec_worker_name, "last")
                else:
                    batch.spec_info = EagleDraftInput(
                        capture_hidden_mode=self.model_router.get_model_capture_hidden_mode_from_table(exec_worker_name, "last")
                    )
                # 暂时这样写，看看能不能先把1-2过了。
            elif draft_model in self.lazy_switch_diff.unload_models:
                # 即将被下，这一部分需要lazy switch。
                pass

        elif next_state == "reconfig":
            logger.info(f"reconfig reload_models:{self.lazy_switch_diff.reload_models}, unload_models: {self.lazy_switch_diff.unload_models}")
            # if len(self.lazy_switch_diff.reload_models) > 0:
            #     self.model_reload(self.lazy_switch_diff.reload_models)

            # if len(self.lazy_switch_diff.unload_models) > 0:
            #     # 进行完draft extend就要离开。
            #     self.model_unload(self.lazy_switch_diff.unload_models) 
            # if len(self.lazy_switch_diff.reload_models) > 0:
            #     pass
            if len(self.lazy_switch_diff.reload_models) > 0:
                # logger.info(f"reconfig, reload_models: {self.lazy_switch_diff.reload_models}")
                last_reload_model = self.lazy_switch_diff.reload_models[0]
                draft_worker = self.worker_map[last_reload_model]
                if draft_worker.speculative_algorithm.is_eagle():
                    filtered_reload_models = self.lazy_switch_diff.reload_models[1:]
                else:
                    filtered_reload_models = self.lazy_switch_diff.reload_models
                # filtered_reload_models = self.lazy_switch_diff.reload_models
                current_seq_lens_for_target = batch.spec_info.seq_lens_for_draft_extend - batch.spec_info.accept_length_for_draft_extend - 1
                # logger.info(f"reconfig, current_seq_lens_for_target: {batch.spec_info.seq_lens_for_draft_extend}, current_seq_lens_for_target: {current_seq_lens_for_target}, accept_length: {batch.spec_info.accept_length_for_draft_extend}")
                # logger.info(f"batch.req_pool_indices: {batch.req_pool_indices}, batch.spec_info.req_pool_indices_for_draft_extend: {batch.spec_info.req_pool_indices_for_draft_extend}")
                self.state_manager.load_model_history(filtered_reload_models, batch, 
                                                    self.model_router.target_model_name, batch.spec_info.req_pool_indices_for_draft_extend, 
                                                    current_seq_lens_for_target, batch.req_pool_indices,
                                                    self.lazy_switch_strategy.current_chain_ids[0], 
                                                    batch.spec_info.verified_id,
                                                    batch.spec_info.drafted_id,
                                                    batch.spec_info.accept_length+1,
                                                    batch.out_cache_loc,
                                                    batch.spec_info.next_out_cache_loc,
                                                    # self.worker_map,
                                                    # self.model_hidden_states_pools,
                                                    )

                specParams = self.state_manager.get_spec_params(draft_worker.model_name)
                draft_worker.update_speculative_args(specParams.num_steps,
                                                specParams.eagle_topk,
                                                specParams.num_draft_tokens)

            if len(self.lazy_switch_diff.unload_models) > 0:
                # 本次推理已经结束，准备下掉所有模型。
                # Reconfig runs on the unfinished-request subset prepared for
                # draft_extend, so use the matching request/length tensors here.
                self.state_manager.save_model_history(
                    self.lazy_switch_diff.unload_models,
                    batch.spec_info.req_pool_indices_for_draft_extend,
                    batch.spec_info.seq_lens_for_draft_extend,
                    batch.spec_info.accept_length_for_draft_extend,
                    stage_name="reconfig",
                )


            # if self.current_chain_strategy.mode == InferenceMode.SPECULATIVE and self.lazy_switch_strategy.mode == InferenceMode.AUTOREGRESSIVE:
            # must speculative
            # self.model_router.update_capture_hidden_mode_table(self.lazy_switch_strategy.current_chain_ids)
            self._maybe_apply_model_chain(self.lazy_switch_strategy.current_chain_ids)
            
            # specParams = self.state_manager.get_spec_params('base_params')
            # self.update_speculative_args(specParams.num_steps, 
            #                                 specParams.eagle_topk, 
            #                                 specParams.num_draft_tokens)
            # for worker in self.model_workers:
                # specParams = self.state_manager.get_spec_params(worker.model_name)
                # worker.update_speculative_args(specParams.num_steps,
                #                                 specParams.eagle_topk,
                #                                 specParams.num_draft_tokens)
            self.restart_done = True
            self.process_lazy_switch = False

            
            self.decode_mem_cache_buf_multiplier = (
                self.speculative_num_draft_tokens
                + (
                    self.speculative_eagle_topk
                    * self.speculative_num_steps
                )
            )


                
        return None, None, None, None, None

    @nvtx_profile
    def _maybe_load_state(self, batch: ScheduleBatch, exec_worker_name: str, last_worker_name: str, num_verify_tokens: int):

        # if exec_worker_name  != self.model_router.target_model_name:
        #     # prepare for submit 
        #     # last_worker_state = self.state_manager.get_snapshot(last_worker_name)
        #     pass
        
        if exec_worker_name == self.model_router.target_model_name:
            is_top_verify = True
        else:
            is_top_verify = False
        if self.submit_flag:
            # assert False
            # logger.info(f"submit batch: {batch.reqs}")
            exec_worker_state = self.state_manager.get_snapshot(exec_worker_name)
            # draft_spec_info = self.state_manager.build_verify_input(batch, exec_worker_state, submit_inputs)
            # self.state_manager.restore_batch_state(batch, exec_worker_state, draft_spec_info)
            # logger.info(f"draft_spec_info: {vars(draft_spec_info)}")
            batch.reqs = exec_worker_state.reqs
            batch.req_pool_indices = exec_worker_state.req_pool_indices
            batch.seq_lens = exec_worker_state.seq_lens
            batch.seq_lens_sum = exec_worker_state.seq_lens_sum
            batch.sampling_info.temperatures = exec_worker_state.sampling_info_temperatures
            batch.sampling_info.top_ps = exec_worker_state.sampling_info_top_ps
            batch.sampling_info.top_ks = exec_worker_state.sampling_info_top_ks
            batch.sampling_info.min_ps = exec_worker_state.sampling_info_min_ps

            # logger.info(f"build_verify_input last_worker_name: {last_worker_name}, self.submit_flag: {self.submit_flag}")
            submit_inputs = self.state_manager.get_submit_inputs(last_worker_name)
            # TODO 暂时不支持skip extend
            # enabled_skip_extend = False
            draft_spec_info = self.state_manager.build_verify_input(exec_worker_state, submit_inputs, batch.req_pool_indices, num_verify_tokens, self.enabled_skip_extend, is_top_verify)
            # logger.info(f"build_verify_input before last_worker_name: {last_worker_name}, self.submit_flag: {self.submit_flag}")
            self.state_manager.clear_submit_inputs(last_worker_name, batch.req_pool_indices)
            if self.enabled_skip_extend:
                extra_out_loc_cache_num = submit_inputs.get_extra_out_loc_cache_num()
                extra_out_loc_cache = batch.token_to_kv_pool_allocator.alloc(extra_out_loc_cache_num)
                batch.out_cache_loc = submit_inputs.update_out_loc_cache(extra_out_loc_cache)
                batch.token_to_kv_pool_allocator.restore_state(exec_worker_state.cache_alloc_state)
                batch.token_to_kv_pool_allocator.precise_alloc(batch.out_cache_loc)
            else:
                batch.token_to_kv_pool_allocator.restore_state(exec_worker_state.cache_alloc_state)
            # batch.spec_info = draft_spec_info
            self.verify_input_spec_info = draft_spec_info
            # self.draft_token = draft_spec_info.draft_token
            # self.draft_token_num = draft_spec_info.draft_token_num
        else:
            self.verify_input_spec_info.enabled_skip_extend = False # no sumbit, no skip extend
            self.verify_input_spec_info.is_top_verify = is_top_verify
        self.verify_input_spec_info.capture_hidden_mode = self.model_router.get_model_capture_hidden_mode_from_table(exec_worker_name, "full")
        specrouter_debug_log(
            logger,
            "maybe_load_state_ready",
            exec_worker_name=exec_worker_name,
            last_worker_name=last_worker_name,
            num_verify_tokens=num_verify_tokens,
            submit_flag=self.submit_flag,
            verify_input_spec_info=describe_spec_info(self.verify_input_spec_info),
            batch=describe_batch(batch),
        )


    @nvtx_profile
    def _maybe_save_state(self, batch: ScheduleBatch, verify_output: EagleVerifyOutput, exec_worker_name: str, last_worker_name: str, req_pool_indices: torch.Tensor, spec_steps: int, spec_topk: int):
        # after verify
        # 存储刚被submit verify更新完且循环仍要继续的模型 （在在三级中不太可能出现）
        if exec_worker_name  != self.model_router.target_model_name:
            if self.submit_flag:
                # TODO
                # self.state_manager.take_snapshot(batch, [exec_worker_name])
                pass
            # 整合本次verify出来的结果。
            # self.state_manager.update_submit_inputs(exec_worker_name, verify_output)
            
            # logger.info(f"append_submit_inputs, {batch.seq_lens}, append id:{verify_output.verified_id}")
            submit_flag, keep_req_pool_indices, keep_reqs_index = self.state_manager.update_submit_inputs(verify_output, exec_worker_name, req_pool_indices, spec_steps, spec_topk)
            # logger.info(f"submit_flag: {submit_flag}, last_worker_name{last_worker_name}")
            self.submit_flag = submit_flag
            self.state_manager.update_reqs(batch, keep_req_pool_indices, keep_reqs_index)
            specrouter_debug_log(
                logger,
                "maybe_save_state_intermediate",
                exec_worker_name=exec_worker_name,
                last_worker_name=last_worker_name,
                submit_flag=self.submit_flag,
                keep_req_pool_indices_len=len(keep_req_pool_indices),
                keep_reqs_index_len=(
                    len(keep_reqs_index) if keep_reqs_index is not None else None
                ),
                batch=describe_batch(batch),
            )

            # if self.state_manager.out_loc_cache_for_token is not None:
                # dst_out_cache_loc = self.state_manager.out_loc_cache_for_token[:self.state_manager.out_loc_cache_num.item()].flatten()
                # logger.info(f"dst_out_cache_loc: {dst_out_cache_loc.shape}")
                # self.worker_map[exec_worker_name].dst_draft_id = self.state_manager.accept_token
                # # exec_worker_name /data/huggingface/TinyLlama-1.1B-Chat-v1.0
                # self.worker_map[exec_worker_name].dst_out_cache_loc = dst_out_cache_loc
                # self.worker_map[exec_worker_name].dst_modelhub_k_cache = self.worker_map[exec_worker_name].model_runner.token_to_kv_pool.get_key_buffer(10)[dst_out_cache_loc].clone()
        else:
            # check target extend
            if self.submit_flag and self.enabled_skip_extend:
                submit_inputs = self.state_manager.get_submit_inputs(last_worker_name)
                out_loc_cache_num = submit_inputs.out_loc_cache_num
                accept_length = verify_output.accept_length.to(out_loc_cache_num.device, dtype=torch.long)+1
                overflow_mask = accept_length > out_loc_cache_num
                # logger.info("\n")
                # logger.info(f" accept_length: {accept_length}, out_loc_cache_num: {out_loc_cache_num}")
                # logger.info(f"overflow_mask: {overflow_mask}, accept_length: {accept_length}, out_loc_cache_num: {out_loc_cache_num}")
                # logger.info(f"extend batch.out_cache_loc: {batch.out_cache_loc}")
                # logger.info(f"submit_flag, drafted_id: {submit_inputs.drafted_id}, verify_output: {verify_output.drafted_id}, accept_length: {accept_length}")
                # check_consistency(submit_inputs.out_loc_cache_for_token, batch.out_cache_loc, accept_length, submit_inputs.accepted_token[:, :-1])
                # logger.info(f"out_loc_cache_num: {out_loc_cache_num}, accept_length: {accept_length}")
                # logger.info(f"submit_inputs.drafted_id: {submit_inputs.drafted_id}, submit_inputs.out_loc_cache_for_token: {submit_inputs.out_loc_cache_for_token}, verify_output.drafted_id: {verify_output.drafted_id}, accept_length: {accept_length}")
                # logger.info(f"flat_drafted_id: {flat_drafted_id}, flat_out_locs: {flat_out_locs}")
                batch.spec_info.enabled_skip_extend = self.enabled_skip_extend
                # batch.spec_info.enabled_skip_extend = False
                if overflow_mask.any():
                    batch.spec_info.enabled_skip_extend = False
            specrouter_debug_log(
                logger,
                "maybe_save_state_target",
                exec_worker_name=exec_worker_name,
                last_worker_name=last_worker_name,
                submit_flag=self.submit_flag,
                batch=describe_batch(batch),
            )
    
    def rebuild_spec_info(self, batch: ScheduleBatch, logits_output: LogitsProcessorOutput, next_token_ids: torch.Tensor):
        device = batch.seq_lens.device
        batch.spec_info.accept_length = torch.zeros_like(next_token_ids, dtype=torch.long)

        keep_indices = [
                        i
                        for i in range(len(batch.reqs))
                        if not batch.reqs[i].finished()
                    ]
        seq_lens = batch.seq_lens.clone()
        # seq_lens_accepted = seq_lens.add_(1)
        seq_lens_accepted = seq_lens
        if len(keep_indices) != len(batch.reqs):
            keep_indices_device = torch.tensor(keep_indices, dtype=torch.long).to(
                device, non_blocking=True
            )
            batch.spec_info.drafted_id = batch.input_ids[keep_indices_device]
            batch.spec_info.verified_id = next_token_ids[keep_indices_device]
            batch.spec_info.req_pool_indices_for_draft_extend = batch.req_pool_indices[keep_indices_device]
            batch.spec_info.accept_length_for_draft_extend = batch.spec_info.accept_length[keep_indices_device]
            batch.spec_info.seq_lens_for_draft_extend = seq_lens_accepted[keep_indices_device]
            batch.spec_info.hidden_states = logits_output.hidden_states[keep_indices_device]
            batch.out_cache_loc = batch.out_cache_loc[keep_indices_device]
        else:
            batch.spec_info.drafted_id = batch.input_ids.clone()
            batch.spec_info.verified_id = next_token_ids.clone()
            batch.spec_info.req_pool_indices_for_draft_extend = batch.req_pool_indices
            batch.spec_info.accept_length_for_draft_extend = batch.spec_info.accept_length
            batch.spec_info.seq_lens_for_draft_extend = seq_lens_accepted
            batch.spec_info.hidden_states = logits_output.hidden_states
        batch.spec_info.accept_length_cpu = batch.spec_info.accept_length_for_draft_extend.tolist()
        batch.spec_info.next_out_cache_loc = batch.alloc_token_slots(len(batch.spec_info.verified_id))
        specrouter_debug_log(
            logger,
            "rebuild_spec_info_complete",
            batch=describe_batch(batch),
        )

    def forward_batch_multi_speculative_generation_hfrouter(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, List[int], int, int, bool]:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        # mid_worker = self.model_workers[1] if len(self.model_workers) > 2 else None
        # print("forward_speculative_generation", batch.forward_mode.name)
        if batch.forward_mode.is_decode():
            # logger.info(f"batch_size{batch.batch_size()}, forward_decode")
            # 若从自回归切回投机，可能没有有效的 spec_info（例如上一轮未进行 capture_for_decode）
            # 首次回到投机的 decode 时，先执行一次“预热式捕获”来补齐 spec_info（使用 decode 路径，避免再次 prefill/extend）
            current_chain_ids = self._current_chain_ids()

            last_state = "start"
            exec_worker_name = None
            self.submit_flag = False
            self.verify_input_spec_info = None

            # self.state_manager.take_snapshot(batch, self._current_chain_ids())

            if len(current_chain_ids) > 2:
                # self.state_manager.reset_all_state()
                # logger.info(f"start seq_lens:{batch.seq_lens}")
                self.state_manager.take_base_snapshot(batch)
                self.state_manager.create_max_draft_list(batch.batch_size(), self.server_args.speculative_num_steps, batch.device, self.dtype)
                self.state_manager.fit_seq_lens(batch, current_chain_ids[-2])
                # logger.info(f"after fit seq_lens:{batch.seq_lens}")
                # logger.info(f"batch: {batch.reqs}")
                # self.state_manager.save_src_info()
            
            
            # with count_time("all_time"):

            while True:
                last_worker_name = exec_worker_name
                current_chain_ids = self._current_chain_ids()
                next_state, exec_worker_name, sync_model_name_list = self.model_router.route_select(last_state, exec_worker_name, current_chain_ids, self.submit_flag, self.process_lazy_switch)
                # logger.info(f"next_state: {next_state}, exec_worker_name: {exec_worker_name}")
                if next_state == "end":
                    break
                with count_time("all_time"):
                    logits_output, next_token_ids, model_worker_batch_bid, num_accepted_tokens, can_run_cuda_graph = self.adaptive_spec_forward(batch, next_state, exec_worker_name, last_worker_name, sync_model_name_list)
                if (exec_worker_name == self.model_router.target_model_name) and (next_state in ["verify", "autoregressive"]):
                    # ready_to_return
                    logits_output_target = logits_output
                    # verify_output_target = verify_output
                    next_token_ids_target = next_token_ids
                    model_worker_batch_bid_target = model_worker_batch_bid
                    num_accepted_tokens_target = num_accepted_tokens
                    can_run_cuda_graph_target = can_run_cuda_graph
                    # logits_output_target, verify_output_target, model_worker_batch_bid_target, can_run_cuda_graph_target = adaptive_spec_output
                # post process
                # if next_state == "verify":
                #     submit_flag = self.check_submit(verify_output, exec_worker_name)
                last_state = next_state
                last_worker_name = exec_worker_name

            if len(current_chain_ids) > 2:
                self.state_manager.reset_all_state()

            # self._update_chain_time_stats()
            # self._maybe_update_chain()
            # self.print_chain_scheduler()

            return (
                logits_output_target,
                next_token_ids_target,
                model_worker_batch_bid_target,
                num_accepted_tokens_target,
                can_run_cuda_graph_target
            )
        elif batch.forward_mode.is_idle():
            # model_worker_batch = batch.get_model_worker_batch()
            return self.forward_batch_autoregressive_generation(batch)
        else:
            # from here prefill
            all_logits_list: List[torch.Tensor] = []
            model_names_in_order: List[str] = []
            weights_list: List[float] = []
            prefill_logits_num = 1
            next_token_ids_target = None

            if len(self.model_workers) > 1:
                target_workers_list = self.model_workers[1:]
                draft_worker = self.model_workers[0]
            else:
                target_workers_list = self.model_workers
                draft_worker = None

            # logger.info(f"batch.spec_info: prefill  {batch.spec_info}, {draft_worker}")
            # 从target开始，到最接近draft的模型结束
            for target_worker in target_workers_list[::-1]:
                self.state_manager.add_request_idx_count_from_prefill(target_worker.model_name, batch.req_pool_indices)
                # hidden_states_pool = self.model_hidden_states_pools[target_worker.model_name]
                # hidden_states_pool = None
                if draft_worker is not None:
                    batch.spec_info = EagleDraftInput(
                        # capture_hidden_mode=self.model_router.model_capture_hidden_mode[target_worker.model_name]
                        capture_hidden_mode=self.model_router.get_model_capture_hidden_mode_from_table(target_worker.model_name, "full")
                    )
                # else:
                #     hidden_states_pool = self.model_hidden_states_pools[target_worker.model_name]
                #     if hidden_states_pool is not None and hidden_states_pool.model_name not in self._current_chain_ids():
                #         batch.spec_info = EagleDraftInput(
                #             # capture_hidden_mode=self.model_router.model_capture_hidden_mode[target_worker.model_name]
                #             capture_hidden_mode=self.model_router.get_capture_hidden_mode_direct("full")
                #         )
                        
                logits_output, next_token_ids, bid = self._profile_call(
                    "prefill",
                    1, # forward_count
                    1.0, # verify_K
                    target_worker,
                    target_worker.forward_target_extend,
                    batch,
                    prefill_logits_num,
                )
                
                # if get_rank() == 0:
                #     logger.info(f"target prefill, batch.input_ids: {batch.input_ids}, batch.out_cache_loc: {batch.out_cache_loc}")
                # if hidden_states_pool is not None:
                #     hidden_states_pool.write_batch(logits_output.hidden_states[:-1, :], batch.out_cache_loc[1:])
                #     logits_output.hidden_states = None
                #     logger.info(f"logits_output.hidden_states out_cache_loc save: {batch.out_cache_loc[1:]}")



                logits_output_draft = logits_output
                # logger.info(f"prefill logits_output: {logits_output.next_token_logits.dtype}, {target_worker.model_runner.dtype}")
                # [B, S, V]
                logits_target = logits_output.input_token_logprobs.detach().squeeze(1).view(batch.batch_size(), prefill_logits_num, -1)
                all_logits_list.append(logits_target)
                model_names_in_order.append(target_worker.model_name)
                weights_list.append(1.0)

                if target_worker.model_name == self.model_router.target_model_name:
                    next_token_ids_target = next_token_ids
                else:
                    # self.state_manager.create_submit_input(target_worker.model_name, target_worker.speculative_num_steps, target_worker.topk, batch.device)
                    self.state_manager.create_submit_input(target_worker.model_name, self.max_speculative_num_steps, self.max_speculative_num_topk, batch.device)
                    # logger.info(f"append_submit_inputs, {batch.seq_lens}, append id:{[next_token_ids_target]}")
                    self.state_manager.append_submit_inputs(target_worker.model_name, [next_token_ids_target], batch.req_pool_indices)

            # for sub_worker in self.model_workers[1:-1][::-1]:
            #     batch.spec_info = EagleDraftInput(
            #         hidden_states=logits_output_draft.hidden_states,
            #         verified_id=next_token_ids,
            #         # capture_hidden_mode=self.model_router.model_capture_hidden_mode[sub_worker.model_name]
            #         capture_hidden_mode=self.model_router.get_model_capture_hidden_mode_from_table(sub_worker.model_name, "last")
            #     )
            #     with sub_worker.draft_tp_context(sub_worker.model_runner.tp_group):
            #         logits_output_draft, _, bid_draft = self._profile_call(
            #             "prefill",
            #             1, # forward_count
            #             1.0, # verify_K
            #             sub_worker,
            #             sub_worker.forward_target_extend,
            #             batch,
            #             prefill_logits_num,
            #         )
            #         # draft_dist = self._compute_distribution_from_logits(logits_output_draft)
            #     # if draft_dist is not None:
            #     #     dist_map[sub_worker.model_name] = draft_dist
            #     # logits_draft = logits_output_draft.next_token_logits.detach()
            #     # [L*B, 1, V] -> [L*B, V]
            #     logits_draft = logits_output_draft.input_token_logprobs.detach().squeeze(1).view(batch.batch_size(), prefill_logits_num+1, -1)[:, :-1, :]
            #     # if logits_draft.ndim == 2: # [B, V] -> [1, B, V]
            #     #     logits_draft = logits_draft.unsqueeze(0)
            #     all_logits_list.append(logits_draft)
            #     model_names_in_order.append(sub_worker.model_name)
            #     weights_list.append(1.0)
            
            # logger.info(f"batch.spec_info: prefill 1  {batch.spec_info}, {draft_worker}")

            if draft_worker is not None:
                self.state_manager.add_request_idx_count_from_prefill(draft_worker.model_name, batch.req_pool_indices)
                # draft worker
                # hidden_states_pool = self.model_hidden_states_pools[draft_worker.model_name]
                # if not draft_worker.speculative_algorithm.is_eagle() and hidden_states_pool is not None:
                #     # eagle base model 
                #     batch.spec_info = EagleDraftInput(
                #         verified_id=next_token_ids_target,
                #         capture_hidden_mode=self.model_router.get_capture_hidden_mode_direct("full")
                #     )
                # else:
                batch.spec_info = EagleDraftInput(
                    hidden_states=logits_output_draft.hidden_states,
                    verified_id=next_token_ids_target,
                    # capture_hidden_mode=self.model_router.model_capture_hidden_mode[sub_worker.model_name]
                    capture_hidden_mode=self.model_router.get_model_capture_hidden_mode_from_table(draft_worker.model_name, "last")
                )
                with draft_worker.draft_tp_context(draft_worker.model_runner.tp_group):
                    logits_output_draft, _, bid_draft = self._profile_call(
                        "prefill",
                        1, # forward_count
                        1.0, # verify_K
                        draft_worker,
                        draft_worker.forward_draft_extend,
                        batch,
                        prefill_logits_num
                    )
                # if not draft_worker.speculative_algorithm.is_eagle() and hidden_states_pool is not None:
                #     # todo
                #     hidden_states_pool.write_batch(logits_output_draft.hidden_states, batch.out_cache_loc)
                #     logits_output_draft.hidden_states = None
                
                    # draft_dist = self._compute_distribution_from_logits(logits_output_draft)
                # if draft_dist is not None:
                #     dist_map[sub_worker.model_name] = draft_dist
                # logits_draft = logits_output_draft.next_token_logits.detach()


                # [L*B, 1, V] -> [L*B, V]
                # logger.info(f"logits_output_draft: {logits_output_draft.input_token_logprobs.shape}, prefill_logits_num {prefill_logits_num}, batch.batch_size {batch.batch_size()}")
                logits_draft = logits_output_draft.input_token_logprobs.detach().squeeze(1).view(batch.batch_size(), prefill_logits_num, -1)
                # logits_draft = logits_output_draft.input_token_logprobs.detach().squeeze(1).view(batch.batch_size(), prefill_logits_num+1, -1)[:, 1:, :]
                # logits_draft = logits_output_draft.input_token_logprobs.detach().squeeze(1).view(batch.batch_size(), prefill_logits_num+1, -1)[:, :-1, :]
                if logits_draft.ndim == 2: # [B, V] -> [1, B, V]
                    logits_draft = logits_draft.unsqueeze(0)
                all_logits_list.append(logits_draft)
                model_names_in_order.append(draft_worker.model_name)
                if draft_worker.speculative_algorithm.is_eagle():
                    # weights_list.append(0.75)
                    weights_list.append(1.0)
                else:
                    weights_list.append(1.0)

                draft_worker.capture_for_decode(logits_output_draft, batch.spec_info)
                # if dist_map:
                #     self._update_similarity_with_probabilities(dist_map)3
                if self.chain_scheduler is not None and self.req_probs_latest is not None:
                    vocab_dims = {tensor.shape[-1] for tensor in all_logits_list}
                    if len(vocab_dims) == 1:
                        dtype = self.req_probs_latest.dtype
                        if len(all_logits_list) > 1:
                            stacked_logits = torch.stack(all_logits_list, dim=0).to(dtype)
                        else:
                            stacked_logits = all_logits_list[0].unsqueeze(0).to(dtype)
                        similarity_matrix = self._batch_compute_and_update_similarity(
                            stacked_logits, weights_list
                        )
                        self._update_req_probs_from_prefill(
                            batch, similarity_matrix, model_names_in_order, 1
                        )
            # logger.info(f"batch.spec_info: prefill end  {batch.spec_info}, {draft_worker}")


            # self._update_chain_time_stats(mode="prefill")
            # self._maybe_update_chain()
            # self._attach_batch_probs(batch)
            # logger.info(f"next_token_ids: {next_token_ids}")
            return logits_output, next_token_ids, bid, 0, False

if __name__ == "__main__":
    model_router = ModelRouter()
    model_router.update_router_chain_mock(["model1", "model2", "model3"])
    draft1_state = torch.randn(10, 100)
    draft2_state = torch.randn(10, 728)
    draft3_state = torch.randn(10, 1024)
    model_router.set_hidden_states("model1", draft1_state)
    model_router.set_hidden_states("model2", draft2_state)
    model_router.set_hidden_states("model3", draft3_state)
    print(model_router.get_hidden_states("model1").shape)
    print(model_router.get_hidden_states("model2").shape)
    print(model_router.get_hidden_states("model3").shape)
    print(model_router.model_draft)
    print(model_router.model_verify)
