"""Model Hub of the SGLang."""

import logging
import os
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import torch
from huggingface_hub import snapshot_download
from torch.distributed import get_rank

# from sglang.srt.distributed import GroupCoordinator, patch_tensor_parallel_group
# from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator
# from sglang.srt.layers.dp_attention import disable_dp_size
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.speculative.eagle_utils import EagleDraftInput

from sglang.srt.managers.schedule_batch import (
    ScheduleBatch,
    get_last_loc,
    global_server_args_dict,
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

if is_cuda():
    from sgl_kernel import segment_packbits

logger = logging.getLogger(__name__)


class ModelRouter:
    def __init__(self,  **kwargs):
        self.hidden_states_dict = {}
        self.model_verify = {}
        self.model_draft = {}
        self.model_capture_hidden_mode = {}
        self.cur_running_model = None
        self.last_running_model = None
        self.router_chain = []

    def set_hidden_states(self, model_name: str, hidden_states: torch.Tensor):
        self.hidden_states_dict[model_name] = hidden_states

    def get_hidden_states(self, model_name: str):
        return self.hidden_states_dict[model_name]
    
    # def init_model_capture_hidden_mode(self, model_attr_list: List[Tuple[str, str]]):
    #     last_speculative_algorithm = SpeculativeAlgorithm.from_string(model_attr_list[0][1])
    #     for model_name, speculative_algorithm_str in model_attr_list:
    #         speculative_algorithm = SpeculativeAlgorithm.from_string(speculative_algorithm_str)
    #         self.model_capture_hidden_mode[model_name] = self.check_capture_hidden_mode(last_speculative_algorithm)
    #         last_speculative_algorithm = speculative_algorithm
    #     # if get_rank() == 0:
    #     #     logger.info(f"model_capture_hidden_mode: {self.model_capture_hidden_mode}")

    def init_model_capture_hidden_mode(self, server_args_list: List[ServerArgs]):
        last_speculative_algorithm = SpeculativeAlgorithm.from_string(server_args_list[0].speculative_algorithm)
        for server_args in server_args_list:
            speculative_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
            self.model_capture_hidden_mode[server_args.model_path] = self.check_capture_hidden_mode(last_speculative_algorithm)
            server_args.capture_hidden_mode = self.model_capture_hidden_mode[server_args.model_path]
            last_speculative_algorithm = speculative_algorithm
        return server_args_list
        # if get_rank() == 0:
        #     logger.info(f"model_capture_hidden_mode: {self.model_capture_hidden_mode}")

    def update_router_chain_mock(self, model_attr_list: List[Tuple[str, SpeculativeAlgorithm]]):
        self.set_router_chain(model_attr_list)

    def set_router_chain(self, router_chain: List[Tuple[str, SpeculativeAlgorithm]]):
        # draft1-draft2-draft……-target
        self.router_chain = router_chain
        self.model_verify[router_chain[-1][0]] = None
        # self.model_draft[router_chain[0]] = None
        last_model_name = None
        
        # last_model_name = router_chain[0]
        for i, (model_name, _) in enumerate(router_chain):
            self.model_draft[model_name] = last_model_name
            if last_model_name is not None:
                self.model_verify[last_model_name] = model_name
            last_model_name = model_name
        
    
    def reset_running_model(self):
        self.cur_running_model = self.router_chain[0]
        self.last_running_model = None
        self.hidden_states_dict = {}
    
    def to_next_model(self):
        self.last_running_model = self.cur_running_model
        self.cur_running_model = self.model_verify[self.cur_running_model]
        return self.cur_running_model
    
    def check_capture_hidden_mode(self, speculative_algorithm: SpeculativeAlgorithm):
        capture_hidden_mode = None
        if speculative_algorithm.is_eagle2():
            capture_hidden_mode = CaptureHiddenMode.LAST
        elif speculative_algorithm.is_eagle3():
            capture_hidden_mode = CaptureHiddenMode.FULL
        elif speculative_algorithm.is_not_eagle():
            capture_hidden_mode = CaptureHiddenMode.NULL
            # capture_hidden_mode = CaptureHiddenMode.LAST
        else:
            raise ValueError(f"Invalid speculative algorithm: {speculative_algorithm}")
        return capture_hidden_mode


class ModelHub:
    def __init__(self, server_args_list: List[ServerArgs], router_args: dict,
                 main_worker_kargs: dict,
                 is_generation: bool = False, **kwargs):
        self.main_model_worker = None
        self.model_workers = []
        # self.model_workers_ids = []
        self.spec_worker = None
        self.defer_memory_init = True
        self.router_args = router_args
        self.model_router = ModelRouter()

        # 动态切换配置：(autoregressive_limit, speculative_limit)
        # self.running_requests_limit = (4097, 256)
        self.running_requests_limit = (4097, 48)
        self.server_args = server_args_list[-1]

        # 保存原始的max_running_requests配置
        self.original_max_running_requests = self.server_args.max_running_requests

        self.is_generation = is_generation
        self.enable_overlap = not self.server_args.disable_overlap_schedule

        self.spec_flag = router_args.get("spec_flag", False)
        self.speculative_num_steps = self.server_args.speculative_num_steps
        self.speculative_num_draft_tokens = self.server_args.speculative_num_draft_tokens
        self.speculative_eagle_topk = self.server_args.speculative_eagle_topk

        self.current_mode = "autoregressive" if not self.spec_flag else "speculative"

        # 动态切换相关状态
        self.mode_switch_enabled = router_args["mode_switch_enabled"]
        self.enabled_skip_extend = router_args["enabled_skip_extend"]
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
            # server_args_list[-1].max_running_requests = self.running_requests_limit[0]
            logger.info("spec_flag is False, only use target model")
            # self.max_running_requests = self.running_requests_limit[0]
            # self.set_inference_mode("autoregressive")
        else:
            # server_args_list[-1].max_running_requests = self.running_requests_limit[1]
            logger.info("spec_flag is True, use target model and draft model")
            # self.max_running_requests = self.running_requests_limit[1]
            # self.set_inference_mode("speculative")

        # model hub
        selected_server_args_list_mock = server_args_list[:]
        model_attr_list = [(server_args.model_path, server_args.speculative_algorithm) for server_args in selected_server_args_list_mock]
        selected_server_args_list_mock = self.model_router.init_model_capture_hidden_mode(selected_server_args_list_mock)
        self.model_workers = self.init_workers(selected_server_args_list_mock, main_worker_kargs, TpWorkerClass)
        self.model_router.update_router_chain_mock(model_attr_list)


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
        draft_worker_kargs = main_worker_kargs.copy()
        draft_worker_kargs["is_draft_worker"] = True
        workers = []
        
        for i, server_args in enumerate(server_args_list[:-1]):
            # TODO 暂时测试性能，不追求优雅设计
            server_args.context_length = main_model_worker.model_runner.model_config.context_len
            draft_worker = TpModelWorker(server_args=server_args, **draft_worker_kargs, 
                                         model_config=model_configs[i],
                                         defer_memory_init=self.defer_memory_init)
            
            if draft_worker.speculative_algorithm.is_eagle():
                # TODO 需要手动设定一种指向eagle base模型的方式，现在先粗糙实现
                draft_worker.set_eagle_embed_and_head(main_model_worker)
            workers.append(draft_worker)
        workers.append(main_model_worker)

        

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

        return workers

    def set_inference_mode(self, mode: str):
        """动态切换推理模式，支持'speculative'和'autoregressive'

        Args:
            mode: 推理模式 ('speculative' 或 'autoregressive')
            scheduler_ref: 调度器引用，用于动态调整max_micro_batch_size

        Returns:
            bool: 切换是否成功
        """
        if mode not in ["speculative", "autoregressive"]:
            raise ValueError(f"Unsupported inference mode: {mode}")

        # if not self.mode_switch_enabled:
        #     logger.warning("Dynamic mode switching is disabled")
        #     return False
        if self.current_mode == mode:
            return True

        old_mode = self.current_mode
        self.current_mode = mode

        if old_mode == "speculative" and mode == "autoregressive":
            # aux autoregressive
            # self.spec_flag = False
            self.speculative_num_steps = 1
            self.speculative_eagle_topk = 1
            self.speculative_num_draft_tokens = 0
            for worker in self.model_workers:
                # worker.spec_flag = False
                worker.speculative_num_steps = 1
                worker.speculative_eagle_topk = 1
                worker.speculative_num_draft_tokens = 0
                # worker.model_runner.keep_spec_info = True

        elif old_mode == "autoregressive" and mode == "speculative":
            # self.spec_flag = True
            self.speculative_num_steps = self.server_args.speculative_num_steps
            self.speculative_eagle_topk = self.server_args.speculative_eagle_topk
            self.speculative_num_draft_tokens = self.server_args.speculative_num_draft_tokens
            for worker in self.model_workers:
                # worker.spec_flag = True
                worker.speculative_num_steps = worker.model_runner.server_args.speculative_num_steps
                worker.speculative_eagle_topk = worker.model_runner.server_args.speculative_eagle_topk
                worker.speculative_num_draft_tokens = worker.model_runner.server_args.speculative_num_draft_tokens
                # worker.model_runner.keep_spec_info = False

        # # 动态调整有效的并发限制
        # if mode == "autoregressive":
        #     self.effective_max_running_requests = self.running_requests_limit[0]  # 4097
        #     target_micro_batch_size = self.running_requests_limit[0]
        # else:  # speculative
        #     self.effective_max_running_requests = self.running_requests_limit[1]  # 48
        #     target_micro_batch_size = self.running_requests_limit[1]

        # # 如果提供了调度器引用，动态调整max_micro_batch_size
        # if scheduler_ref is not None:
        #     # success = self._update_scheduler_batch_size(scheduler_ref, target_micro_batch_size)
        #     if not success:
        #         # 回滚模式切换
        #         self.current_mode = old_mode
        #         if old_mode == "autoregressive":
        #             self.effective_max_running_requests = self.running_requests_limit[0]
        #         else:
        #             self.effective_max_running_requests = self.running_requests_limit[1]
        #         return False

        logger.info(f"Successfully switched inference mode from {old_mode} to {mode}, ")
        return True

    # def set_inference_mode(self, mode: str, scheduler_ref=None):
    #     """动态切换推理模式，支持'speculative'和'autoregressive'

    #     Args:
    #         mode: 推理模式 ('speculative' 或 'autoregressive')
    #         scheduler_ref: 调度器引用，用于动态调整max_micro_batch_size

    #     Returns:
    #         bool: 切换是否成功
    #     """
    #     if mode not in ["speculative", "autoregressive"]:
    #         raise ValueError(f"Unsupported inference mode: {mode}")

    #     if not self.mode_switch_enabled:
    #         logger.warning("Dynamic mode switching is disabled")
    #         return False

    #     old_mode = self.current_mode
    #     self.current_mode = mode

    #     # 动态调整有效的并发限制
    #     if mode == "autoregressive":
    #         self.effective_max_running_requests = self.running_requests_limit[0]  # 4097
    #         target_micro_batch_size = self.running_requests_limit[0]
    #     else:  # speculative
    #         self.effective_max_running_requests = self.running_requests_limit[1]  # 48
    #         target_micro_batch_size = self.running_requests_limit[1]

    #     # 如果提供了调度器引用，动态调整max_micro_batch_size
    #     if scheduler_ref is not None:
    #         success = self._update_scheduler_batch_size(scheduler_ref, target_micro_batch_size)
    #         if not success:
    #             # 回滚模式切换
    #             self.current_mode = old_mode
    #             if old_mode == "autoregressive":
    #                 self.effective_max_running_requests = self.running_requests_limit[0]
    #             else:
    #                 self.effective_max_running_requests = self.running_requests_limit[1]
    #             return False

    #     logger.info(f"Successfully switched inference mode from {old_mode} to {mode}, "
    #                f"effective_max_running_requests={self.effective_max_running_requests}")
    #     return True

    # def _update_scheduler_batch_size(self, scheduler_ref, target_batch_size):
    #     """更新调度器的max_micro_batch_size"""
    #     try:
    #         from sglang.srt.managers.io_struct import SetInternalStateReq

    #         # 计算合适的max_micro_batch_size
    #         pp_size = getattr(scheduler_ref, 'pp_size', 1)
    #         max_allowed = self.max_running_requests // pp_size
    #         new_micro_batch_size = min(target_batch_size // pp_size, max_allowed)
    #         new_micro_batch_size = max(new_micro_batch_size, 1)  # 至少为1

    #         # 构造更新请求
    #         update_req = SetInternalStateReq(
    #             server_args={"max_micro_batch_size": new_micro_batch_size}
    #         )

    #         # 调用调度器的更新方法
    #         result = scheduler_ref.set_internal_state(update_req)

    #         if result.updated:
    #             logger.info(f"Updated max_micro_batch_size to {new_micro_batch_size} for mode {self.current_mode}")
    #             return True
    #         else:
    #             logger.error("Failed to update scheduler internal state")
    #             return False

    #     except Exception as e:
    #         logger.error(f"Failed to update scheduler batch size: {e}")
    #         return False

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
        
    def forward_batch(self, batch: ScheduleBatch) -> Tuple[LogitsProcessorOutput, List[int], int, int, bool]:
        """统一的推理入口，根据当前模式选择执行路径
        Args:
            batch: 要处理的批次
            
        Returns:
            统一的返回格式: (logits_output, next_token_ids, batch_id, accepted_tokens, can_run_cuda_graph)
        """
        # if batch.batch_size() >= 40:
        #     self.set_inference_mode("autoregressive")
        #     # batch.spec_flag = False
        # else:
        #     self.set_inference_mode("speculative")
            # batch.spec_flag = True
        # logger.info(f"forward_batch, current_mode: {self.current_mode}")
            
        # if self.current_mode == "speculative":
        if self.spec_flag:
            # print("forward_batch_speculative_generation")
            # return self.forward_batch_speculative_generation(batch)
            # return self.forward_batch_multi_speculative_generation(batch)
            return self.forward_batch_multi_speculative_generation_hfrouter(batch)
        # elif self.current_mode == "autoregressive":
        else:
            # print("forward_batch_autoregressive_generation")
            return self.forward_batch_autoregressive_generation(batch)
        # else:
        #     raise ValueError(f"Invalid mode: {self.current_mode}")
    
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
        
        # 执行自回归推理
        logits_output, next_token_ids, can_run_cuda_graph = (
            target_worker.forward_batch_generation(model_worker_batch)
        )
        
        # 返回与投机推理相同格式的结果
        return logits_output, next_token_ids, model_worker_batch.bid, 0, can_run_cuda_graph
    
    def forward_batch_speculative_generation(
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
        draft_worker, target_worker = self.model_workers[0], self.model_workers[-1]
        # print("forward_speculative_generation", batch.forward_mode.name)
        if batch.forward_mode.is_decode():
            # logger.info(f"batch_size{batch.batch_size()}, forward_decode")
            with draft_worker.draft_tp_context(draft_worker.model_runner.tp_group):
                spec_info = draft_worker.draft(batch)

                # retrive_next_token, retrive_next_sibling, retrive_index = spec_info.retrive_next_token.to("cpu").tolist(), spec_info.retrive_next_sibling.to("cpu").tolist(), spec_info.retrive_index.to("cpu").tolist()
                # if get_rank() == 0:
                #     logger.info(f"retrive_next_token: {retrive_next_token}, retrive_next_sibling: {retrive_next_sibling}")
            # logger.info(f" forward_verify")
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                target_worker.verify(batch, spec_info)
            )
            # logger.info(f"logits_output: {logits_output}, verify_output: {verify_output}, model_worker_batch: {model_worker_batch}, can_run_cuda_graph: {can_run_cuda_graph}")

            # If it is None, it means all requests are finished
            if batch.spec_info.verified_id is not None:
                with draft_worker.draft_tp_context(draft_worker.model_runner.tp_group):
                    draft_worker.forward_draft_extend_after_decode(batch)
            return (
                logits_output,
                verify_output.verified_id,
                model_worker_batch.bid,
                sum(verify_output.accept_length_per_req_cpu),
                can_run_cuda_graph,
            )
        elif batch.forward_mode.is_idle():
            # logger.info(f"batch_size{batch.batch_size()}, forward_idle")
            model_worker_batch = batch.get_model_worker_batch()
            # logits_output, next_token_ids, _ = (
            #     target_worker.forward_batch_generation(model_worker_batch)
            # )

            # return logits_output, next_token_ids, model_worker_batch.bid, 0, False
            return self.forward_batch_autoregressive_generation(batch)
        else:
            # logger.info(f"batch_size{batch.batch_size()}, spec_info:{batch.spec_info}, forward_target_and_draft_extend prefill")
            logits_output, next_token_ids, bid = target_worker.forward_target_extend(batch)
            # logger.info(f"batch_size{batch.batch_size()}, spec_info:{batch.spec_info}, forward_target_extend prefill done")
            with draft_worker.draft_tp_context(draft_worker.model_runner.tp_group):
                draft_worker.forward_draft_extend(
                    batch, logits_output.hidden_states, next_token_ids
                )
            # logger.info(f"batch_size{batch.batch_size()}, spec_info:{batch.spec_info}, forward_draft_extend prefill done")
            return logits_output, next_token_ids, bid, 0, False


    
    def forward_batch_multi_speculative_generation(
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
        draft_worker, target_worker = self.model_workers[0], self.model_workers[-1]
        mid_worker = self.model_workers[1] if len(self.model_workers) > 2 else None
        # print("forward_speculative_generation", batch.forward_mode.name)
        if batch.forward_mode.is_decode():
            # logger.info(f"batch_size{batch.batch_size()}, forward_decode")
            
            with draft_worker.draft_tp_context(draft_worker.model_runner.tp_group):
                spec_info = draft_worker.draft(batch)

                # retrive_next_token, retrive_next_sibling, retrive_index = spec_info.retrive_next_token.to("cpu").tolist(), spec_info.retrive_next_sibling.to("cpu").tolist(), spec_info.retrive_index.to("cpu").tolist()
                # if get_rank() == 0:
                #     logger.info(f"retrive_next_token: {retrive_next_token}, retrive_next_sibling: {retrive_next_sibling}")
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                target_worker.verify(batch, spec_info)
            )
            # If it is None, it means all requests are finished
            if batch.spec_info.verified_id is not None:
                with draft_worker.draft_tp_context(draft_worker.model_runner.tp_group):
                    draft_worker.forward_draft_extend_after_decode(batch)
            return (
                logits_output,
                verify_output.verified_id,
                model_worker_batch.bid,
                sum(verify_output.accept_length_per_req_cpu),
                can_run_cuda_graph,
            )
        elif batch.forward_mode.is_idle():
            model_worker_batch = batch.get_model_worker_batch()
            return self.forward_batch_autoregressive_generation(batch)
        else:
            batch.spec_info = EagleDraftInput(
                capture_hidden_mode=self.model_router.model_capture_hidden_mode[target_worker.model_name]
            )
            logits_output, next_token_ids, bid = target_worker.forward_target_extend(batch)
            logits_output_draft = logits_output

            for sub_worker in self.model_workers[:-1][::-1]:
                batch.spec_info = EagleDraftInput(
                    hidden_states=logits_output_draft.hidden_states,
                    verified_id=next_token_ids,
                    capture_hidden_mode=self.model_router.model_capture_hidden_mode[sub_worker.model_name]
                )
                with sub_worker.draft_tp_context(sub_worker.model_runner.tp_group):
                    logits_output_draft, _, bid_draft = sub_worker.forward_draft_extend(
                        batch
                    )
            draft_worker.capture_for_decode(logits_output_draft, batch.spec_info)
            return logits_output, next_token_ids, bid, 0, False



    
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
        draft_worker, target_worker = self.model_workers[0], self.model_workers[-1]
        mid_worker = self.model_workers[1] if len(self.model_workers) > 2 else None
        # print("forward_speculative_generation", batch.forward_mode.name)
        if batch.forward_mode.is_decode():
            # logger.info(f"batch_size{batch.batch_size()}, forward_decode")
            
            with draft_worker.draft_tp_context(draft_worker.model_runner.tp_group):
                spec_info = draft_worker.draft(batch)

            # if torch.distributed.get_rank() == 0:
            #     logger.info(f"draft out out_cache_loc: {batch.out_cache_loc}")
                # retrive_next_token, retrive_next_sibling, retrive_index = spec_info.retrive_next_token.to("cpu").tolist(), spec_info.retrive_next_sibling.to("cpu").tolist(), spec_info.retrive_index.to("cpu").tolist()
                # if get_rank() == 0:
                #     logger.info(f"retrive_next_token: {retrive_next_token}, retrive_next_sibling: {retrive_next_sibling}")
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                target_worker.verify(batch, spec_info)
            )
            # If it is None, it means all requests are finished
            if batch.spec_info.verified_id is not None:
                with draft_worker.draft_tp_context(draft_worker.model_runner.tp_group):
                    draft_worker.forward_draft_extend_after_decode(batch)

            return (
                logits_output,
                verify_output.verified_id,
                model_worker_batch.bid,
                sum(verify_output.accept_length_per_req_cpu),
                can_run_cuda_graph,
            )
        elif batch.forward_mode.is_idle():
            model_worker_batch = batch.get_model_worker_batch()
            return self.forward_batch_autoregressive_generation(batch)
        else:
            batch.spec_info = EagleDraftInput(
                capture_hidden_mode=self.model_router.model_capture_hidden_mode[target_worker.model_name]
            )
            logits_output, next_token_ids, bid = target_worker.forward_target_extend(batch)
            logits_output_draft = logits_output

            for sub_worker in self.model_workers[:-1][::-1]:
                batch.spec_info = EagleDraftInput(
                    hidden_states=logits_output_draft.hidden_states,
                    verified_id=next_token_ids,
                    capture_hidden_mode=self.model_router.model_capture_hidden_mode[sub_worker.model_name]
                )
                with sub_worker.draft_tp_context(sub_worker.model_runner.tp_group):
                    logits_output_draft, _, bid_draft = sub_worker.forward_draft_extend(
                        batch
                    )
            draft_worker.capture_for_decode(logits_output_draft, batch.spec_info)
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