

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)




class ModelRouter:
    def __init__(self,  **kwargs):
        self.hidden_states_dict = {}
        # self.model_verify = {}
        # self.model_draft = {}
        # self.cur_running_model = None
        # self.last_running_model = None
        self.router_chain = []
        self.target_model_name = None
        self.model_capture_hidden_mode = {}
        self.model_algorithm_table = {}

    # def set_hidden_states(self, model_name: str, hidden_states: torch.Tensor):
    #     self.hidden_states_dict[model_name] = hidden_states

    # def get_hidden_states(self, model_name: str):
    #     return self.hidden_states_dict[model_name]
    
    def update_model_algorithm_table(self, model_algorithm_table: Dict[str, SpeculativeAlgorithm]):
        for model_name, algorithm in model_algorithm_table.items():
            self.model_algorithm_table[model_name] = algorithm if isinstance(algorithm, SpeculativeAlgorithm) else SpeculativeAlgorithm.from_string(algorithm)

    def set_router_chain(self, router_chain: List[str]):
        # draft1-draft2-draft…… -target
        self.router_chain = router_chain

    def set_target_model_name(self, target_model_name: str):
        self.target_model_name = target_model_name
        
    
    # def reset_running_model(self):
    #     self.cur_running_model = self.router_chain[0]
    #     self.last_running_model = None
    #     self.hidden_states_dict = {}
    
    # def to_next_model(self):
    #     self.last_running_model = self.cur_running_model
    #     self.cur_running_model = self.model_verify[self.cur_running_model]
    #     return self.cur_running_model

    @property
    def draft_model_name(self):
        return self.router_chain[0]
    
    @property
    def verify_model_name(self):
        return self.router_chain[-1]
    
    def sync_model_name_list(self, last_model_name: str) -> List[str]:
        index = self.router_chain.index(last_model_name)
        return self.router_chain[:index]
    

    def to_next_model(self, last_model_name: str) -> str | None:
        try:
            index = self.router_chain.index(last_model_name)
            
            if index < len(self.router_chain) - 1:
                return self.router_chain[index + 1]
            else:
                return None
                
        except ValueError:
            return None

    def get_model_capture_hidden_mode(self, model_name: str, mode: str, target_chain: List[str]):
        index = target_chain.index(model_name)
        draft_index = max(0, index-1)
        # model_algorithm = self.model_algorithm_table[self.router_chain[index]]
        draft_model_algorithm = self.model_algorithm_table[target_chain[draft_index]]
        capture_mode = CaptureHiddenMode.NULL
        if draft_model_algorithm.is_eagle():
            if mode == "last":
                capture_mode = CaptureHiddenMode.LAST
            elif mode == "full":
                capture_mode = CaptureHiddenMode.FULL
            else:
                raise ValueError(f"Invalid mode: {mode}")
        return capture_mode

    def update_capture_hidden_mode_table(self, target_chain: List[str]=None):
        if target_chain is None:
            target_chain = self.router_chain
        for model_name in target_chain:
            self.model_capture_hidden_mode[model_name] = self.get_model_capture_hidden_mode(model_name, "last", target_chain)
    
    def get_model_capture_hidden_mode_from_table(self, model_name: str, mode: str):
        capture_mode = self.model_capture_hidden_mode[model_name]
        if capture_mode in [CaptureHiddenMode.LAST, CaptureHiddenMode.FULL]:
            if mode == "last":
                capture_mode = CaptureHiddenMode.LAST
            elif mode == "full":
                capture_mode = CaptureHiddenMode.FULL
            else:
                raise ValueError(f"Invalid mode: {mode}")
        return capture_mode
    
    def get_capture_hidden_mode_direct(self, mode: str):
        if mode == "last":
            return CaptureHiddenMode.LAST
        elif mode == "full":
            return CaptureHiddenMode.FULL
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def route_select(self, last_state: str, last_worker_name: str, current_chain_ids: List[str], submit_flag: bool, process_lazy_switch: bool):
        if last_state == "start":
            if process_lazy_switch:
                next_state = "unload"
                exec_worker_name = self.target_model_name
                return next_state, exec_worker_name, []
            elif len(current_chain_ids) > 1:
                next_state = "draft"
                exec_worker_name = self.draft_model_name
                return next_state, exec_worker_name, []
            else:
                next_state = "autoregressive"
                exec_worker_name = self.target_model_name
                return next_state, exec_worker_name, []

        elif last_state == "unload":
            if len(current_chain_ids) > 1:
                next_state = "draft"
                exec_worker_name = self.draft_model_name
                return next_state, exec_worker_name, []
            else:
                next_state = "autoregressive"
                exec_worker_name = self.target_model_name
                return next_state, exec_worker_name, []
            
            # next_state = "draft"
            # exec_worker_name = self.draft_model_name
            # return next_state, exec_worker_name, []
        
        elif last_state == "autoregressive":
            if process_lazy_switch:
                next_state = "reconfig"
                return next_state, last_worker_name, []
            else:
                next_state = "end"
                return next_state, None, []

        elif last_state == "reconfig":
            next_state = "draft_extend"
            return next_state, last_worker_name, self.sync_model_name_list(last_worker_name)


        elif last_state == "draft":
            next_state = "verify"
            exec_worker_name = self.to_next_model(last_worker_name)
            return next_state, exec_worker_name, []

        elif last_state == "verify":
            if last_worker_name == self.target_model_name and process_lazy_switch:
                next_state = "reconfig"
                return next_state, last_worker_name, []
            elif last_worker_name == self.target_model_name or submit_flag == False:
                next_state = "draft_extend"
                return next_state, last_worker_name, self.sync_model_name_list(last_worker_name)
            else: 
                # last_worker_name != self.target_model_name and submit_flag==True  continue verify
                next_state = "verify"
                exec_worker_name = self.to_next_model(last_worker_name)
                return next_state, exec_worker_name, []
        elif last_state == "draft_extend":
            if last_worker_name == self.target_model_name:
                next_state = "end"
                return next_state, None, []
            else:
                next_state = "draft"
                exec_worker_name = self.draft_model_name
                return next_state, exec_worker_name, []
        else:
            raise ValueError(f"Invalid next state: {last_state}")