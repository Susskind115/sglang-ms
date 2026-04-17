"""Minimal SWA (Sliding Window Attention) KV Pool for mixed head-dim models like Gemma4.

Keeps two separate MHATokenToKVPool instances: one for sliding-window layers
(smaller head_dim/kv_heads) and one for full-attention layers (larger head_dim/kv_heads).
Routes get/set operations via a layer_id -> (pool_index, is_swa) mapping.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024


class SWAKVPool(KVCache):
    """KV cache with separate pools for full and SWA attention layers."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        # full attention params
        head_num: int,
        head_dim: int,
        # swa attention params
        swa_head_num: int,
        swa_head_dim: int,
        # layer routing
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        device: str,
    ):
        # Don't call super().__init__ with layer_num since we manage two sub-pools
        self.size = size
        self.size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.start_layer = 0
        self.swa_loc = None
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        self.swa_kv_pool = MHATokenToKVPool(
            size=size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=swa_head_num,
            head_dim=swa_head_dim,
            layer_num=len(swa_attention_layer_ids),
            device=device,
            enable_memory_saver=False,
        )
        self.full_kv_pool = MHATokenToKVPool(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=len(full_attention_layer_ids),
            device=device,
            enable_memory_saver=False,
        )

        # {layer_id: (index_in_sub_pool, is_swa_layer)}
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for pool_idx, global_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_id] = (pool_idx, False)
        for pool_idx, global_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_id] = (pool_idx, True)

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"SWAKVPool allocated. swa_size={size_swa}, full_size={size}, "
            f"K+V={((k_size + v_size) / GB):.2f} GB"
        )

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def get_kv_size_bytes(self):
        k1, v1 = self.full_kv_pool.get_kv_size_bytes()
        k2, v2 = self.swa_kv_pool.get_kv_size_bytes()
        return k1 + k2, v1 + v2

    def get_key_buffer(self, layer_id: int):
        pool_idx, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_key_buffer(pool_idx)

    def get_value_buffer(self, layer_id: int):
        pool_idx, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_value_buffer(pool_idx)

    def get_kv_buffer(self, layer_id: int):
        pool_idx, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_kv_buffer(pool_idx)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        if self.full_to_swa_index_mapping is not None:
            return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)
        return kv_indices

    def set_swa_loc(self, loc: torch.Tensor):
        self.swa_loc = loc

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        layer_id_override: Optional[int] = None,
    ):
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        pool_idx, is_swa = self.layers_mapping[layer_id]

        if is_swa:
            actual_loc = loc
            if self.swa_loc is not None:
                actual_loc = self.swa_loc
            elif self.full_to_swa_index_mapping is not None:
                actual_loc = self.translate_loc_from_full_to_swa(loc)
            self.swa_kv_pool.set_kv_buffer(
                None, actual_loc, cache_k, cache_v, k_scale, v_scale,
                layer_id_override=pool_idx,
            )
        else:
            self.full_kv_pool.set_kv_buffer(
                None, loc, cache_k, cache_v, k_scale, v_scale,
                layer_id_override=pool_idx,
            )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)
        tgt_swa = self.translate_loc_from_full_to_swa(tgt_loc)
        src_swa = self.translate_loc_from_full_to_swa(src_loc)
        self.swa_kv_pool.move_kv_cache(tgt_swa, src_swa)

    def get_flat_data(self, indices):
        return self.full_kv_pool.get_flat_data(indices)

    def transfer(self, indices, flat_data):
        return self.full_kv_pool.transfer(indices, flat_data)

    def transfer_per_layer(self, indices, flat_data, layer_id):
        pool_idx, is_swa = self.layers_mapping.get(layer_id, (layer_id, False))
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.transfer_per_layer(indices, flat_data, pool_idx)
