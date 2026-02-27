import torch
from typing import Tuple

def shift_right_ragged(
    out_cache_loc: torch.Tensor,
    next_out_cache_loc: torch.Tensor,
    out_cache_loc_lens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_device = out_cache_loc.device
    
    if out_cache_loc_lens.device != target_device:
        out_cache_loc_lens = out_cache_loc_lens.to(target_device)
    
    if next_out_cache_loc.device != target_device:
        next_out_cache_loc = next_out_cache_loc.to(device=target_device, dtype=out_cache_loc.dtype)

    cumsum_lens = torch.cumsum(out_cache_loc_lens, dim=0)
    end_indices = cumsum_lens - 1
    start_indices = cumsum_lens - out_cache_loc_lens
    
    popped_tails = out_cache_loc[end_indices]
    
    new_out_cache_loc = torch.empty_like(out_cache_loc)
    new_out_cache_loc[start_indices] = next_out_cache_loc
    
    total_len = out_cache_loc.numel()
    source_mask = torch.ones(total_len, device=target_device, dtype=torch.bool)
    source_mask[end_indices] = False
    
    dest_mask = torch.ones(total_len, device=target_device, dtype=torch.bool)
    dest_mask[start_indices] = False
    
    new_out_cache_loc[dest_mask] = out_cache_loc[source_mask]
    
    return new_out_cache_loc, popped_tails


def shift_right_fixed(
    out_cache_loc: torch.Tensor,
    next_out_cache_loc: torch.Tensor,
    draft_num: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = next_out_cache_loc.size(0)
    
    if next_out_cache_loc.device != out_cache_loc.device:
        next_out_cache_loc = next_out_cache_loc.to(device=out_cache_loc.device, dtype=out_cache_loc.dtype)

    loc_view = out_cache_loc.view(batch_size, draft_num)
    
    popped_tails = loc_view[:, -1].clone()
    
    new_out_cache_loc = torch.empty_like(out_cache_loc)
    new_view = new_out_cache_loc.view(batch_size, draft_num)
    
    new_view[:, 0] = next_out_cache_loc
    new_view[:, 1:] = loc_view[:, :-1]
    
    return new_out_cache_loc, popped_tails

def shift_left_ragged(
    out_cache_loc: torch.Tensor,
    next_out_cache_loc: torch.Tensor,
    out_cache_loc_lens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_device = out_cache_loc.device
    
    # Device alignment
    if out_cache_loc_lens.device != target_device:
        out_cache_loc_lens = out_cache_loc_lens.to(target_device)
        
    if next_out_cache_loc.device != target_device:
        next_out_cache_loc = next_out_cache_loc.to(device=target_device, dtype=out_cache_loc.dtype)

    # Calculate boundaries
    cumsum_lens = torch.cumsum(out_cache_loc_lens, dim=0)
    end_indices = cumsum_lens - 1
    start_indices = cumsum_lens - out_cache_loc_lens
    
    # 1. Pop Head (Extract the first element of each request)
    popped_heads = out_cache_loc[start_indices]
    
    # Create container
    new_out_cache_loc = torch.empty_like(out_cache_loc)
    
    # 2. Append (Place new element at the end of each request)
    new_out_cache_loc[end_indices] = next_out_cache_loc
    
    # 3. Shift Left (Move remaining elements: old[start+1:] -> new[:end-1])
    total_len = out_cache_loc.numel()
    
    # Source: Skip the old heads
    source_mask = torch.ones(total_len, device=target_device, dtype=torch.bool)
    source_mask[start_indices] = False
    
    # Dest: Skip the new tails
    dest_mask = torch.ones(total_len, device=target_device, dtype=torch.bool)
    dest_mask[end_indices] = False
    
    new_out_cache_loc[dest_mask] = out_cache_loc[source_mask]
    
    return new_out_cache_loc, popped_heads


def shift_left_fixed(
    out_cache_loc: torch.Tensor,
    next_out_cache_loc: torch.Tensor,
    draft_num: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = next_out_cache_loc.size(0)
    
    if next_out_cache_loc.device != out_cache_loc.device:
        next_out_cache_loc = next_out_cache_loc.to(device=out_cache_loc.device, dtype=out_cache_loc.dtype)

    # View as 2D matrix
    loc_view = out_cache_loc.view(batch_size, draft_num)
    
    # 1. Pop Head (First column)
    popped_heads = loc_view[:, 0].clone()
    
    # Create new container and view
    new_out_cache_loc = torch.empty_like(out_cache_loc)
    new_view = new_out_cache_loc.view(batch_size, draft_num)
    
    # 2. Shift Left: Move [1:] to [:-1]
    new_view[:, :-1] = loc_view[:, 1:]
    
    # 3. Append: Fill last column
    new_view[:, -1] = next_out_cache_loc
    
    return new_out_cache_loc, popped_heads