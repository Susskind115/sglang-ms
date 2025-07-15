# NOTE: Please run this file to make sure the test cases are correct.

from typing import List

import torch

from sglang.srt.utils import is_cuda, is_hip

import torch.distributed as dist

import pytest

if is_cuda() or is_hip():
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )

def first_rank_print(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        # 单机环境或分布式未初始化时直接打印
        print(*args, **kwargs)

def build_tree_kernel_efficient_preprocess(
    verified_id: torch.Tensor,
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_verify_tokens: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(
        1
    )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
    ss_token_list = torch.cat(
        token_list, dim=1
    )  # b, (self.topk + (num_steps-1) * self.topk)
    top_scores = torch.topk(score_list, num_verify_tokens - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
):
    parent_list, top_scores_index, draft_tokens = (
        build_tree_kernel_efficient_preprocess(
            verified_id,
            score_list,
            token_list,
            parents_list,
            num_verify_tokens,
        )
    )

    # first_rank_print(f"parent_list: {parent_list}", flush=True)
    # first_rank_print(f"top_scores_index: {top_scores_index}", flush=True)
    # first_rank_print(f"draft_tokens: {draft_tokens}", flush=True)

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    tree_mask = torch.full(
        (
            seq_lens_sum * num_verify_tokens
            + num_verify_tokens * num_verify_tokens * bs,
        ),
        True,
        device=device,
    )
    retrive_index = torch.full(
        (bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_next_token = torch.full(
        (bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_next_sibling = torch.full(
        (bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    positions = torch.empty((bs * num_verify_tokens,), device=device, dtype=torch.long)

    # sgl_build_tree_kernel_efficient(
    #     parent_list,
    #     top_scores_index,
    #     seq_lens.to(torch.int32),
    #     tree_mask,
    #     positions,
    #     retrive_index,
    #     retrive_next_token,
    #     retrive_next_sibling,
    #     topk,
    #     spec_steps,
    #     num_verify_tokens,
    # )
    sgl_build_tree_kernel_efficient_python(
        parent_list,
        top_scores_index,
        seq_lens.to(torch.int32),
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        spec_steps,
        num_verify_tokens,
    )

    
    reconstruct_tree_lineage(
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        positions=positions,
        final_draft_tokens=draft_tokens,
        top_scores_index=top_scores_index,
        original_score_list=score_list,
        original_token_list=token_list,
        original_parents_list=parents_list,
        verified_id=verified_id,
        seq_lens=seq_lens,
        topk=topk,
    )

    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )

import torch

def sgl_build_tree_kernel_efficient_python(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
):
    """
    A Python implementation of the `build_tree_efficient` CUDA kernel.

    This function serves as a reference for understanding and debugging the CUDA kernel's logic.
    It processes each sequence in the batch and each token within the sequence iteratively.
    """
    bs = parent_list.size(0)

    # Calculate the starting offset for each sequence in the flattened tree_mask.
    seq_len_cumsum = torch.cat([torch.tensor([0], device=verified_seq_len.device), torch.cumsum(verified_seq_len, dim=0)])
    
    # Outer loop simulates the CUDA grid
    for bid in range(bs):
        seq_len = verified_seq_len[bid].item()
        seq_tree_offset = (seq_len_cumsum[bid] * draft_token_num + 
                           draft_token_num * draft_token_num * bid)

        # Inner loop simulates the CUDA block
        for tid in range(draft_token_num):
            token_tree_row_start_idx = seq_tree_offset + (seq_len + draft_token_num) * tid
            draft_tokens_mask_start_idx = token_tree_row_start_idx + seq_len
            
            for i in range(draft_token_num - 1):
                 tree_mask[draft_tokens_mask_start_idx + 1 + i] = False

            # Logic for thread 0
            if tid == 0:
                # Use 1D indexing for `positions`
                positions[bid * draft_token_num + 0] = seq_len
                
                retrive_index_offset = bid * draft_token_num
                # Use 2D indexing for `retrive_index`
                retrive_index[bid, 0] = retrive_index_offset

                for i in range(draft_token_num - 1, 0, -1):
                    current_token_global_idx = retrive_index_offset + i
                    # Use 2D indexing for `retrive_index`
                    retrive_index[bid, i] = current_token_global_idx
                    
                    parent_tb_idx = selected_index[bid, i - 1] // topk
                    parent_position_in_draft_list = 0
                    
                    if parent_tb_idx > 0:
                        parent_token_original_idx = parent_list[bid, parent_tb_idx]
                        found = False
                        for p_pos in range(draft_token_num - 1):
                            if selected_index[bid, p_pos] == parent_token_original_idx:
                                parent_position_in_draft_list = p_pos + 1
                                found = True
                                break
                        if not found:
                           print(f"WARNING: Invalid eagle tree!!! Detected a token with no parent token selected for bid={bid}, token_i={i}")
                           continue
                    
                    # Use 2D indexing for `retrive_next_token` and `retrive_next_sibling`
                    origin_next_token = retrive_next_token[bid, parent_position_in_draft_list].item()
                    
                    retrive_next_token[bid, parent_position_in_draft_list] = i
                    if origin_next_token != -1:
                        retrive_next_sibling[bid, i] = origin_next_token

            # Logic for other threads
            else:
                position_depth = 0
                current_token_in_selected_list_idx = tid - 1

                while True:
                    position_depth += 1
                    
                    parent_in_draft_list_idx = current_token_in_selected_list_idx
                    mask_idx_to_set = draft_tokens_mask_start_idx + parent_in_draft_list_idx
                    tree_mask[mask_idx_to_set] = True

                    parent_tb_idx = selected_index[bid, current_token_in_selected_list_idx] // topk
                    
                    if parent_tb_idx == 0:
                        break
                    
                    parent_token_original_idx = parent_list[bid, parent_tb_idx]

                    found = False
                    for p_pos in range(draft_token_num - 1):
                        if selected_index[bid, p_pos] == parent_token_original_idx:
                            current_token_in_selected_list_idx = p_pos
                            found = True
                            break
                    if not found:
                        break
                
                # Use 1D indexing for `positions`
                positions[bid * draft_token_num + tid] = position_depth + seq_len




import torch
from typing import List

def reconstruct_tree_lineage(
    # --- Final Tree Structure (Stage 3) ---
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    positions: torch.Tensor,
    final_draft_tokens: torch.Tensor,
    
    # --- Preprocessed Data (Stage 2) ---
    top_scores_index: torch.Tensor,
    
    # --- Raw Draft Outputs (Stage 1) ---
    original_score_list: List[torch.Tensor],
    original_token_list: List[torch.Tensor],
    original_parents_list: List[torch.Tensor],
    
    # --- Other necessary context ---
    verified_id: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
):
    """
    Analyzes the final tree structure and maps each node back to its origin 
    in the raw draft outputs. This version corrects the indexing for the 2D
    tensors within `original_token_list`.
    """
    bs, num_verify_tokens = retrive_next_token.shape
    
    parent_counts = [s.shape[1] for s in original_score_list]
    cum_parent_counts = torch.cumsum(torch.tensor([0] + parent_counts), dim=0)

    print("="*80)
    print("Reconstructing Lineage for Each Node in the Final Verification Tree")
    print("="*80)

    def _dfs_traverse_and_map(b_idx: int, token_idx_in_draft: int, depth: int):
        prefix = "    " * depth
        
        global_token_idx = b_idx * num_verify_tokens + token_idx_in_draft
        token_id = final_draft_tokens[global_token_idx].item()
        position = positions[global_token_idx].item()
        
        print(f"{prefix}Node(final_idx={token_idx_in_draft}, final_token_id={token_id}, final_pos={position})")

        if token_idx_in_draft == 0:
            print(f"{prefix}  └─ Origin: Root of draft tree (last verified token).")
        else:
            flat_idx = top_scores_index[b_idx, token_idx_in_draft - 1].item()
            
            original_parent_group_idx = flat_idx // topk
            original_topk_choice = flat_idx % topk
            
            original_list_idx = torch.searchsorted(cum_parent_counts, original_parent_group_idx, right=True).item() - 1
            
            parent_idx_in_list = original_parent_group_idx - cum_parent_counts[original_list_idx]
            
            original_score = original_score_list[original_list_idx][b_idx, parent_idx_in_list, original_topk_choice].item()
            
            # --- FIX IS HERE ---
            # Calculate the flattened index for the 2D token tensor
            flat_token_idx = parent_idx_in_list * topk + original_topk_choice
            original_token = original_token_list[original_list_idx][b_idx, flat_token_idx].item()
            # --- END FIX ---
            
            original_parent_global_idx = original_parents_list[original_list_idx][b_idx, parent_idx_in_list].item()

            print(f"{prefix}  └─ Origin Details:")
            print(f"{prefix}      - Flat Index from top_scores_index: {flat_idx}")
            print(f"{prefix}      - Original Draft Step (list_idx): {original_list_idx}")
            print(f"{prefix}      - Original Parent's Index in its step (parent_idx_in_list): {parent_idx_in_list}")
            print(f"{prefix}      - Original Parent's Global Index: {original_parent_global_idx}")
            print(f"{prefix}      - Choice among topK (original_topk_choice): {original_topk_choice}")
            print(f"{prefix}      - Original Token ID: {original_token} (Matches final? {original_token == token_id})")
            print(f"{prefix}      - Original Score: {original_score:.4f}")

        child_idx = retrive_next_token[b_idx, token_idx_in_draft].item()
        
        while child_idx != -1:
            _dfs_traverse_and_map(b_idx, child_idx, depth + 1)
            child_idx = retrive_next_sibling[b_idx, child_idx].item()

    for b in range(bs):
        print(f"\n--- Analyzing Batch Item {b} (Original SeqLen: {seq_lens[b].item()}) ---")
        _dfs_traverse_and_map(b_idx=b, token_idx_in_draft=0, depth=0)
        print("-" * 55)

@pytest.mark.parametrize("device", ["cuda"])
def test_build_tree_kernel_efficient(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    verified_id = torch.tensor([29974, 13], device="cuda", dtype=torch.int32)
    score_list = [
        torch.tensor(
            [
                [[7.1127e-01, 2.8292e-01, 2.2995e-03, 1.7357e-03]],
                [[9.7476e-01, 2.2219e-02, 6.5031e-04, 1.3212e-04]],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    [6.9142e-01, 1.2863e-02, 1.6873e-03, 1.1871e-03],
                    [2.4787e-01, 1.8818e-02, 1.4204e-02, 9.2235e-04],
                    [2.2971e-03, 1.6700e-06, 1.8737e-07, 8.3146e-08],
                    [1.2771e-03, 2.4374e-04, 1.7832e-04, 1.1947e-05],
                ],
                [
                    [8.4832e-02, 6.6068e-02, 5.8304e-02, 5.7851e-02],
                    [2.3616e-03, 1.1243e-03, 5.4368e-04, 2.7768e-04],
                    [2.5286e-04, 1.5578e-04, 2.8817e-05, 1.2888e-05],
                    [1.2834e-04, 2.5417e-06, 1.1279e-06, 1.6088e-08],
                ],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    [6.6438e-01, 2.6997e-02, 2.4236e-05, 4.0821e-06],
                    [2.4402e-01, 2.8409e-03, 5.0935e-04, 2.9022e-04],
                    [1.6178e-02, 2.0567e-03, 4.5892e-04, 3.0034e-05],
                    [1.3023e-02, 5.0497e-04, 3.6371e-04, 8.7750e-05],
                ],
                [
                    [2.3263e-02, 2.0054e-02, 9.3990e-03, 2.7783e-03],
                    [6.4156e-02, 5.5506e-04, 1.0429e-04, 9.7211e-05],
                    [4.9950e-02, 5.0630e-03, 9.0068e-04, 3.3656e-04],
                    [7.5817e-03, 8.5731e-04, 6.9972e-04, 6.0793e-04],
                ],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    [6.6420e-01, 1.0525e-04, 6.5864e-05, 1.2253e-06],
                    [1.3019e-01, 1.0461e-01, 5.2083e-03, 1.6777e-03],
                    [2.0103e-02, 6.7335e-03, 1.2625e-04, 1.0364e-05],
                    [1.5142e-02, 7.0819e-04, 9.6595e-05, 8.7951e-05],
                ],
                [
                    [5.8608e-02, 1.8840e-03, 7.8535e-04, 4.4400e-04],
                    [1.2185e-02, 2.0684e-03, 1.7418e-03, 1.4327e-03],
                    [6.2455e-03, 6.1487e-03, 2.6862e-03, 1.8034e-03],
                    [1.8590e-03, 1.6151e-03, 1.2481e-03, 3.6038e-04],
                ],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
    ]
    token_list = [
        torch.tensor(
            [[29896, 29906, 29900, 29945], [13, 2, 29871, 28956]],
            dtype=torch.int64,
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    29889,
                    29974,
                    29945,
                    29900,
                    29974,
                    29922,
                    29930,
                    29958,
                    29889,
                    29974,
                    29930,
                    29945,
                    29974,
                    29922,
                    29930,
                    29958,
                ],
                [
                    22550,
                    4136,
                    16492,
                    8439,
                    29871,
                    2,
                    3001,
                    13,
                    2,
                    13,
                    29906,
                    29946,
                    2,
                    13,
                    29871,
                    259,
                ],
            ],
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    29946,
                    29945,
                    29953,
                    29906,
                    29896,
                    29945,
                    29900,
                    29906,
                    29896,
                    29945,
                    29906,
                    29953,
                    29896,
                    29945,
                    29906,
                    29946,
                ],
                [
                    29871,
                    2,
                    29901,
                    29889,
                    29871,
                    2,
                    395,
                    259,
                    29901,
                    29871,
                    2,
                    29889,
                    3001,
                    1234,
                    7146,
                    2186,
                ],
            ],
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    29946,
                    29974,
                    29945,
                    29930,
                    29889,
                    29922,
                    29974,
                    29930,
                    29974,
                    29946,
                    29930,
                    29922,
                    29889,
                    29974,
                    29945,
                    29922,
                ],
                [
                    29941,
                    29906,
                    2,
                    29946,
                    29871,
                    450,
                    319,
                    14990,
                    29946,
                    29941,
                    2,
                    29906,
                    29871,
                    2,
                    3001,
                    13,
                ],
            ],
            device="cuda",
        ),
    ]
    parents_list = [
        torch.tensor(
            [[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]], dtype=torch.int64, device="cuda"
        ),
        torch.tensor([[4, 8, 9, 10], [4, 5, 6, 7]], dtype=torch.int64, device="cuda"),
        torch.tensor(
            [[20, 24, 21, 28], [24, 28, 20, 21]], dtype=torch.int64, device="cuda"
        ),
        torch.tensor(
            [[36, 40, 41, 44], [36, 40, 44, 45]], dtype=torch.int64, device="cuda"
        ),
    ]
    seq_lens = torch.tensor([5, 100], dtype=torch.int64, device="cuda")
    # seq_lens = torch.tensor([5, 10], dtype=torch.int64, device="cuda")
    topk = 4
    depth = 4
    num_draft_token = 8


    # first_rank_print("=========== build tree kernel inputs ==========")
    # first_rank_print(f"{parents_list=}", flush=True)

    (
        tree_mask,
        position,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    ) = build_tree_kernel_efficient(
        verified_id=verified_id,
        score_list=score_list,
        token_list=token_list,
        parents_list=parents_list,
        seq_lens=seq_lens,
        seq_lens_sum=torch.sum(seq_lens).item(),
        topk=topk,
        spec_steps=depth,
        num_verify_tokens=num_draft_token,
    )


    first_rank_print("=========== build tree kernel efficient ==========")
    # first_rank_print(f"{tree_mask=}", flush=True)
    
    first_rank_print(f"{position=}", flush=True)
    first_rank_print(f"{retrive_index=}", flush=True)
    first_rank_print(f"{retrive_next_token=}", flush=True)
    first_rank_print(f"{retrive_next_sibling=}", flush=True)
    first_rank_print(f"{draft_tokens=}", flush=True)
    assert position.tolist() == [5, 6, 6, 7, 7, 8, 8, 9, 100, 101, 102, 102, 102, 102, 103, 104]
    # assert position.tolist() == [5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14]
    assert retrive_index.tolist() == [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
    ]
    assert retrive_next_token.tolist() == [
        [1, 3, 4, 5, 6, 7, -1, -1],
        [1, 2, -1, 6, -1, -1, 7, -1],
    ]
    assert retrive_next_sibling.tolist() == [
        [-1, 2, -1, -1, -1, -1, -1, -1],
        [-1, -1, 3, 4, 5, -1, -1, -1],
    ]
    assert draft_tokens.tolist() == [
        29974,
        29896,
        29906,
        29889,
        29974,
        29946,
        29896,
        29946,
        13,
        13,
        22550,
        4136,
        16492,
        8439,
        29871,
        29941,
    ]


if __name__ == "__main__":
    test_build_tree_kernel_efficient()
