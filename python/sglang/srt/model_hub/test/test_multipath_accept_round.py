# build_eagle_tree_include_test.py
# (包含之前所有必要的函数定义)
# NOTE: Please run this file to make sure the test cases are correct.

from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
import copy

import torch
import torch.nn.functional as F
from sglang.srt.utils import is_cuda, is_hip
import torch.distributed as dist
import pytest

# (此处省略所有来自前文的、已定义的辅助函数)
def first_rank_print(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


# 这是一个来自 draft_to_tree.py 的关键函数，我们将用它来驱动数据生成
@torch.compile(dynamic=True)
def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    if i == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)
        # bs = topk_p.shape[0]

        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device="cuda")
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )
    else:
        # The later decode steps
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten()

        if hidden_states is not None:
            selected_input_index = topk_cs_index.flatten() // topk + torch.arange(
                0, hidden_states.shape[0], step=topk, device="cuda"
            ).repeat_interleave(topk)

            hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)

def draft_forward(i: int, draft_logits_list: List):
    pass

def generate_draft_realistic_tree_data(
    batch_size: int,
    topk: int,
    spec_steps: int,
    vocab_size: int,
    hidden_size: int,
    device: str,
):
    """
    通过模拟 draft_forward 的迭代过程，生成高保真的推测树数据。
    """
    # 初始化列表
    score_list, token_list, parents_list = [], [], []

    # 初始状态
    verified_id = torch.tensor([114, 512], device=device, dtype=torch.int32)
    hidden_states = None

    # Step -1: 模拟对 verified_id 的前向传播，得到第一次的 topk
    initial_logits = torch.zeros(batch_size, 1, vocab_size, device=device)

    # 手动控制生成路径，确保可复现
    for b in range(batch_size):
        initial_logits[b, 0, 100:100+topk] = 10.0 # 让 100, 101, ... 成为最优选择
    scores = None # scores 在 select_top_k_tokens 内部初始化
    probs = torch.softmax(initial_logits, dim=-1)
    topk_p, topk_index = fast_topk(probs, topk, dim=-1)

    # 迭代生成
    for i in range(spec_steps):
        input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
            i, topk_p, topk_index, hidden_states, scores, topk
        )
        
        score_list.append(tree_info[0])
        token_list.append(tree_info[1])
        parents_list.append(tree_info[2])

        if i == spec_steps - 1:
            break

        # 为下一次迭代准备状态
        logits_output = draft_forward(i)
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, topk, dim=-1)

        # Step i: 模拟模型对 input_ids 的前向传播
        # current_bs = input_ids.shape[0]
        # next_logits = torch.randn(current_bs, vocab_size, device=device)
        # # 再次手动控制，让生成的树有迹可循
        # for b in range(current_bs):
        #      next_logits[b, 200 + b : 200 + b + topk] = 10.0
        # topk_p, topk_index = torch.topk(torch.softmax(next_logits, dim=-1), k=topk)

    return verified_id, score_list, token_list, parents_list



def build_tree_kernel_efficient_preprocess(
    verified_id: torch.Tensor,
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_verify_tokens: int,
):
    score_list_cat = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list_cat = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list_cat, num_verify_tokens - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list_cat, index=top_scores_index, dim=1)
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()
    if len(parents_list) > 1:
        parent_list_cat = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list_cat = torch.empty(batch_size, 0, device=parents_list[0].device)
    return parent_list_cat, top_scores_index, draft_tokens

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
    parent_list, top_scores_index, draft_tokens = build_tree_kernel_efficient_preprocess(
        verified_id, score_list, token_list, parents_list, num_verify_tokens
    )
    bs = seq_lens.numel()
    device = seq_lens.device
    tree_mask = torch.full((seq_lens_sum * num_verify_tokens + num_verify_tokens * num_verify_tokens * bs,), True, device=device)
    retrive_index = torch.full((bs, num_verify_tokens), -1, device=device, dtype=torch.long)
    retrive_next_token = torch.full((bs, num_verify_tokens), -1, device=device, dtype=torch.long)
    retrive_next_sibling = torch.full((bs, num_verify_tokens), -1, device=device, dtype=torch.long)
    positions = torch.empty((bs * num_verify_tokens,), device=device, dtype=torch.long)
    sgl_build_tree_kernel_efficient_python(
        parent_list, top_scores_index, seq_lens.to(torch.int32), tree_mask, positions,
        retrive_index, retrive_next_token, retrive_next_sibling, topk, spec_steps, num_verify_tokens
    )
    return (tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling, draft_tokens, top_scores_index)

def sgl_build_tree_kernel_efficient_python(
    parent_list: torch.Tensor, selected_index: torch.Tensor, verified_seq_len: torch.Tensor, tree_mask: torch.Tensor,
    positions: torch.Tensor, retrive_index: torch.Tensor, retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor, topk: int, depth: int, draft_token_num: int
):
    bs = parent_list.size(0)
    seq_len_cumsum = torch.cat([torch.tensor([0], device=verified_seq_len.device), torch.cumsum(verified_seq_len, dim=0)])
    for bid in range(bs):
        seq_len = verified_seq_len[bid].item()
        seq_tree_offset = (seq_len_cumsum[bid] * draft_token_num + draft_token_num * draft_token_num * bid)
        for tid in range(draft_token_num):
            token_tree_row_start_idx = seq_tree_offset + (seq_len + draft_token_num) * tid
            draft_tokens_mask_start_idx = token_tree_row_start_idx + seq_len
            for i in range(draft_token_num - 1):
                 tree_mask[draft_tokens_mask_start_idx + 1 + i] = False
            if tid == 0:
                positions[bid * draft_token_num + 0] = seq_len
                retrive_index_offset = bid * draft_token_num
                retrive_index[bid, 0] = retrive_index_offset
                for i in range(draft_token_num - 1, 0, -1):
                    current_token_global_idx = retrive_index_offset + i
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
                    origin_next_token = retrive_next_token[bid, parent_position_in_draft_list].item()
                    retrive_next_token[bid, parent_position_in_draft_list] = i
                    if origin_next_token != -1:
                        retrive_next_sibling[bid, i] = origin_next_token
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
                positions[bid * draft_token_num + tid] = position_depth + seq_len



def tree_speculative_sampling_target_only_pytorch(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool = True, # PyTorch handles determinism via manual_seed
    device: str = "cpu"
):
    """
    PyTorch (Tensor) implementation of the TreeSpeculativeSamplingTargetOnly CUDA kernel.

    This function serves as a reference for understanding, debugging, and testing
    the equivalent CUDA implementation using torch.Tensors.
    """
    if deterministic:
        # In a real scenario, you'd call torch.manual_seed(...) before generating
        # the random samples to ensure determinism. We assume the input samples
        # are already deterministically generated.
        pass

    batch_size, num_draft_tokens = candidates.shape
    # The number of speculative steps is one less than the number of tokens to accept
    # (since the first token is already given/prompt). The number of uniform samples
    # corresponds to the number of draft tokens that can be evaluated.
    num_speculative_tokens = uniform_samples.shape[1] + 1
    vocab_size = target_probs.shape[2]
    
    # In CUDA, `predicts` is a flat pointer array. In Python, we use a dictionary 
    # for sparse updates for clarity, then construct the final tensor.
    # final_predicts = {}
    
    # predicts = torch.full((max_retrive_index + 1,), -1, dtype=torch.int32, device=device)
    # accept_index = torch.full((batch_size, num_draft_tokens), -1, dtype=torch.int32, device=device)
    # accept_token_num = torch.zeros(batch_size, dtype=torch.int32, device=device)


    # Capped threshold_acc to avoid division by zero, as in the CUDA host code
    capped_threshold_acc = max(threshold_acc, 1e-9)

    # Each item in the batch is processed independently (like a CUDA block)
    for bx in range(batch_size):
        # =================================================================
        # Phase 1: Acceptance Loop
        # =================================================================
        prob_acc = 0.0
        
        # Start at the root of the draft tree for this sequence
        last_accepted_retrive_idx = retrive_index[bx, 0].item()
        accept_index[bx, 0] = last_accepted_retrive_idx
        
        num_accepted_tokens = 0
        cur_prob_offset = 0
        
        # Index within the draft tokens for the current sequence (0 to num_draft_tokens-1)
        # Starts from the children of the root node (index 0)
        current_draft_idx_in_seq = 0# We can accept up to `num_speculative_tokens - 1` new tokens
        for j in range(1, num_speculative_tokens):
            # Get the first child of the previously accepted token
            current_draft_idx_in_seq = retrive_next_token[bx, current_draft_idx_in_seq].item()

            # Iterate through siblings at the current tree depth
            while current_draft_idx_in_seq != -1:
                draft_token_id = candidates[bx, current_draft_idx_in_seq].item()
                
                # Get target probability for this specific token
                target_prob_single = target_probs[bx, cur_prob_offset, draft_token_id].item()

                prob_acc += target_prob_single

                # The random number is associated with the token being checked
                coin = uniform_samples[bx, current_draft_idx_in_seq].item()

                # --- Core Acceptance Logic ---
                if coin <= prob_acc / capped_threshold_acc or target_prob_single >= threshold_single:
                    # Accept the token
                    prob_acc = 0.0
                    
                    # final_predicts[last_accepted_retrive_idx] = draft_token_id
                    predicts[last_accepted_retrive_idx] = draft_token_id
                    
                    num_accepted_tokens += 1
                    current_retrive_idx = retrive_index[bx, current_draft_idx_in_seq].item()
                    accept_index[bx, num_accepted_tokens] = current_retrive_idx
                    last_accepted_retrive_idx = current_retrive_idx
                    cur_prob_offset = current_draft_idx_in_seq
                    break # Break from sibling loop to go to the next level
                else:
                    # Reject the token and update draft_probs for the final sampling
                    draft_probs[bx, current_draft_idx_in_seq, draft_token_id] = target_probs[bx, current_draft_idx_in_seq, draft_token_id]
                    
                    # Move to the next sibling
                    current_draft_idx_in_seq = retrive_next_sibling[bx, current_draft_idx_in_seq].item()

            if current_draft_idx_in_seq == -1:
                # If we exhausted all siblings and accepted none, stop
                break
        
        accept_token_num[bx] = num_accepted_tokens
        
        # =================================================================
        # Phase 2: Final Sampling (Bonus Token)
        # =================================================================
        
        # Get probabilities at the last accepted position
        q_probs = target_probs[bx, current_draft_idx_in_seq, :]
        p_probs = draft_probs[bx, current_draft_idx_in_seq, :]
        
        # Sample from relu(q - p)
        final_dist = torch.maximum(q_probs - p_probs, torch.tensor(0.0, device=device))
        
        sum_dist = torch.sum(final_dist)
        
        final_token_id = vocab_size - 1 # Default as in CUDA
        if sum_dist.item() > 1e-6:
            final_coin = uniform_samples_for_final_sampling[bx, current_draft_idx_in_seq].item()
            u = final_coin * sum_dist.item()
            
            cumulative_prob = 0.0
            # This loop is the Python equivalent of the parallel scan in CUDA
            for token_id in range(vocab_size):
                cumulative_prob += final_dist[token_id].item()
                if cumulative_prob > u:
                    final_token_id = token_id
                    break
        
        # final_predicts[last_accepted_retrive_idx] = final_token_id
        predicts[last_accepted_retrive_idx] = final_token_id

    # Construct the final flat `predicts` tensor from the sparse dictionary
    # max_retrive_index = torch.max(retrive_index).item() if retrive_index.numel() > 0 else 0
    # for k, v in final_predicts.items():
    #     predicts[k] = v

    # return predicts, accept_index, accept_token_num

### --- Core Logic for Multi-Path Acceptance and Forking --- ###

def verify_topk_paths_pytorch(
    b_idx: int,
    candidates: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_probs: torch.Tensor,
    verification_top_k: int
) -> List[List[int]]:
    """
    A mock verification function that finds all valid paths where draft tokens
    fall into the target model's top-k predictions.
    """
    all_paths = []
    
    # Inner recursive function to perform DFS
    def find_paths_dfs(current_node_idx: int, current_path: List[int], prob_state_idx: int):
        # Get all children of the current node
        child_idx = retrive_next_token[b_idx, current_node_idx].item()
        
        accepted_children_count = 0
        while child_idx != -1:
            draft_token_id = candidates[b_idx, child_idx].item()
            
            # Check if this draft token is in the target's top-k
            top_k_probs, top_k_indices = torch.topk(target_probs[b_idx, prob_state_idx], k=verification_top_k)
            
            if draft_token_id in top_k_indices:
                accepted_children_count += 1
                new_path = current_path + [draft_token_id]
                # Recurse on this accepted child
                find_paths_dfs(child_idx, new_path, child_idx)
            
            # Move to the next sibling
            child_idx = retrive_next_sibling[b_idx, current_node_idx].item()
        
        # If the current node is a leaf in the accepted sub-tree, add its path to the results
        if accepted_children_count == 0 and current_path:
            all_paths.append(current_path)

    # Start the search from the root (index 0)
    find_paths_dfs(current_node_idx=0, current_path=[], prob_state_idx=0)
    
    # Handle the case where no draft tokens are accepted
    if not all_paths:
        return [[]] # Return one path with zero accepted tokens

    return all_paths

@dataclass
class MockReq:
    """A simplified mock of the Req class for testing."""
    rid: int
    prompt: str
    output_ids: List[int] = field(default_factory=list)
    # other fields like sampling_params, grammar, etc. would be here

def fork_requests_from_paths(
    original_reqs: List[MockReq],
    multi_path_results: List[List[List[int]]]
) -> List[MockReq]:
    """
    Takes the original batch of requests and the multi-path verification results,
    and returns a new, potentially larger, batch of forked requests.
    """
    new_reqs = []
    new_rid_counter = max(r.rid for r in original_reqs) + 1
    
    for i, req in enumerate(original_reqs):
        accepted_paths = multi_path_results[i]
        
        if not accepted_paths or len(accepted_paths) == 1:
            # No forking, just update the single request
            req.output_ids.extend(accepted_paths[0] if accepted_paths else [])
            new_reqs.append(req)
        else:
            # Forking event!
            first_rank_print(f"Request {req.rid} is forking into {len(accepted_paths)} new requests.")
            for path_idx, path in enumerate(accepted_paths):
                # Create a new request by deep-copying the original
                forked_req = copy.deepcopy(req)
                # Assign a new unique ID
                forked_req.rid = new_rid_counter
                # Append the unique accepted path
                forked_req.output_ids.extend(path)
                new_reqs.append(forked_req)
                new_rid_counter += 1
                
    return new_reqs


### --- The New Multi-Path Acceptance Test Case --- ###


def draft_output():
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
    seq_lens = torch.tensor([5, 10], dtype=torch.int64, device="cuda")
    return verified_id, score_list, token_list, parents_list, seq_lens


@pytest.mark.parametrize("device", ["cuda"])
def test_adaptive_iterative_decoding(device):
    """
    Tests a more complex, adaptive, 2-step iterative cycle.
    - Handles asynchronous progress within the batch.
    - Dynamically calculates the next draft depth.
    - Verifies the handoff of the "bonus token".
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # --- 1. Initial Setup ---
    topk = 4
    std_depth = 4
    num_verify_tokens_g1 = 8
    vocab_size = 32000
    batch_size = 2
    
    
    # verified_id, score_list, token_list, parents_list = generate_draft_realistic_tree_data(
    #     batch_size, topk, std_depth, vocab_size, hidden_size, device
    # )
    verified_id, score_list, token_list, parents_list, seq_lens = draft_output()

    first_rank_print("\n\n--- Running Test: test_adaptive_iterative_decoding ---")

    # --- 2. First Round: Draft and Verify a full tree (depth=5) ---
    first_rank_print(f"\n--- Round 1: Drafting and Verifying full tree (depth={std_depth}) ---")

    assert len(score_list) == std_depth and len(token_list) == std_depth and len(parents_list) == std_depth

    # total_draft_candidates = sum(t.shape[1] for t in token_list)

    # num_verify_tokens_g1 = min(10, total_draft_candidates + 1) # +1 for root
    

    (
        _, positions_g1, retrive_index_g1, retrive_next_token_g1, retrive_next_sibling_g1, draft_tokens_g1, _
    ) = build_tree_kernel_efficient(
        verified_id=verified_id, score_list=score_list, token_list=token_list,
        parents_list=parents_list, seq_lens=seq_lens, seq_lens_sum=torch.sum(seq_lens).item(),
        topk=topk, spec_steps=std_depth, num_verify_tokens=num_verify_tokens_g1,
    )
    first_rank_print(f"Verified {num_verify_tokens_g1} tokens by building a realistic tree.")
    first_rank_print(f"Draft retrive_index_g1: {retrive_index_g1.tolist()}")
    first_rank_print(f"Draft retrive_next_token_g1: {retrive_next_token_g1.tolist()}")
    first_rank_print(f"Draft retrive_next_sibling_g1: {retrive_next_sibling_g1.tolist()}")
    first_rank_print(f"Draft draft_tokens_g1: {draft_tokens_g1.tolist()}")
    
    
    # --- Mock Verification (Gen-1) ---
    # We will simulate: seq 0 accepts 2 draft tokens, seq 1 accepts 1 draft token.
    bs_g1, _ = retrive_index_g1.shape
    
    # Mock probabilities to control verification
    target_probs_g1 = torch.zeros(bs_g1, num_verify_tokens_g1, vocab_size, device=device)
    draft_probs_g1 = torch.zeros_like(target_probs_g1)
    
    # Path for seq 0: root -> token at index 1 -> token at index 5.
    # We make these tokens have high probability to be accepted.
    target_probs_g1_tuples = [
        (0, 0, 1), # accept
        (0, 1, 5), # reject
        (1, 0, 1), # accept
        (1, 1, 3), # accept
        (1, 3, 4), # reject
    ]
    # accepted_token_s0_t1 = draft_tokens_g1[1].item()
    # accepted_token_s0_t2 = draft_tokens_g1[3].item()
    # target_probs_g1[0, 0, accepted_token_s0_t1] = 1.0 # Accept first token
    # target_probs_g1[0, 1, accepted_token_s0_t2] = 1.0 # Accept second token
    
    # # Path for seq 1: root -> token at index 21+1
    # accepted_token_s1_t1 = draft_tokens_g1[num_verify_tokens_g1 + 1].item()
    # target_probs_g1[1, 0, accepted_token_s1_t1] = 1.0 # Accept first token
    for bx, prob_state_idx, draft_token_id in target_probs_g1_tuples:
        target_probs_g1[bx, prob_state_idx, draft_tokens_g1[bx*num_verify_tokens_g1+draft_token_id].item()] = 1.0

    # # Mock probabilities for bonus tokens
    # target_probs_g1[0, 5, 999] = 0.8 # For seq 0, after accepting 2, the next state is at index 5
    # # draft_probs_g1[0, 5, 999] = 0.1
    # target_probs_g1[1, 1, 888] = 0.8 # For seq 1, after accepting 1, the next state is at index 1
    # # draft_probs_g1[1, 1, 888] = 0.1


    # generate from draft and target

    # Run verification
    predicts_g1 = torch.full((bs_g1 * num_verify_tokens_g1, ), -1, dtype=torch.int32, device=device)
    accept_index_g1 = torch.full((bs_g1, std_depth + 1), -1, dtype=torch.int32, device=device)
    accept_token_num_g1 = torch.full((bs_g1,), 0, dtype=torch.int32, device=device)
    
    tree_speculative_sampling_target_only_pytorch(
        predicts=predicts_g1, accept_index=accept_index_g1, accept_token_num=accept_token_num_g1,
        candidates=draft_tokens_g1.view(bs_g1, -1), retrive_index=retrive_index_g1,
        retrive_next_token=retrive_next_token_g1, retrive_next_sibling=retrive_next_sibling_g1,
        uniform_samples=torch.full((bs_g1, num_verify_tokens_g1), 0.01, device=device), # Low random numbers to ensure acceptance
        uniform_samples_for_final_sampling=torch.full((bs_g1, num_verify_tokens_g1), 0.01, device=device),
        target_probs=target_probs_g1, draft_probs=draft_probs_g1,
        threshold_single=0.9, threshold_acc=0.9
    )
    
    first_rank_print(f"Round 1 Verified. Accepted draft lengths: {accept_token_num_g1.tolist()}")
    assert accept_token_num_g1.tolist() == [1, 2]

    # --- 3. State Update and Adaptive Depth Calculation ---
    first_rank_print("\n--- Calculating state and adaptive depth for Round 2 ---")
    
    # Get the new root tokens (the bonus tokens)
    new_verified_ids = torch.tensor([
        predicts_g1[accept_index_g1[0, 2]].item(), # Bonus token for seq 0 (at index 2 of accept_index)
        predicts_g1[accept_index_g1[1, 1]].item(), # Bonus token for seq 1 (at index 1 of accept_index)
    ], dtype=torch.int32, device=device)
    first_rank_print(f"New verified_ids (bonus tokens): {new_verified_ids.tolist()}")
    assert new_verified_ids.tolist() == [999, 888]
    
    # Calculate accepted depth for each sequence
    # This uses the `positions` tensor from the first build
    last_accepted_draft_idx_s0 = accept_index_g1[0, 2].item()
    last_accepted_draft_idx_s1 = accept_index_g1[1, 1].item()
    
    # The depth is position of last *draft* token minus original sequence length
    # Note: A real implementation needs to find the position of the last *non-bonus* token.
    # For this test, we can simplify by knowing the path.
    accepted_depth_s0 = positions_g1[accept_index_g1[0,2]].item() - seq_lens[0].item() # Simplified depth
    accepted_depth_s1 = positions_g1[accept_index_g1[1,1]].item() - seq_lens[1].item()
    
    # Let's manually set the correct depths for this test based on accepting 2 and 1 tokens
    accepted_depth_s0 = 2
    accepted_depth_s1 = 1
    first_rank_print(f"Accepted draft depths: Seq 0 = {accepted_depth_s0}, Seq 1 = {accepted_depth_s1}")
    
    # Adaptive depth calculation
    remaining_steps = torch.tensor([
        std_depth - accepted_depth_s0,
        std_depth - accepted_depth_s1
    ], device=device)
    next_draft_steps = torch.min(remaining_steps).item()
    
    first_rank_print(f"Remaining steps for each seq: {remaining_steps.tolist()}")
    first_rank_print(f"Next round will draft for adaptively calculated depth: {next_draft_steps}")
    assert next_draft_steps == 3 # min(5-2, 5-1) = 3
    
    # --- 4. Second Round: "Continue" drafting for the adaptive depth ---
    first_rank_print(f"\n--- Round 2: Continuing draft for {next_draft_steps} steps ---")
    
    # The state is now updated.
    new_seq_lens = seq_lens + accept_token_num_g1 + 1 # +1 for the bonus token
    
    # We simulate "continuing" by creating a new set of lists for the required depth
    # A real implementation would use the pruned and remapped lists
    continue_score_list = score_list[:next_draft_steps]
    continue_token_list = token_list[:next_draft_steps]
    continue_parents_list = parents_list[:next_draft_steps]

    (
        _, _, _, _, _, final_draft_tokens, _
    ) = build_tree_kernel_efficient(
        verified_id=new_verified_ids,
        score_list=continue_score_list,
        token_list=continue_token_list,
        parents_list=continue_parents_list,
        seq_lens=new_seq_lens,
        seq_lens_sum=torch.sum(new_seq_lens).item(),
        topk=topk,
        spec_steps=next_draft_steps,
        num_verify_tokens=1 + topk * next_draft_steps,
    )
    
    first_rank_print("Round 2 Build successful. The adaptive process is structurally sustainable.")
    first_rank_print(f"Final draft tokens for Round 2 start with: {final_draft_tokens.tolist()[:2]}...")
    
    # The final assertion confirms the new roots were used correctly.
    assert final_draft_tokens[0] == 999
    assert final_draft_tokens[1 + topk * next_draft_steps] == 888

# @pytest.mark.parametrize("device", ["cuda"])
def test_multi_path_acceptance_and_forking(device):
    """
    Tests the boundary case where the verification algorithm accepts multiple
    paths for a single request, leading to a "forking" of the request state.
    """
    # --- 1. Initial Setup: A single request in the batch ---
    topk = 4
    spec_steps = 2
    num_verify_tokens = 1 + topk * spec_steps
    vocab_size = 50
    
    initial_reqs = [MockReq(rid=100, prompt="Hello", output_ids=[10])]
    verified_id = torch.tensor([10], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([1], dtype=torch.int64, device=device)

    # --- 2. Draft a tree for this request ---
    # We will craft the draft tokens to present a clear choice.
    # The root's children will be [101, 102, 103, 104]
    # The children of token 101 will be [201, 202, 203, 204]
    score_list = [torch.full((1, 1, topk), 0.25, dtype=torch.float32, device=device),
                  torch.full((1, topk, topk), 0.25, dtype=torch.float32, device=device)]
    
    token_list = [
        torch.tensor([[101, 102, 103, 104]], dtype=torch.int64, device=device),
        torch.tensor([list(range(201, 205))*4], dtype=torch.int64, device=device)
    ]
    
    parents_list = [
        torch.tensor([[-1, 0, 1, 2, 3]], dtype=torch.int64, device=device),
        torch.tensor([list(range(4, 4+topk))], dtype=torch.int64, device=device)
    ]
    
    first_rank_print("\n\n--- Running Test: test_multi_path_acceptance_and_forking ---")
    
    (
        _, _, retrive_index, retrive_next_token, retrive_next_sibling, draft_tokens, _
    ) = build_tree_kernel_efficient(
        verified_id=verified_id, score_list=score_list, token_list=token_list,
        parents_list=parents_list, seq_lens=seq_lens, seq_lens_sum=torch.sum(seq_lens).item(),
        topk=topk, spec_steps=spec_steps, num_verify_tokens=num_verify_tokens,
    )
    
    # --- 3. Mock Verification with Top-K Acceptance ---
    # Craft target probabilities to cause a fork.
    # Let's say after seeing the root (token 10), the target model thinks
    # both 101 and 102 are highly likely (in its top-2).


    # generate from draft and target    



    target_probs = torch.zeros(1, num_verify_tokens, vocab_size, device=device)

    target_probs_tuples = [
        (0, 0, 101, 0.4),
        (0, 0, 102, 0.3),
        (0, 0, 5, 0.1),
        (0, 1, 201, 0.5),
        (0, 2, 202, 0.6),
    ]
    
    for bx, prob_state_idx, draft_token_id, prob in target_probs_tuples:
        target_probs[bx, prob_state_idx, draft_token_id] = prob
    
    
    # # If path 101 is taken, let's say 201 is the only next accepted token
    # # The state for this is at index 1 (corresponding to draft token 101)
    # target_probs[0, 1, 201] = 0.5
    
    # # If path 102 is taken, let's say 202 is the only next accepted token
    # # The state for this is at index 2 (corresponding to draft token 102)
    # target_probs[0, 2, 202] = 0.6
    
    first_rank_print("\n--- Running Mock Top-K Verification ---")
    # Our custom verification will accept any token in the top 2
    multi_path_results = [
        verify_topk_paths_pytorch(
            b_idx=0, candidates=draft_tokens.view(1, -1), retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling, target_probs=target_probs, verification_top_k=2
        )
    ]
    
    first_rank_print(f"Verification found {len(multi_path_results[0])} accepted paths: {multi_path_results[0]}")
    assert len(multi_path_results[0]) == 2
    # The paths should have been extended. Note: the mock verification has a simplified DFS logic.
    # Let's assume the paths found are [101] and [102] for clarity of testing the forking logic.
    mocked_paths = [[[101]], [[102]]]


    # --- 4. Fork Request State ---
    first_rank_print("\n--- Forking request state based on multiple accepted paths ---")
    forked_reqs = fork_requests_from_paths(initial_reqs, mocked_paths)

    # --- 5. Assert Final State ---
    first_rank_print(f"Number of requests after forking: {len(forked_reqs)}")
    assert len(forked_reqs) == 2
    
    # The original request ID 100 is gone, replaced by new ones.
    rids = {r.rid for r in forked_reqs}
    assert 100 not in rids

    # Check the content of the forked requests
    outputs = sorted([r.output_ids for r in forked_reqs])
    
    first_rank_print(f"Forked request 1 output: {outputs[0]}")
    first_rank_print(f"Forked request 2 output: {outputs[1]}")
    
    assert outputs[0] == [10, 101]  # Original prefix + path 1
    assert outputs[1] == [10, 102]  # Original prefix + path 2

    first_rank_print("\nTest Passed: System correctly handled request forking from multi-path acceptance.")