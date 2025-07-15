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
        hidden_states = hidden_states.repeat_interleave(topk, dim=0) if hidden_states is not None else None
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


def target_forward(i: int):

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



# # mock
# #【新增】用于模拟的辅助函数，实际应为更复杂的实现
# def mock_get_nodes_at_layer(tree, layer):
#     # 这是一个模拟函数，根据树的结构返回指定层的节点
#     # 在真实测试中，您需要根据 tree 的数据结构来实现它
#     print(f"    (MOCK) Getting nodes for layer {layer}...")
#     # 简化返回，假设我们能拿到节点的索引和累积概率
#     return [(f"node_{layer}_{i}", 0.1 * (5-i)) for i in range(4)] # (node_id, cumulative_prob)

# def mock_compete_k(tree, target_model, layer, topk):
#     # 模拟目标模型评估和竞争过程
#     print(f"    (MOCK) Competing for best {topk} nodes at layer {layer} based on target model...")
#     # 简单返回一组“新”的、被目标模型认可的节点
#     # 注意：为了测试'RE-DRAFT'路径，我们在第0层故意制造不同
#     if layer == 0:
#         return [(f"new_node_{layer}_{i}", 0.12 * (5-i)) for i in range(4)]
#     # 后续层假设与原始草稿一致
#     else:
#         return [(f"node_{layer}_{i}", 0.11 * (5-i)) for i in range(4)]

# def mock_calculate_overlap_ratio(old_nodes, new_nodes, tree):
#     # 模拟计算保留度
#     old_node_ids = {n[0] for n in old_nodes}
#     new_node_ids = {n[0] for n in new_nodes}
#     overlap_ids = old_node_ids.intersection(new_node_ids)
    
#     # 简化概率计算
#     sum_prob_overlap = len(overlap_ids) * 0.1 
#     sum_prob_new = len(new_nodes) * 0.12
#     ratio = sum_prob_overlap / sum_prob_new if sum_prob_new > 0 else 0
#     print(f"    (MOCK) Calculating overlap. Old_nodes: {len(old_node_ids)}, New_nodes: {len(new_node_ids)}, Overlap: {len(overlap_ids)}, Ratio: {ratio:.2f}")
#     return ratio

# def mock_replace_and_truncate_tree(tree, layer, new_nodes):
#     # 模拟替换树的某一层并截断
#     print(f"    (MOCK) Evolving tree: Replacing layer {layer} and truncating.")
#     # 返回一个象征性的新树对象
#     return {"name": f"evolved_tree_at_layer_{layer}", "data": new_nodes}

# def mock_draft_model_complete_tree(partial_tree):
#     # 模拟草稿模型补全树
#     print(f"    (MOCK) Draft model completing the partially evolved tree...")
#     # 返回一个象征性的完整树对象
#     return {"name": "re-drafted_complete_tree", "data": partial_tree['data']}


# # # 【修改/实现】这是我们讨论的核心演进函数 tree_evolve_target_only_pytorch 的模拟版本
# # def tree_evolve_target_only_pytorch(
# #     draft_tree: object,
# #     target_model: object, # 在测试中，这只是一个象征性对象
# #     node_r_thresh: float,
# #     spec_num: int,
# #     topk: int
# # ) -> (object, str):
# #     """
# #     【MOCK IMPLEMENTATION】
# #     逐层验证并进化草稿树的模拟实现。
# #     """
# #     current_tree = draft_tree
    

# #     for i in range(spec_num):
# #         old_nodes = mock_get_nodes_at_layer(current_tree, i)
# #         new_nodes = mock_compete_k(current_tree, target_model, i, topk)
        
# #         overlap_ratio = mock_calculate_overlap_ratio(old_nodes, new_nodes, current_tree)

# #         if overlap_ratio < node_r_thresh:
# #             evolved_tree = mock_replace_and_truncate_tree(current_tree, i, new_nodes)
# #             return (evolved_tree, 'RE-DRAFT')

# #     # 如果所有层都通过了验证
# #     return (draft_tree, 'PROCEED_SAMPLING')

# def mock_redraft_from_evolved_tree(evolved_tree_info, base_verified_id, base_seq_lens):
#     print("    (MOCK) Draft model is re-drafting from an evolved tree base...")
#     # 在真实场景中，这里会用进化后的节点作为新的输入，让草稿模型推理后续节点
#     # 在测试中，我们直接返回一套全新的、可控的草稿数据
#     return draft_output(base_verified_id, [], [], [], base_seq_lens)




def draft_output(verified_id, score_list, token_list, parents_list, seq_lens):
    # mock
    verified_id = torch.tensor([29974, 13], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([5, 10], dtype=torch.int64, device="cuda")

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
    return verified_id, score_list, token_list, parents_list, seq_lens


def find_indices(tensor_a, tensor_b):
    """
    找到tensor_b中每个元素在tensor_a中的索引
    """
    # 扩展维度进行广播比较
    expanded_b = tensor_b.unsqueeze(-1)  # [batchsize, final_players, 1]
    expanded_a = tensor_a.unsqueeze(1)   # [batchsize, 1, k_candidates]
    
    # 找到匹配的位置
    matches = (expanded_b == expanded_a)  # [batchsize, final_players, k_candidates]
    
    # 获取索引
    indices = torch.argmax(matches.float(), dim=-1)
    
    return indices

def reconstruct_draft_probs(
    score_list: list[torch.Tensor],
    token_list: list[torch.Tensor],
    parents_list: list[torch.Tensor],
    selected_indices: torch.Tensor,
    vocab_size: int,
    num_verify_tokens: int,
    topk: int = 4,
    device: str = "cuda"
) -> torch.Tensor:
    """
    根据原始的草稿输出和最终选择的token索引，重建用于推测采样的draft_probs张量。

    Args:
        score_list: 来自草稿模型的原始概率列表。
        token_list: 来自草稿模型的原始token ID列表。
        selected_indices: 最终被选中用于验证的token的索引，形状为 (batch_size, num_verify_tokens)。
                          第一列是根节点（通常为0），后续列是草稿token的索引。
        vocab_size: 词汇表大小。
        num_verify_tokens: 验证token的总数（包括根节点）。
        device: 计算设备。

    Returns:
        一个形状为 (batch_size, num_verify_tokens, vocab_size) 的draft_probs张量。
    """
    # print("--- 开始重建 draft_probs 张量 ---")

    # --- 步骤 1: 数据扁平化 ---
    # 将分步的score_list连接并扁平化为 (batch_size, total_draft_candidates)
    score_list_cat = torch.cat([s.flatten(1) for s in score_list], dim=1)

    parents_list_cat = torch.cat([s.flatten(1) for s in parents_list], dim=1)
    
    # 将分步的token_list连接成 (batch_size, total_draft_candidates)
    # 注意：token_list的第二项及以后形状为(bs, topk*topk)，需要reshape
    token_list_reshaped = [token_list[0]] # 第一步的形状是 (bs, topk)
    for t in token_list[1:]:
        # t 的形状是 (bs, topk*topk) 或类似，是扁平的，可以直接用
        token_list_reshaped.append(t)
    token_list_cat = torch.cat(token_list_reshaped, dim=1)

    # print(f"扁平化后 score_list_cat 的形状: {score_list_cat.shape}")
    # print(f"扁平化后 token_list_cat 的形状: {token_list_cat.shape}")
    # print(f"扁平化后 parents_list_cat 的形状: {parents_list_cat}")

    score_list_token = score_list_cat.view(score_list_cat.shape[0], -1, topk)
    token_list_token = token_list_cat.view(token_list_cat.shape[0], -1, topk).to(torch.int64)

    # print(f"扁平化后 score_list_token 的形状: {score_list_token.shape}")
    # print(f"扁平化后 token_list_token 的形状: {token_list_token.shape}")
    batch_size, parents_size = parents_list_cat.shape

    # print(f"batch_size: {batch_size}, parents_size: {parents_size}")

    draft_probs_src = torch.zeros(batch_size, parents_size, vocab_size, device=device)
    draft_probs = torch.zeros(batch_size, num_verify_tokens, vocab_size, device=device)
    draft_probs_src.scatter_(2, token_list_token, score_list_token)
    # print(f"scatter_ draft_probs_src 的形状: {draft_probs_src.shape}")

    selected_prob_indices = torch.cat([torch.full((selected_indices.shape[0], 1), -1, device=device), selected_indices[:, :-1]], dim=1)

    map_indices = find_indices(parents_list_cat[:, :-topk], selected_prob_indices)
    # print(f"parents_list_cat:{parents_list_cat}, selected_prob_indices: {selected_prob_indices}, map_indices: {map_indices}")

    batch_indices = torch.arange(batch_size).unsqueeze(1)
    num_verify_indices = torch.arange(num_verify_tokens-1).unsqueeze(0)
    draft_probs[batch_indices, num_verify_indices] = draft_probs_src[batch_indices, map_indices]
    draft_probs[batch_indices, -1] = draft_probs_src[batch_indices, 0]

    

    # # --- 步骤 2: 提取选中项 ---
    # # 我们只关心草稿生成的token，所以忽略selected_indices的第一列（根节点）
    # selected_draft_indices = selected_indices[:, 1:]
    
    # # 使用gather根据索引提取出被选中的token ID
    # draft_tokens = torch.gather(token_list_cat, 1, selected_draft_indices)
    
    # # 使用gather根据索引提取出被选中的token的概率
    # draft_scores = torch.gather(score_list_cat, 1, selected_draft_indices)

    # print(f"提取出的 draft_tokens (形状: {draft_tokens.shape}):\n{draft_tokens.tolist()}")
    # print(f"提取出的 draft_scores (形状: {draft_scores.shape}):\n{draft_scores.tolist()}")

    # # --- 步骤 3 & 4: 构建索引并赋值 ---
    # batch_size = selected_indices.shape[0]
    
    # # 初始化目标张量
    # draft_probs = torch.zeros(batch_size, num_verify_tokens, vocab_size, device=device)

    # # 创建批次索引：[0, 0, ..., 1, 1, ...]
    # batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_verify_tokens - 1)
    
    # # 创建验证token索引：[1, 2, 3, ..., 1, 2, 3, ...]
    # # 我们从1开始，因为索引0对应的是根节点，它没有草稿概率
    # verify_idx = torch.arange(1, num_verify_tokens, device=device).repeat(batch_size)
    
    # # 创建词汇表索引，即被选中的token ID本身
    # token_idx = draft_tokens.flatten()

    # # 使用高级索引，将每个draft_scores中的值填入draft_probs的正确位置
    # # draft_probs[批次, 验证步数, token_id] = 该token的概率
    # draft_probs[batch_idx, verify_idx, token_idx] = draft_scores.flatten()
    
    # print("\n--- 重建完成 ---")
    # print(f"最终 draft_probs 张量的形状: {draft_probs.shape}")

    return draft_probs, token_list_token, score_list_token, map_indices

def target_output(target_probs_g1: torch.Tensor, draft_probs_g1:torch.Tensor, draft_tokens_g1: torch.Tensor, num_verify_tokens_g1: int, target_mode = "draft_score"):
    if target_mode == "custom_list":
        target_probs_g1_tuples = [
            (0, 0, 1, 0.6, "draft"), # accept
            (0, 0, 1, 0.8, False), # accept
            (0, 1, 5, ), # reject
            (0, 2, 1), # accept
            (1, 0, 1), # accept
            (1, 1, 3), # accept
            (1, 3, 4), # reject
            (1, )
        ]
        # accepted_token_s0_t1 = draft_tokens_g1[1].item()
        # accepted_token_s0_t2 = draft_tokens_g1[3].item()
        # target_probs_g1[0, 0, accepted_token_s0_t1] = 1.0 # Accept first token
        # target_probs_g1[0, 1, accepted_token_s0_t2] = 1.0 # Accept second token
        
        # # Path for seq 1: root -> token at index 21+1
        # accepted_token_s1_t1 = draft_tokens_g1[num_verify_tokens_g1 + 1].item()
        # target_probs_g1[1, 0, accepted_token_s1_t1] = 1.0 # Accept first token
        for bx, prob_state_idx, draft_token_id, score, is_direct_ids in target_probs_g1_tuples:
            if is_direct_ids:
                target_probs_g1[bx, prob_state_idx, draft_token_id] = score
            else:
                target_probs_g1[bx, prob_state_idx, draft_tokens_g1[bx*num_verify_tokens_g1+draft_token_id].item()] = score

    elif target_mode == "draft_score":
        target_probs_g1[:] = draft_probs_g1.clone()
    # return target_probs_g1

def print_layer_analysis(results, batch_idx=0):
    """
    打印层级分析结果
    """
    print(f"\n=== Layer Analysis for Batch {batch_idx} ===")
    
    for depth in range(results['layer_node_counts'].shape[1]):
        node_count = results['layer_node_counts'][batch_idx, depth].item()
        if node_count == 0:
            continue
            
        original_ratio = results['layer_original_ratio'][batch_idx, depth].item()
        total_prob = results['layer_total_prob_sum'][batch_idx, depth].item()
        original_prob = results['layer_original_prob_sum'][batch_idx, depth].item()
        
        print(f"\nLayer {depth}:")
        print(f"  Total nodes: {node_count}")
        print(f"  Original tree coverage: {original_ratio:.4f}")
        print(f"  Total probability mass: {total_prob:.4f}")
        print(f"  Original probability mass: {original_prob:.4f}")
        
        print("  Nodes:")
        for i in range(node_count):
            node_id = results['layer_node_indices'][batch_idx, depth, i].item()
            prob = results['layer_cumulative_probs'][batch_idx, depth, i].item()
            is_orig = results['layer_is_original'][batch_idx, depth, i].item()
            marker = "[ORIG]" if is_orig else "[TOPK]"
            print(f"    {marker} Token {node_id}: {prob:.6f}")

def tree_evolve_target_only_pytorch(
    # --- 输入接口: 接收Tensors进行计算 ---
    candidates: torch.Tensor,
    relative_positions: torch.Tensor,
    top_score_idx: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_probs: torch.Tensor,
    # --- 额外需要的测试控制参数 ---
    node_r_thresh: float,
    spec_num: int,
    topk: int
) -> (str, dict):
    """
    【KERNEL IMPLEMENTATION】
    对给定的树进行一次性的质量检查。根据“层级择优”和“保留度”算法，
    判断草稿树是否被目标模型验证。
    - 如果所有层都合格，返回状态 'VALIDATED'。
    - 如果有任何一层不合格，立即返回状态 'EVOLVE_NEEDED' 及相关信息。
    此函数自身不包含任何循环。
    """
    print("  >>> Entering kernel: tree_evolve_target_only_pytorch...")
    
    # 象征性的对象
    # mock_target_model = object()
    # current_tree_obj = {"name": "tree_to_check"}
    print(f"retrive_index: {retrive_index.tolist()}")
    print(f"retrive_next_token: {retrive_next_token.tolist()}")
    print(f"retrive_next_sibling: {retrive_next_sibling.tolist()}")
    print(f"draft_tokens: {candidates.tolist()}")

    # print(f'target_probs: {target_probs.shape}')

    

    batch_size, num_draft_tokens = candidates.shape
    
    vocab_size = target_probs.shape[2]

    # Parse args
    # spec_info = forward_batch.spec_info
    # out_cache_loc = forward_batch.out_cache_loc
    topk_p, topk_index = fast_topk(target_probs, topk, dim=-1)
    print(f"topk_p:{topk_p}, {topk_p.shape} topk_index:{topk_index}, {topk_index.shape}")

    results_batch = []

    # 遍历每个批次中的样本
    for bx in range(batch_size):
        # 预先构建原始树的子节点映射，方便快速查找
        # 格式: {parent_draft_idx: {child_token_id: child_draft_idx}}
        original_children_map = {}
        for i in range(num_draft_tokens):
            child_map = {}
            child_draft_idx = retrive_next_token[bx, i].item()
            while child_draft_idx != -1:
                child_token_id = candidates[bx, child_draft_idx].item()
                child_map[child_token_id] = child_draft_idx
                child_draft_idx = retrive_next_sibling[bx, child_draft_idx].item()
            original_children_map[i] = child_map

        # accepted_nodes 存储上一层通过验证的节点信息: (draft_idx, cumulative_prob)
        # 初始时，只有根节点（在draft list中的索引为0），累积概率为1.0
        accepted_nodes = [(0, 1.0)]

        # 逐层进行验证，从第0层开始，验证其子节点能否构成第1层
        results_info = None
        for depth in range(spec_num):
            if not accepted_nodes:
                # 如果上一层没有任何节点被接受，无法继续，验证通过
                break

            candidate_pool = [] # 存储(cum_prob, token_id, is_original, parent_draft_idx)
            layer_node_num = 0

            # 步骤 A: 收集所有父节点的 top-k 候选子节点
            print(f"\naccepted_nodes: {accepted_nodes}")
            for parent_draft_idx, parent_cum_prob in accepted_nodes:
                # 获取目标模型对该父节点的预测
                # parent_target_probs = target_probs[bx, parent_draft_idx, :]
                
                # 获取 top-k 预测
                # child_probs, child_token_ids = torch.topk(parent_target_probs, topk)
                child_probs, child_token_ids = topk_p[bx, parent_draft_idx, :], topk_index[bx, parent_draft_idx, :]
                
                
                # 获取该父节点在原始树中的子节点
                original_child_nodes = original_children_map.get(parent_draft_idx, {})
                # print(f"original_child_nodes: {original_child_nodes}")
                layer_node_num += len(original_child_nodes)

                for i in range(topk):
                    token_id = child_token_ids[i].item()
                    prob = child_probs[i].item()
                    
                    # 计算新的累积概率
                    child_cum_prob = parent_cum_prob * prob
                    
                    # 检查该token是否在原始子节点中
                    is_original = token_id in original_child_nodes
                    
                    candidate_pool.append((child_cum_prob, token_id, is_original, parent_draft_idx))
            
            if not candidate_pool:
                break # 没有候选节点，验证结束

            # 步骤 B: 全局剪枝，确定正式子节点
            candidate_pool.sort(key=lambda x: x[0], reverse=True)
            formal_children = candidate_pool[:topk]
            defined_formal_children = candidate_pool[:layer_node_num]
            print(f"formal_children: {formal_children}, layer_node_num:{layer_node_num}")
            print(f"defined_formal_children: {defined_formal_children}")

            # 步骤 C: 计算保留度
            retained_prob_sum = sum(p for p, _, is_orig, _ in defined_formal_children if is_orig)
            defined_formal_prob_sum = sum(p for p, _, _, _ in defined_formal_children)

            retention_rate = retained_prob_sum / defined_formal_prob_sum if defined_formal_prob_sum > 1e-9 else 0.0
            
            print(f"  [Batch {bx}, Layer {depth+1}] Retention Rate: {retention_rate:.4f} (Threshold: {node_r_thresh})")

            # 步骤 D: 决策
            if retention_rate < node_r_thresh:
                print(f"  Kernel check failed at layer {depth+1}. Reporting back to main loop.")
                # 准备进化所需的信息
                new_nodes_info = [{'token_id': token_id, 'prob': prob} for prob, token_id, _, _ in formal_children]
                info = {'layer_to_evolve': depth + 1, 'new_nodes': new_nodes_info}
                # assert False
                results_info = ('EVOLVE_NEEDED', info)
                break
            else:
                # 该层通过验证，准备下一层的输入
                next_accepted_nodes = []
                # 筛选出被保留的原始节点
                retained_children = [(p, tid, p_idx) for p, tid, is_orig, p_idx in formal_children if is_orig]
                for cum_prob, token_id, parent_idx in retained_children:
                    # 找到这个保留节点的 draft_idx
                    child_draft_idx = original_children_map[parent_idx][token_id]
                    next_accepted_nodes.append((child_draft_idx, cum_prob))
                
                accepted_nodes = next_accepted_nodes
        if results_info is None:
            results_batch.append(('VALIDATED', {}))
        else:
            results_batch.append(results_info)
    # 如果所有层的检查都通过了（或者对于某个批次验证提前结束）
    # 在这个实现中，只要有一个batch需要演进，我们就返回。
    # 如果所有batch都成功完成循环，则意味着全部验证通过。
    print(f"  Kernel check passed for Batch {bx}. Continuing to next batch...")

    # 如果所有批次都通过了检查
    print("  Kernel check passed for all batches. Tree is validated.")
    
    # assert False
    
    # return ('VALIDATED', {})
    print(f"results_batch: {results_batch}")
    return results_batch
    # assert False



    # for bx in range(batch_size):
    #     # =================================================================
    #     # Phase 1: Acceptance Loop
    #     # =================================================================
    #     # prob_acc = 0.0
        
    #     # Start at the root of the draft tree for this sequence
    #     last_accepted_retrive_idx = retrive_index[bx, 0].item()
    #     accept_index[bx, 0] = last_accepted_retrive_idx
        
    #     num_accepted_layers = 0
    #     cur_prob_offset = 0
        
    #     # Index within the draft tokens for the current sequence (0 to num_draft_tokens-1)
    #     # Starts from the children of the root node (index 0)
    #     current_draft_idx_in_seq = 0# We can accept up to `num_speculative_tokens - 1` new tokens
    #     for j in range(1, spec_num):
    #         # Get the first child of the previously accepted token
    #         current_draft_idx_in_seq = retrive_next_token[bx, current_draft_idx_in_seq].item()

    #         # Iterate through siblings at the current tree depth
    #         while current_draft_idx_in_seq != -1:
    #             draft_token_id = candidates[bx, current_draft_idx_in_seq].item()
                
    #             # Get target probability for this specific token
    #             target_prob_single = target_probs[bx, cur_prob_offset, draft_token_id].item()

    #             prob_acc += target_prob_single

    #             # The random number is associated with the token being checked
    #             coin = uniform_samples[bx, current_draft_idx_in_seq].item()

    #             # --- Core Acceptance Logic ---
    #             if coin <= prob_acc / capped_threshold_acc or target_prob_single >= threshold_single:
    #                 # Accept the token
    #                 prob_acc = 0.0
                    
    #                 # final_predicts[last_accepted_retrive_idx] = draft_token_id
    #                 predicts[last_accepted_retrive_idx] = draft_token_id
                    
    #                 num_accepted_tokens += 1
    #                 current_retrive_idx = retrive_index[bx, current_draft_idx_in_seq].item()
    #                 accept_index[bx, num_accepted_tokens] = current_retrive_idx
    #                 last_accepted_retrive_idx = current_retrive_idx
    #                 cur_prob_offset = current_draft_idx_in_seq
    #                 break # Break from sibling loop to go to the next level
    #             else:
    #                 # Reject the token and update draft_probs for the final sampling
    #                 draft_probs[bx, current_draft_idx_in_seq, draft_token_id] = target_probs[bx, current_draft_idx_in_seq, draft_token_id]
                    
    #                 # Move to the next sibling
    #                 current_draft_idx_in_seq = retrive_next_sibling[bx, current_draft_idx_in_seq].item()

    #         if current_draft_idx_in_seq == -1:
    #             # If we exhausted all siblings and accepted none, stop
    #             break
        
    #     accept_token_num[bx] = num_accepted_tokens
        
    #     # =================================================================
    #     # Phase 2: Final Sampling (Bonus Token)
    #     # =================================================================
        
    #     # Get probabilities at the last accepted position
    #     q_probs = target_probs[bx, current_draft_idx_in_seq, :]
    #     p_probs = draft_probs[bx, current_draft_idx_in_seq, :]
        
    #     # Sample from relu(q - p)
    #     final_dist = torch.maximum(q_probs - p_probs, torch.tensor(0.0, device=device))
        
    #     sum_dist = torch.sum(final_dist)
        
    #     final_token_id = vocab_size - 1 # Default as in CUDA
    #     if sum_dist.item() > 1e-6:
    #         final_coin = uniform_samples_for_final_sampling[bx, current_draft_idx_in_seq].item()
    #         u = final_coin * sum_dist.item()
            
    #         cumulative_prob = 0.0
    #         # This loop is the Python equivalent of the parallel scan in CUDA
    #         for token_id in range(vocab_size):
    #             cumulative_prob += final_dist[token_id].item()
    #             if cumulative_prob > u:
    #                 final_token_id = token_id
    #                 break
        
    #     # final_predicts[last_accepted_retrive_idx] = final_token_id
    #     predicts[last_accepted_retrive_idx] = final_token_id

    # return score_list, token_list, parents_list


    # # Each item in the batch is processed independently (like a CUDA block)
    # for bx in range(batch_size):
    #     # =================================================================
    #     # Phase 1: Acceptance Loop
    #     # =================================================================
    #     prob_acc = 0.0
        
    #     # Start at the root of the draft tree for this sequence
    #     last_accepted_retrive_idx = retrive_index[bx, 0].item()
    #     # accept_index[bx, 0] = last_accepted_retrive_idx
        
        
    #     num_accepted_tokens = 0
    #     cur_prob_offset = 0
        
    #     # Index within the draft tokens for the current sequence (0 to num_draft_tokens-1)
    #     # Starts from the children of the root node (index 0)
    #     current_draft_idx_in_seq = 0# We can accept up to `num_speculative_tokens - 1` new tokens
    #     for j in range(1, spec_num):
    #         # Get the first child of the previously accepted token
    #         current_draft_idx_in_seq = retrive_next_token[bx, current_draft_idx_in_seq].item()

    #         # Iterate through siblings at the current tree depth
    #         while current_draft_idx_in_seq != -1:
    #             draft_token_id = candidates[bx, current_draft_idx_in_seq].item()
                
    #             # Get target probability for this specific token
    #             target_prob_single = target_probs[bx, cur_prob_offset, draft_token_id].item()

    #             prob_acc += target_prob_single

    #             # The random number is associated with the token being checked
    #             coin = uniform_samples[bx, current_draft_idx_in_seq].item()

    #             # --- Core Acceptance Logic ---
    #             if coin <= prob_acc / capped_threshold_acc or target_prob_single >= threshold_single:
    #                 # Accept the token
    #                 prob_acc = 0.0
                    
    #                 # final_predicts[last_accepted_retrive_idx] = draft_token_id
    #                 predicts[last_accepted_retrive_idx] = draft_token_id
                    
    #                 num_accepted_tokens += 1
    #                 current_retrive_idx = retrive_index[bx, current_draft_idx_in_seq].item()
    #                 accept_index[bx, num_accepted_tokens] = current_retrive_idx
    #                 last_accepted_retrive_idx = current_retrive_idx
    #                 cur_prob_offset = current_draft_idx_in_seq
    #                 break # Break from sibling loop to go to the next level
    #             else:
    #                 # Reject the token and update draft_probs for the final sampling
    #                 draft_probs[bx, current_draft_idx_in_seq, draft_token_id] = target_probs[bx, current_draft_idx_in_seq, draft_token_id]
                    
    #                 # Move to the next sibling
    #                 current_draft_idx_in_seq = retrive_next_sibling[bx, current_draft_idx_in_seq].item()

    #         if current_draft_idx_in_seq == -1:
    #             # If we exhausted all siblings and accepted none, stop
    #             break
        
    #     accept_token_num[bx] = num_accepted_tokens
        
    #     # =================================================================
    #     # Phase 2: Final Sampling (Bonus Token)
    #     # =================================================================
        
    #     # Get probabilities at the last accepted position
    #     q_probs = target_probs[bx, current_draft_idx_in_seq, :]
    #     p_probs = draft_probs[bx, current_draft_idx_in_seq, :]
        
    #     # Sample from relu(q - p)
    #     final_dist = torch.maximum(q_probs - p_probs, torch.tensor(0.0, device=device))
        
    #     sum_dist = torch.sum(final_dist)
        
    #     final_token_id = vocab_size - 1 # Default as in CUDA
    #     if sum_dist.item() > 1e-6:
    #         final_coin = uniform_samples_for_final_sampling[bx, current_draft_idx_in_seq].item()
    #         u = final_coin * sum_dist.item()
            
    #         cumulative_prob = 0.0
    #         # This loop is the Python equivalent of the parallel scan in CUDA
    #         for token_id in range(vocab_size):
    #             cumulative_prob += final_dist[token_id].item()
    #             if cumulative_prob > u:
    #                 final_token_id = token_id
    #                 break
        
    #     # final_predicts[last_accepted_retrive_idx] = final_token_id
    #     predicts[last_accepted_retrive_idx] = final_token_id


    # # # Capped threshold_acc to avoid division by zero, as in the CUDA host code
    # # capped_threshold_acc = max(threshold_acc, 1e-9)

    # for bx in range(batch_size):
    #     # =================================================================
    #     # Phase 1: Acceptance Loop
    #     # =================================================================
    #     prob_acc = 0.0
        
    #     # Start at the root of the draft tree for this sequence
    #     last_accepted_retrive_idx = retrive_index[bx, 0].item()
    #     accept_index[bx, 0] = last_accepted_retrive_idx
        
    #     num_accepted_tokens = 0
    #     cur_prob_offset = 0
    
    # for i in range(spec_num):
    #     old_nodes = mock_get_nodes_at_layer(current_tree_obj, i)
    #     new_nodes = mock_compete_k(current_tree_obj, mock_target_model, i, topk)
    #     overlap_ratio = mock_calculate_overlap_ratio(old_nodes, new_nodes, current_tree_obj)

    #     if overlap_ratio < node_r_thresh:
    #         print(f"  Kernel check failed at layer {i}. Reporting back to main loop.")
    #         # 返回一个需要进化的信号，以及进化所需的信息
    #         info = {'layer_to_evolve': i, 'new_nodes': new_nodes}
    #         return ('EVOLVE_NEEDED', info)

    # # 如果所有层都通过了检查
    # print("  Kernel check passed. All layers are valid.")
    # return ('VALIDATED', {})

def check_continue():
    # TODO: Implement this
    # Check if the tree is complete
    return False
    pass


def visualize_tree(
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    draft_tokens: torch.Tensor,
    batch_size: int,
    num_draft_tokens: int
):
    """
    使用字符可视化一个批次中每个样本的推测树结构。

    Args:
        retrive_next_token (torch.Tensor): (bs, num_draft_tokens) 形状的张量，指向第一个子节点。
        retrive_next_sibling (torch.Tensor): (bs, num_draft_tokens) 形状的张量，指向下一个兄弟节点。
        draft_tokens (torch.Tensor): (bs * num_draft_tokens,) 形状的张量，包含所有草稿token的ID。
        batch_size (int): 批次大小。
        num_draft_tokens (int): 每个样本的草稿token数量。
    """
    # 将Tensor转到CPU并转换为列表以便处理
    next_token = retrive_next_token.cpu().tolist()
    next_sibling = retrive_next_sibling.cpu().tolist()
    tokens = draft_tokens.cpu().reshape(batch_size, num_draft_tokens).tolist()

    # 递归辅助函数，用于打印子树
    def _print_subtree(b_idx: int, node_idx: int, prefix: str):
        # 找到当前节点的所有子节点
        children = []
        child_idx = next_token[b_idx][node_idx]
        while child_idx != -1:
            children.append(child_idx)
            child_idx = next_sibling[b_idx][child_idx]

        # 遍历并打印所有子节点
        for i, child in enumerate(children):
            # 判断是否是最后一个子节点，以决定使用哪种连接符
            is_last = (i == len(children) - 1)
            connector = "└── " if is_last else "├── "
            
            # 打印当前子节点信息
            token_id = tokens[b_idx][child]
            print(f"{prefix}{connector}Node[{child}] (Token: {token_id})")
            
            # 更新下一层的前缀并递归打印
            new_prefix = prefix + ("    " if is_last else "│   ")
            _print_subtree(b_idx, child, new_prefix)

    print("\n" + "="*20 + " 推测树结构可视化 " + "="*20)
    for b in range(batch_size):
        print(f"\n--- Batch Index {b} ---")
        print("(Verified Token) [root]")
        # 节点0是所有第一层草稿token的逻辑父节点
        _print_subtree(b_idx=b, node_idx=0, prefix="")
    print("\n" + "="*57)

# @pytest.mark.parametrize("device", ["cuda"])
# def test_tree_iterative_decoding(device):
#     """
#     【FINAL CORRECT VERSION - WITH EXPOSED WHILE LOOP】
#     测试主流程包含“验证-进化”的while循环，清晰地展示了控制逻辑。
#     """
#     if device == "cuda" and not torch.cuda.is_available():
#         pytest.skip("CUDA not available")

#     # --- 1. 初始设置 ---
#     first_rank_print("\n\n--- Running Test: test_tree_iterative_decoding (Exposed Loop Version) ---")
#     topk, std_depth, num_verify_tokens_g1 = 4, 4, 8
#     vocab_size, batch_size = 32000, 2
    
#     # --- 2. 准备初版草稿树 ---
#     first_rank_print("Step 1: Preparing initial draft tree...")
#     verified_id, score_list, token_list, parents_list, seq_lens = draft_output(None, [], [], [], None)

#     print(f"parents_list: {parents_list}")
#     print(f"token_list: {token_list}")
#     print(f"score_list: {score_list}")
    
#     # 构建初版的Tensors
#     (
#         tree_mask_g1, positions_g1, retrive_index_g1, retrive_next_token_g1,
#         retrive_next_sibling_g1, draft_tokens_g1, top_score_idx_g1
#     ) = build_tree_kernel_efficient(
#         verified_id, score_list, token_list, parents_list, seq_lens,
#         torch.sum(seq_lens).item(), topk, std_depth, num_verify_tokens_g1
#     )
#     bs_g1 = retrive_index_g1.shape[0]
#     relative_pos = positions_g1.view(bs_g1, -1)-seq_lens.unsqueeze(1)
#     print(f"top_score_idx_g1: {top_score_idx_g1}")

#     visualize_tree(
#         retrive_next_token=retrive_next_token_g1,
#         retrive_next_sibling=retrive_next_sibling_g1,
#         draft_tokens=draft_tokens_g1,
#         batch_size=bs_g1,
#         num_draft_tokens=num_verify_tokens_g1
#     )

#     # print(f"retrive_index: {retrive_index_g1.tolist()}")
#     # print(f"retrive_next_token: {retrive_next_token_g1.tolist()}")
#     # print(f"retrive_next_sibling: {retrive_next_sibling_g1.tolist()}")
#     # print(f"draft_tokens: {draft_tokens_g1.tolist()}")

#     target_probs_g1 = torch.zeros(bs_g1, num_verify_tokens_g1, vocab_size, device=device) 
#     draft_probs_g1 = torch.zeros_like(target_probs_g1)

#     draft_prob, _, _, _ = reconstruct_draft_probs(score_list=score_list, token_list=token_list, parents_list=parents_list, selected_indices=top_score_idx_g1, vocab_size=vocab_size, num_verify_tokens=num_verify_tokens_g1, device=device)
    
#     target_output(target_probs_g1, draft_prob, draft_tokens_g1, num_verify_tokens_g1)


#     # --- 3. 【核心】验证与进化循环 (置于外部) ---
#     first_rank_print("\nStep 2: Entering the main Verification & Evolution loop...")
#     is_tree_validated = False
#     evolution_attempts = 0

#     # 使用局部变量来持有可能会在循环中被更新的树Tensors
#     current_parents_list = parents_list
#     current_top_score_idx = top_score_idx_g1
#     current_draft_tokens = draft_tokens_g1
#     current_retrive_index = retrive_index_g1
#     current_retrive_next_token = retrive_next_token_g1
#     current_retrive_next_sibling = retrive_next_sibling_g1
    
#     while not is_tree_validated and evolution_attempts < 3:
#         evolution_attempts += 1
#         print(f"\n  Main loop: Evolution attempt #{evolution_attempts}")
        
#         # 调用简单的“内核”函数进行检查
#         status, info = tree_evolve_target_only_pytorch(
#             candidates=current_draft_tokens.view(bs_g1, -1),
#             relative_positions=relative_pos,
#             top_score_idx=current_top_score_idx,
#             retrive_index=current_retrive_index,
#             retrive_next_token=current_retrive_next_token,
#             retrive_next_sibling=current_retrive_next_sibling,
#             target_probs=target_probs_g1,
#             node_r_thresh=0.8,
#             spec_num=std_depth,
#             topk=topk
#         )

#         # 在主流程中进行决策
#         if status == 'EVOLVE_NEEDED':
#             print("  Main loop: Received 'EVOLVE_NEEDED'. Simulating re-draft.")
            
#             # 模拟进化和重新草稿
#             evolved_tree_info = mock_replace_and_truncate_tree({}, info['layer_to_evolve'], info['new_nodes'])
#             (
#                 new_verified_id, new_score_list, new_token_list,
#                 new_parents_list, new_seq_lens
#             ) = mock_redraft_from_evolved_tree(evolved_tree_info, verified_id, seq_lens)

#             # 使用重新草稿的结果，更新下一次循环所需的Tensors
#             (
#                 tree_mask, positions, current_retrive_index, current_retrive_next_token,
#                 current_retrive_next_sibling, current_draft_tokens, current_top_score_idx
#             ) = build_tree_kernel_efficient(
#                 new_verified_id, new_score_list, new_token_list, new_parents_list,
#                 new_seq_lens, torch.sum(new_seq_lens).item(), topk, std_depth,
#                 num_verify_tokens=num_verify_tokens_g1
#             )
#             # 在真实场景中，target_probs也需要更新
#             target_probs_g1.zero_() 

#         elif status == 'VALIDATED':
#             print("  Main loop: Received 'VALIDATED'. Exiting evolution loop.")
#             is_tree_validated = True

#     assert is_tree_validated, "Tree evolution loop failed to produce a valid tree."
    
#     # --- 4. 最终采样 ---
#     first_rank_print("\nStep 3: Performing final sampling on the validated tree...")
    
#     # 设置赠送token
#     target_probs_g1[0, 1, 999] = 0.9
#     target_probs_g1[1, 3, 888] = 0.9

#     predicts_g1 = torch.full((bs_g1 * num_verify_tokens_g1, ), -1, dtype=torch.int32, device=device)
#     accept_index_g1 = torch.full((bs_g1, std_depth + 1), -1, dtype=torch.int32, device=device)
#     accept_token_num_g1 = torch.full((bs_g1,), 0, dtype=torch.int32, device=device)
    
#     # 使用经过循环验证后的、最终的Tensors进行采样
#     tree_speculative_sampling_target_only_pytorch(
#         predicts=predicts_g1, accept_index=accept_index_g1, accept_token_num=accept_token_num_g1,
#         candidates=current_draft_tokens.view(bs_g1, -1), retrive_index=current_retrive_index,
#         retrive_next_token=current_retrive_next_token, retrive_next_sibling=current_retrive_next_sibling,
#         uniform_samples=torch.full((bs_g1, num_verify_tokens_g1), 0.01, device=device),
#         uniform_samples_for_final_sampling=torch.full((bs_g1, num_verify_tokens_g1), 0.01, device=device),
#         target_probs=target_probs_g1, draft_probs=draft_probs_g1,
#         threshold_single=0.8, threshold_acc=0.8
#     )

#     # --- 5. 最终断言 ---
#     first_rank_print("\nStep 4: Asserting the final results...")
#     assert accept_token_num_g1.tolist() == [1, 2]
#     # ... (其他断言)
#     first_rank_print("Test case passed successfully.")




@pytest.mark.parametrize("device", ["cuda"])
def test_tree_iterative_decoding_reality(device):
    """
    【FINAL CORRECT VERSION - WITH EXPOSED WHILE LOOP】
    测试主流程包含“验证-进化”的while循环，清晰地展示了控制逻辑。
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # --- 1. 初始设置 ---
    first_rank_print("\n\n--- Running Test: test_tree_iterative_decoding (Exposed Loop Version) ---")
    topk, std_depth, num_verify_tokens_g1 = 4, 4, 8
    vocab_size, batch_size = 32000, 2
    
    # --- 2. 准备初版草稿树 ---
    first_rank_print("Step 1: Preparing initial draft tree...")
    verified_id, score_list, token_list, parents_list, seq_lens = draft_output(None, [], [], [], None)

    print(f"parents_list: {parents_list}")
    print(f"token_list: {token_list}")
    print(f"score_list: {score_list}")
    
    # 构建初版的Tensors
    (
        tree_mask_g1, positions_g1, retrive_index_g1, retrive_next_token_g1,
        retrive_next_sibling_g1, draft_tokens_g1, top_score_idx_g1
    ) = build_tree_kernel_efficient(
        verified_id, score_list, token_list, parents_list, seq_lens,
        torch.sum(seq_lens).item(), topk, std_depth, num_verify_tokens_g1
    )
    bs_g1 = retrive_index_g1.shape[0]
    relative_pos = positions_g1.view(bs_g1, -1)-seq_lens.unsqueeze(1)
    print(f"top_score_idx_g1: {top_score_idx_g1}")

    visualize_tree(
        retrive_next_token=retrive_next_token_g1,
        retrive_next_sibling=retrive_next_sibling_g1,
        draft_tokens=draft_tokens_g1,
        batch_size=bs_g1,
        num_draft_tokens=num_verify_tokens_g1
    )

    # print(f"retrive_index: {retrive_index_g1.tolist()}")
    # print(f"retrive_next_token: {retrive_next_token_g1.tolist()}")
    # print(f"retrive_next_sibling: {retrive_next_sibling_g1.tolist()}")
    # print(f"draft_tokens: {draft_tokens_g1.tolist()}")

    target_probs_g1 = torch.zeros(bs_g1, num_verify_tokens_g1, vocab_size, device=device) 
    draft_probs_g1 = torch.zeros_like(target_probs_g1)

    draft_prob, _, _, _ = reconstruct_draft_probs(score_list=score_list, token_list=token_list, parents_list=parents_list, selected_indices=top_score_idx_g1, vocab_size=vocab_size, num_verify_tokens=num_verify_tokens_g1, device=device)
    
    target_output(target_probs_g1, draft_prob, draft_tokens_g1, num_verify_tokens_g1)


    # --- 3. 【核心】验证与进化循环 (置于外部) ---
    first_rank_print("\nStep 2: Entering the main Verification & Evolution loop...")
    is_tree_validated = False
    evolution_attempts = 0

    # 使用局部变量来持有可能会在循环中被更新的树Tensors
    current_parents_list = parents_list
    current_top_score_idx = top_score_idx_g1
    current_draft_tokens = draft_tokens_g1
    current_retrive_index = retrive_index_g1
    current_retrive_next_token = retrive_next_token_g1
    current_retrive_next_sibling = retrive_next_sibling_g1
    
    while not is_tree_validated and evolution_attempts < 3:
        evolution_attempts += 1
        print(f"\n  Main loop: Evolution attempt #{evolution_attempts}")
        
        # 调用简单的“内核”函数进行检查
        status, info = tree_evolve_target_only_pytorch(
            candidates=current_draft_tokens.view(bs_g1, -1),
            relative_positions=relative_pos,
            top_score_idx=current_top_score_idx,
            retrive_index=current_retrive_index,
            retrive_next_token=current_retrive_next_token,
            retrive_next_sibling=current_retrive_next_sibling,
            target_probs=target_probs_g1,
            node_r_thresh=0.8,
            spec_num=std_depth,
            topk=topk
        )

        # 在主流程中进行决策
        if status == 'EVOLVE_NEEDED':
            print("  Main loop: Received 'EVOLVE_NEEDED'. Simulating re-draft.")
            
            # 模拟进化和重新草稿
            evolved_tree_info = mock_replace_and_truncate_tree({}, info['layer_to_evolve'], info['new_nodes'])
            (
                new_verified_id, new_score_list, new_token_list,
                new_parents_list, new_seq_lens
            ) = mock_redraft_from_evolved_tree(evolved_tree_info, verified_id, seq_lens)

            # 使用重新草稿的结果，更新下一次循环所需的Tensors
            (
                tree_mask, positions, current_retrive_index, current_retrive_next_token,
                current_retrive_next_sibling, current_draft_tokens, current_top_score_idx
            ) = build_tree_kernel_efficient(
                new_verified_id, new_score_list, new_token_list, new_parents_list,
                new_seq_lens, torch.sum(new_seq_lens).item(), topk, std_depth,
                num_verify_tokens=num_verify_tokens_g1
            )
            # 在真实场景中，target_probs也需要更新
            target_probs_g1.zero_() 

        elif status == 'VALIDATED':
            print("  Main loop: Received 'VALIDATED'. Exiting evolution loop.")
            is_tree_validated = True

    assert is_tree_validated, "Tree evolution loop failed to produce a valid tree."
    
    # --- 4. 最终采样 ---
    first_rank_print("\nStep 3: Performing final sampling on the validated tree...")
    
    # 设置赠送token
    target_probs_g1[0, 1, 999] = 0.9
    target_probs_g1[1, 3, 888] = 0.9

    predicts_g1 = torch.full((bs_g1 * num_verify_tokens_g1, ), -1, dtype=torch.int32, device=device)
    accept_index_g1 = torch.full((bs_g1, std_depth + 1), -1, dtype=torch.int32, device=device)
    accept_token_num_g1 = torch.full((bs_g1,), 0, dtype=torch.int32, device=device)
    
    # 使用经过循环验证后的、最终的Tensors进行采样
    tree_speculative_sampling_target_only_pytorch(
        predicts=predicts_g1, accept_index=accept_index_g1, accept_token_num=accept_token_num_g1,
        candidates=current_draft_tokens.view(bs_g1, -1), retrive_index=current_retrive_index,
        retrive_next_token=current_retrive_next_token, retrive_next_sibling=current_retrive_next_sibling,
        uniform_samples=torch.full((bs_g1, num_verify_tokens_g1), 0.01, device=device),
        uniform_samples_for_final_sampling=torch.full((bs_g1, num_verify_tokens_g1), 0.01, device=device),
        target_probs=target_probs_g1, draft_probs=draft_probs_g1,
        threshold_single=0.8, threshold_acc=0.8
    )

    # --- 5. 最终断言 ---
    first_rank_print("\nStep 4: Asserting the final results...")
    assert accept_token_num_g1.tolist() == [1, 2]
    # ... (其他断言)
    first_rank_print("Test case passed successfully.")

@pytest.mark.parametrize("device", ["cuda"])
def test_create_dense_prob_tensor(device):
    
    # 输入数据
    # score_list = [torch.tensor([[[7.1127e-01, 2.8292e-01, 2.2995e-03, 1.7357e-03]],[[9.7476e-01, 2.2219e-02, 6.5031e-04, 1.3212e-04]]], device=device), torch.tensor([[[6.9142e-01, 1.2863e-02, 1.6873e-03, 1.1871e-03],[2.4787e-01, 1.8818e-02, 1.4204e-02, 9.2235e-04],[2.2971e-03, 1.6700e-06, 1.8737e-07, 8.3146e-08],[1.2771e-03, 2.4374e-04, 1.7832e-04, 1.1947e-05]],[[8.4832e-02, 6.6068e-02, 5.8304e-02, 5.7851e-02],[2.3616e-03, 1.1243e-03, 5.4368e-04, 2.7768e-04],[2.5286e-04, 1.5578e-04, 2.8817e-05, 1.2888e-05],[1.2834e-04, 2.5417e-06, 1.1279e-06, 1.6088e-08]]], device=device), torch.tensor([[[6.6438e-01, 2.6997e-02, 2.4236e-05, 4.0821e-06],[2.4402e-01, 2.8409e-03, 5.0935e-04, 2.9022e-04],[1.6178e-02, 2.0567e-03, 4.5892e-04, 3.0034e-05],[1.3023e-02, 5.0497e-04, 3.6371e-04, 8.7750e-05]],[[2.3263e-02, 2.0054e-02, 9.3990e-03, 2.7783e-03],[6.4156e-02, 5.5506e-04, 1.0429e-04, 9.7211e-05],[4.9950e-02, 5.0630e-03, 9.0068e-04, 3.3656e-04],[7.5817e-03, 8.5731e-04, 6.9972e-04, 6.0793e-04]]], device=device), torch.tensor([[[6.6420e-01, 1.0525e-04, 6.5864e-05, 1.2253e-06],[1.3019e-01, 1.0461e-01, 5.2083e-03, 1.6777e-03],[2.0103e-02, 6.7335e-03, 1.2625e-04, 1.0364e-05],[1.5142e-02, 7.0819e-04, 9.6595e-05, 8.7951e-05]],[[5.8608e-02, 1.8840e-03, 7.8535e-04, 4.4400e-04],[1.2185e-02, 2.0684e-03, 1.7418e-03, 1.4327e-03],[6.2455e-03, 6.1487e-03, 2.6862e-03, 1.8034e-03],[1.8590e-03, 1.6151e-03, 1.2481e-03, 3.6038e-04]]], device=device)]
    # token_list = [torch.tensor([[29896, 29906, 29900, 29945], [13, 2, 29871, 28956]], dtype=torch.int64, device=device), torch.tensor([[29889, 29974, 29945, 29900, 29974, 29922, 29930, 29958, 29889, 29974, 29930, 29945, 29974, 29922, 29930, 29958], [22550, 4136, 16492, 8439, 29871, 2, 3001, 13, 2, 13, 29906, 29946, 2, 13, 29871, 259]], device=device), torch.tensor([[29946, 29945, 29953, 29906, 29896, 29945, 29900, 29906, 29896, 29945, 29906, 29953, 29896, 29945, 29906, 29946], [29871, 2, 29901, 29889, 29871, 2, 395, 259, 29901, 29871, 2, 29889, 3001, 1234, 7146, 2186]], device=device), torch.tensor([[29946, 29974, 29945, 29930, 29889, 29922, 29974, 29930, 29974, 29946, 29930, 29922, 29889, 29974, 29945, 29922], [29941, 29906, 2, 29946, 29871, 450, 319, 14990, 29946, 29941, 2, 29906, 29871, 2, 3001, 13]], device=device)]
    top_score_idx_g1 = torch.tensor([[ 0,  1,  4,  8, 20, 24, 36], [ 0,  4,  5,  6,  7, 24, 36]], dtype=torch.int64, device=device)
    verified_id, score_list, token_list, parents_list, seq_lens = draft_output(None, [], [], [], None)
    topk = 4

    # 在测试用例中定义的常量
    VOCAB_SIZE = 32000
    NUM_VERIFY_TOKENS = 8 # g1, or 8

    # 调用转换函数
    final_draft_probs, token_list_token, score_list_token, map_indices = reconstruct_draft_probs(
        score_list, token_list, parents_list, top_score_idx_g1, VOCAB_SIZE, NUM_VERIFY_TOKENS, topk,device
    )

    # --- 验证结果 ---
    # print("\n--- 验证重建结果 ---")
    # 检查第一个批次，第二个验证token (verify_idx=1)
    bx, v_idx, topk_idx = 0, 1, 1
    # 从输入中找到期望值
    expected_selected_idx = map_indices[bx, v_idx].item()
    expected_token_id = token_list_token[bx, expected_selected_idx, topk_idx].item()
    expected_prob = score_list_token[bx, expected_selected_idx, topk_idx].item()
    
    # 从输出中找到实际值
    actual_prob = final_draft_probs[bx, v_idx, expected_token_id].item()

    # print(f"检查 Batch {bx}, Verify Idx {v_idx}:")
    # print(f"  - 期望的 Token ID: {expected_token_id}")
    # print(f"  - 期望的概率: {expected_prob:.6f}")
    # print(f"  - 重建的概率: {actual_prob:.6f}")
    assert torch.isclose(torch.tensor(expected_prob), torch.tensor(actual_prob)), "验证失败！"
    # print("  - 验证成功！")
    
    # 检查第二个批次，第四个验证token (verify_idx=3)
    bx, v_idx, topk_idx = 1, 6, 1
    expected_selected_idx = map_indices[bx, v_idx].item()
    expected_token_id = token_list_token[bx, expected_selected_idx, topk_idx].item()
    expected_prob = score_list_token[bx, expected_selected_idx, topk_idx].item()
    actual_prob = final_draft_probs[bx, v_idx, expected_token_id].item()
    
    # print(f"检查 Batch {bx}, Verify Idx {v_idx}:")
    # print(f"  - 期望的 Token ID: {expected_token_id}")
    # print(f"  - 期望的概率: {expected_prob:.6f}")
    # print(f"  - 重建的概率: {actual_prob:.6f}")
    assert torch.isclose(torch.tensor(expected_prob), torch.tensor(actual_prob)), "验证失败！"
    # print("  - 验证成功！")