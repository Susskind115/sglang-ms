import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import tree_speculative_sampling_target_only




import numpy as np

import torch

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

test_cases = [
    (
        1,
        1,
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
    (
        0,  # threshold_single
        0,  # threshold_acc
        [1, 2, 18, -1, -1, -1, 11, -1, -1, -1, 12, 18],
        [[0, 1, 2, -1], [6, 10, 11, -1]],
        [2, 2],
    ),
]


@pytest.mark.parametrize(
    "threshold_single, threshold_acc, expected_predicts, expected_accept_index, expected_accept_token_num",
    test_cases,
)
def test_tree_speculative_sampling_target_only(
    threshold_single,
    threshold_acc,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
):
    """
    Tests the tree_speculative_sampling_target_only function using Pytest parameterization.
    """
    device = "cuda"

    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int32,
        device=device,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int32,
        device=device,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device=device)
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10

    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i, j]) < 10:
                target_logits[i, j, 18] = 10

    temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
    bs, num_draft_tokens = candidates.shape
    num_spec_step = len(expected_accept_index[0])
    predict_shape = (len(expected_predicts),)

    predicts = torch.full(predict_shape, -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, num_spec_step), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device=device)

    expanded_temperature = temperatures.unsqueeze(1).unsqueeze(1)
    target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
    draft_probs = torch.full_like(target_probs, 0, dtype=torch.float32, device=device)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)

    # print(f"target_probs: {target_probs}")
    # print(f"candidates: {candidates}")

    # tree_speculative_sampling_target_only(
    #     predicts=predicts,
    #     accept_index=accept_index,
    #     accept_token_num=accept_token_num,
    #     candidates=candidates,
    #     retrive_index=retrive_index,
    #     retrive_next_token=retrive_next_token,
    #     retrive_next_sibling=retrive_next_sibling,
    #     uniform_samples=coins,
    #     target_probs=target_probs,
    #     draft_probs=draft_probs,
    #     threshold_single=threshold_single,
    #     threshold_acc=threshold_acc,
    #     deterministic=True,
    # )

    # predicts, accept_index, accept_token_num = tree_speculative_sampling_target_only_pytorch(
    tree_speculative_sampling_target_only_pytorch(
        predicts=predicts, # shallow copy
        accept_index=accept_index, # shallow copy
        accept_token_num=accept_token_num, # shallow copy
        
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
    )
    # 张量确实会传递下去并发生改变，因为这个张量属于复杂数据类型，传递是浅拷贝，拷贝内存位置过去了。

    print(f"predicts: {predicts.tolist()}")
    print(f"accept_index: {accept_index.tolist()}")
    print(f"accept_token_num: {accept_token_num.tolist()}")

    assert (
        predicts.tolist() == expected_predicts
    ), f"Predicts mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_index.tolist() == expected_accept_index
    ), f"Accept index mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_token_num.tolist() == expected_accept_token_num
    ), f"Accept token num mismatch for thresholds ({threshold_single}, {threshold_acc})"


if __name__ == "__main__":
    pytest.main([__file__])


"""

void tree_speculative_sampling_target_only(
    at::Tensor predicts,
    at::Tensor accept_index,
    at::Tensor accept_token_num,  // mutable
    at::Tensor candidates,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor uniform_samples,
    at::Tensor target_probs,
    at::Tensor draft_probs,
    double threshold_single,
    double threshold_acc,
    bool deterministic = true,
    int64_t cuda_stream = 0) {
	
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = sampling::TreeSpeculativeSamplingTargetOnly<float, int>(
      static_cast<int*>(predicts.data_ptr()),
      static_cast<int*>(accept_index.data_ptr()),
      static_cast<int*>(accept_token_num.data_ptr()),
      static_cast<int*>(candidates.data_ptr()),
      static_cast<int*>(retrive_index.data_ptr()),
      static_cast<int*>(retrive_next_token.data_ptr()),
      static_cast<int*>(retrive_next_sibling.data_ptr()),
      static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<float*>(target_probs.data_ptr()),
      static_cast<float*>(draft_probs.data_ptr()),
      batch_size,
      num_spec_step,
      num_draft_tokens,
      vocab_size,
      static_cast<float>(threshold_single),
      static_cast<float>(threshold_acc),
      deterministic,
      stream);

  TORCH_CHECK(
      status == cudaSuccess,
      "TreeSpeculativeSamplingTargetOnly failed with error code " + std::string(cudaGetErrorString(status)));
}


"""