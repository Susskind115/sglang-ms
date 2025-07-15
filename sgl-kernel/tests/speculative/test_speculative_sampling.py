import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import tree_speculative_sampling_target_only

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

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )
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