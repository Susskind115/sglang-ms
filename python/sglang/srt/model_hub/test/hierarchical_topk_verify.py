# test_hierarchical_topk_verify.py

import torch
import pytest
from typing import List, Dict, Tuple

def first_rank_print(*args, **kwargs):
    """统一的打印函数，兼容分布式环境"""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def hierarchical_topk_verify(
    # === 输入：与 tree_speculative_sampling_target_only 完全一致 ===
    candidates: torch.Tensor,           # [bs, num_draft_tokens]
    retrive_index: torch.Tensor,        # [bs, num_draft_tokens] 
    retrive_next_token: torch.Tensor,   # [bs, num_draft_tokens]
    retrive_next_sibling: torch.Tensor, # [bs, num_draft_tokens]
    target_probs: torch.Tensor,         # [bs, num_draft_tokens, vocab_size]
    uniform_samples: torch.Tensor,      # [bs, num_draft_tokens]
    threshold_single: float,
    threshold_acc: float,
    
    # === 新增参数：层级验证特有 ===
    topk: int,                          # 每层保留的候选数
    draft_depth_threshold: int = None,   # 深度阈值
    enable_iteration: bool = True,       # 是否启用迭代模式
    
) -> Tuple[
    # === 输出1：build_tree_kernel_efficient 输入格式 ===
    torch.Tensor,                       # verified_id
    List[torch.Tensor],                 # score_list
    List[torch.Tensor],                 # token_list  
    List[torch.Tensor],                 # parents_list
    
    # === 输出2：tree_speculative_sampling_target_only 输出格式 ===
    torch.Tensor,                       # predicts
    torch.Tensor,                       # accept_index
    torch.Tensor,                       # accept_token_num
]:
    """
    层级Top-k验证算子：支持迭代优化和精确回退的双重输出
    
    功能：
    1. 将当前草稿树通过验证模型概率分布进行质量提升
    2. 输出可直接用于下一轮 build_tree_kernel_efficient 的数据
    3. 同时输出可用于 KV 缓存回退的精确控制信息
    """
    bs, num_draft_tokens = candidates.shape
    vocab_size = target_probs.shape[-1]
    
    # === 第一阶段：层级概率修正 ===
    # 应用您的"联合候选与修正选择"算法
    enhanced_score_list = []
    enhanced_token_list = []
    enhanced_parents_list = []
    
    for depth in range(max_tree_depth):
        layer_nodes = get_nodes_at_depth(depth, retrive_next_token, retrive_next_sibling)
        
        for node_info in layer_nodes:
            # 联合候选池构建
            draft_candidates = extract_draft_candidates(node_info, candidates)
            verifier_candidates = extract_verifier_topk(node_info, target_probs, topk)
            joint_pool = merge_candidates(draft_candidates, verifier_candidates)
            
            # 概率竞争机制
            if len(joint_pool) > topk:
                selected_tokens, selected_probs = competitive_selection(
                    joint_pool, target_probs[node_info], topk
                )
            else:
                selected_tokens, selected_probs = joint_pool, get_probs(joint_pool, target_probs[node_info])
            
            enhanced_score_list.append(selected_probs)
            enhanced_token_list.append(selected_tokens)
            enhanced_parents_list.append(compute_parents(node_info, depth))
    
    # === 第二阶段：构建 build_tree_kernel_efficient 输入 ===
    verified_id = extract_root_tokens(candidates, retrive_index)
    score_list = organize_by_layers(enhanced_score_list)
    token_list = organize_by_layers(enhanced_token_list)  
    parents_list = organize_by_layers(enhanced_parents_list)
    
    # === 第三阶段：模拟 tree_speculative_sampling_target_only 输出 ===
    # 基于增强后的概率分布，计算接受情况
    predicts = torch.full((total_accepted_tokens,), -1, dtype=torch.int32, device=candidates.device)
    accept_index = torch.full((bs, max_spec_steps), -1, dtype=torch.int32, device=candidates.device)
    accept_token_num = torch.zeros(bs, dtype=torch.int32, device=candidates.device)
    
    # 使用增强后的概率执行采样决策
    perform_enhanced_sampling(
        predicts, accept_index, accept_token_num,
        enhanced_score_list, enhanced_token_list,
        uniform_samples, threshold_single, threshold_acc
    )
    
    return (
        # build_tree_kernel_efficient 输入
        verified_id, score_list, token_list, parents_list,
        # tree_speculative_sampling_target_only 输出  
        predicts, accept_index, accept_token_num
    )

def create_mock_draft_tree_structure():
    """创建模拟的草稿树结构（来自build_tree_kernel_efficient的输出）"""
    device = "cuda"
    
    # 这些数据来自build_tree_kernel_efficient的测试输出
    draft_tree_structure = {
        'retrive_index': torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ], device=device, dtype=torch.long),
        
        'retrive_next_token': torch.tensor([
            [1, 3, 4, 5, 6, 7, -1, -1],
            [1, 2, -1, 6, -1, -1, 7, -1],
        ], device=device, dtype=torch.long),
        
        'retrive_next_sibling': torch.tensor([
            [-1, 2, -1, -1, -1, -1, -1, -1],
            [-1, -1, 3, 4, 5, -1, -1, -1],
        ], device=device, dtype=torch.long),
        
        'draft_tokens': torch.tensor([
            29974, 29896, 29906, 29889, 29974, 29946, 29896, 29946,
            13, 13, 22550, 4136, 16492, 8439, 29871, 29941,
        ], device=device, dtype=torch.int32),
        
        'positions': torch.tensor([
            5, 6, 6, 7, 7, 8, 8, 9, 
            10, 11, 12, 12, 12, 12, 13, 14
        ], device=device, dtype=torch.long),
        
        # tree_mask会很大，这里简化表示
        'tree_mask': torch.ones(200, dtype=torch.bool, device=device)  # 简化
    }
    
    return draft_tree_structure


def create_mock_verifier_logits():
    """创建模拟的验证模型logits"""
    device = "cuda"
    vocab_size = 32000
    num_draft_positions = 16  # 对应draft_tokens的长度
    
    # 模拟验证模型对每个draft position的预测
    verifier_logits = torch.randn(
        num_draft_positions, vocab_size, 
        device=device, dtype=torch.float32
    )
    
    # 为了测试，我们让某些特定token有更高的概率
    target_tokens = [29974, 29896, 29906, 29889, 29974, 29946, 29896, 29946,
                    13, 13, 22550, 4136, 16492, 8439, 29871, 29941]
    
    for i, token_id in enumerate(target_tokens):
        if i < num_draft_positions and token_id < vocab_size:
            verifier_logits[i, token_id] += 5.0  # 增加目标token的logit
    
    return verifier_logits


@pytest.mark.parametrize("device", ["cuda"])
def test_hierarchical_topk_verify_basic(device):
    """基础功能测试：验证输入输出格式正确性"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # 准备测试数据
    draft_tree_structure = create_mock_draft_tree_structure()
    verifier_logits = create_mock_verifier_logits()
    seq_lens = torch.tensor([5, 10], dtype=torch.int64, device=device)
    topk = 4
    
    # 调用函数
    score_list, token_list, parents_list = hierarchical_topk_verify(
        draft_tree_structure=draft_tree_structure,
        verifier_logits=verifier_logits,
        topk=topk,
        seq_lens=seq_lens,
        draft_depth_threshold=None
    )
    
    # 验证输出格式
    assert isinstance(score_list, list), "score_list应该是List[torch.Tensor]"
    assert isinstance(token_list, list), "token_list应该是List[torch.Tensor]"
    assert isinstance(parents_list, list), "parents_list应该是List[torch.Tensor]"
    
    # 验证输出可以直接用于构建新的EagleVerifyInput
    batch_size = seq_lens.shape[0]
    for i, (scores, tokens, parents) in enumerate(zip(score_list, token_list, parents_list)):
        assert scores.shape[0] == batch_size, f"第{i}层score_list batch维度不匹配"
        assert tokens.shape[0] == batch_size, f"第{i}层token_list batch维度不匹配"
        assert parents.shape[0] == batch_size, f"第{i}层parents_list batch维度不匹配"
        
        # 每层的token数量应该不超过topk
        assert tokens.shape[1] <= topk, f"第{i}层token数量超过topk"
    
    first_rank_print("✅ 基础功能测试通过")


@pytest.mark.parametrize("device", ["cuda"])
def test_hierarchical_topk_verify_distribution_preservation(device):
    """分布保持测试：验证输出概率来源于验证模型"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    draft_tree_structure = create_mock_draft_tree_structure()
    verifier_logits = create_mock_verifier_logits()
    seq_lens = torch.tensor([5, 10], dtype=torch.int64, device=device)
    topk = 4
    
    score_list, token_list, parents_list = hierarchical_topk_verify(
        draft_tree_structure=draft_tree_structure,
        verifier_logits=verifier_logits,
        topk=topk,
        seq_lens=seq_lens
    )
    
    # 验证概率分布来源于验证模型
    for scores in score_list:
        # 检查概率和是否接近1（考虑数值误差）
        prob_sums = torch.sum(scores, dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
            "概率分布不归一化"
        
        # 检查所有概率都是非负的
        assert torch.all(scores >= 0), "存在负概率"
    
    first_rank_print("✅ 分布保持测试通过")


@pytest.mark.parametrize("device", ["cuda"])
def test_hierarchical_topk_verify_competitive_selection(device):
    """竞争机制测试：验证k1 > k时的竞争逻辑"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    draft_tree_structure = create_mock_draft_tree_structure()
    verifier_logits = create_mock_verifier_logits()
    seq_lens = torch.tensor([5, 10], dtype=torch.int64, device=device)
    
    # 使用较小的topk来触发竞争机制
    topk = 2  # 小于原始的4，应该触发竞争
    
    score_list, token_list, parents_list = hierarchical_topk_verify(
        draft_tree_structure=draft_tree_structure,
        verifier_logits=verifier_logits,
        topk=topk,
        seq_lens=seq_lens
    )
    
    # 验证输出的token数量确实被限制在topk
    for tokens in token_list:
        assert tokens.shape[-1] <= topk, f"竞争后token数量应该不超过{topk}"
    
    # 验证竞争机制保持了高概率候选
    for scores in score_list:
        # 每行的分数应该是递减的（topk选择的结果）
        for batch_idx in range(scores.shape[0]):
            row_scores = scores[batch_idx]
            # 检查是否按降序排列（允许相等）
            assert torch.all(row_scores[:-1] >= row_scores[1:]), \
                "竞争选择后的分数不是降序排列"
    
    first_rank_print("✅ 竞争机制测试通过")


@pytest.mark.parametrize("device", ["cuda"])
def test_hierarchical_topk_verify_compatibility(device):
    """兼容性测试：验证与现有管道的无缝集成"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    draft_tree_structure = create_mock_draft_tree_structure()
    verifier_logits = create_mock_verifier_logits()
    seq_lens = torch.tensor([5, 10], dtype=torch.int64, device=device)
    topk = 4
    
    # 调用我们的函数
    score_list, token_list, parents_list = hierarchical_topk_verify(
        draft_tree_structure=draft_tree_structure,
        verifier_logits=verifier_logits,
        topk=topk,
        seq_lens=seq_lens
    )
    
    # 模拟后续的build_tree_kernel_efficient调用
    # 这应该能够成功执行而不报错
    verified_id = torch.tensor([29974, 13], device=device, dtype=torch.int32)
    spec_steps = len(score_list)
    num_verify_tokens = 8
    
    try:
        # 这里我们模拟调用build_tree_kernel_efficient_preprocess
        # 实际测试中应该能调用完整的build_tree_kernel_efficient
        from build_eagle_tree_test import build_tree_kernel_efficient_preprocess
        
        parent_list, top_scores_index, draft_tokens = build_tree_kernel_efficient_preprocess(
            verified_id=verified_id,
            score_list=score_list,
            token_list=token_list,
            parents_list=parents_list,
            num_verify_tokens=num_verify_tokens,
        )
        
        # 如果执行到这里没有异常，说明格式兼容
        first_rank_print("✅ 兼容性测试通过 - 输出格式与现有管道兼容")
        
    except Exception as e:
        pytest.fail(f"兼容性测试失败：{e}")


@pytest.mark.parametrize("device", ["cuda"])
def test_hierarchical_topk_verify_depth_threshold(device):
    """深度阈值测试：验证累积深度驱动逻辑"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    draft_tree_structure = create_mock_draft_tree_structure()
    verifier_logits = create_mock_verifier_logits()
    seq_lens = torch.tensor([5, 10], dtype=torch.int64, device=device)
    topk = 4
    
    # 测试不同的深度阈值
    for threshold in [2, 4, 6]:
        score_list, token_list, parents_list = hierarchical_topk_verify(
            draft_tree_structure=draft_tree_structure,
            verifier_logits=verifier_logits,
            topk=topk,
            seq_lens=seq_lens,
            draft_depth_threshold=threshold
        )
        
        # 验证输出层数合理（应该与阈值相关）
        assert len(score_list) >= 1, f"阈值{threshold}时输出层数过少"
        assert len(score_list) == len(token_list) == len(parents_list), \
            f"阈值{threshold}时三个输出列表长度不一致"
    
    first_rank_print("✅ 深度阈值测试通过")


def test_hierarchical_topk_verify_edge_cases():
    """边界情况测试"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = "cuda"
    
    # 测试单批次情况
    single_batch_tree = create_mock_draft_tree_structure()
    # 修改为单批次
    for key in ['retrive_index', 'retrive_next_token', 'retrive_next_sibling']:
        single_batch_tree[key] = single_batch_tree[key][:1]  # 只保留第一行
    
    single_batch_tree['draft_tokens'] = single_batch_tree['draft_tokens'][:8]
    single_batch_tree['positions'] = single_batch_tree['positions'][:8]
    
    verifier_logits = create_mock_verifier_logits()[:8]  # 匹配tokens数量
    seq_lens = torch.tensor([5], dtype=torch.int64, device=device)
    
    score_list, token_list, parents_list = hierarchical_topk_verify(
        draft_tree_structure=single_batch_tree,
        verifier_logits=verifier_logits,
        topk=4,
        seq_lens=seq_lens
    )
    
    assert len(score_list) > 0, "单批次情况下应该有输出"
    first_rank_print("✅ 边界情况测试通过")


if __name__ == "__main__":
    """运行所有测试"""
    first_rank_print("开始运行 hierarchical_topk_verify 测试套件...")
    
    # 注意：这些测试现在会失败，因为hierarchical_topk_verify还没实现
    # 但它们定义了清晰的接口和预期行为
    
    try:
        test_hierarchical_topk_verify_basic("cuda")
        test_hierarchical_topk_verify_distribution_preservation("cuda")
        test_hierarchical_topk_verify_competitive_selection("cuda")
        test_hierarchical_topk_verify_compatibility("cuda")
        test_hierarchical_topk_verify_depth_threshold("cuda")
        test_hierarchical_topk_verify_edge_cases()
        
        first_rank_print("🎉 所有测试通过！")
        
    except Exception as e:
        first_rank_print(f"❌ 测试失败：{e}")
        first_rank_print("这是预期的，因为 hierarchical_topk_verify 还未实现")
        first_rank_print("请按照测试用例的接口要求实现该函数")