import torch
from dataclasses import dataclass
import copy # 用于深度复制对象状态
import time

# ---------------------------------------------------
# 1. 模拟 EagleVerifyOutput 的数据类
# ---------------------------------------------------
@dataclass
class MockEagleVerifyOutput:
    """模拟的验证输出对象"""
    verified_id: torch.Tensor
    accept_length_per_req_cpu: list[int]

# ---------------------------------------------------
# 2. 包含两种实现的主机类
# ---------------------------------------------------
class SubmissionCache:
    """
    一个用于存储和更新已接受token的缓存对象。
    包含 'loop' 和 'vectorized' 两种更新实现以供对比。
    """
    def __init__(self, src_req_pool_indices: torch.Tensor, spec_draft_num: int, device: torch.device):
        self.device = device
        self.spec_draft_num = spec_draft_num # 即 max_len
        self.src_req_pool_indices = src_req_pool_indices.to(device)
        
        B_total = src_req_pool_indices.shape[0]
        
        # 初始填充 -1, 便于观察
        self.accepted_token = torch.full((B_total, spec_draft_num), -1, dtype=torch.long, device=device)
        self.accepted_token_num = torch.zeros(B_total, dtype=torch.long, device=device)

    def clone(self) -> 'SubmissionCache':
        """创建一个状态完全相同的副本，用于对比测试"""
        new_cache = SubmissionCache(self.src_req_pool_indices, self.spec_draft_num, self.device)
        new_cache.accepted_token = self.accepted_token.clone()
        new_cache.accepted_token_num = self.accepted_token_num.clone()
        return new_cache

    # ---
    # 版本一: 循环实现
    # ---
    def update_from_verify_output_loop(self, verify_output: MockEagleVerifyOutput, req_pool_indices: torch.Tensor):
        
        device = self.src_req_pool_indices.device
        
        accept_length_cpu = verify_output.accept_length_per_req_cpu
        if not accept_length_cpu: # 处理空列表
            return
            
        total_accepted_this_step = torch.tensor(accept_length_cpu, device=device, dtype=torch.long) + 1
        verified_tokens = verify_output.verified_id 

        if verified_tokens.shape[0] == 0:
             return
             
        tokens_per_req = torch.split(verified_tokens, total_accepted_this_step.tolist())

        # 向量化查找 "master_idx"
        master = self.src_req_pool_indices.unsqueeze(1)
        current = req_pool_indices.unsqueeze(0)
        comparison = (master == current)
        row_indices, col_indices = comparison.nonzero(as_tuple=True)
        
        master_indices_for_current = torch.empty_like(row_indices)
        master_indices_for_current[col_indices] = row_indices

        # 循环并更新
        max_len = self.spec_draft_num
        
        for i, master_idx in enumerate(master_indices_for_current):
            tokens_to_add = tokens_per_req[i]
            num_to_add_orig = tokens_to_add.shape[0] # 即 total_accepted_this_step[i]
            
            start_col = self.accepted_token_num[master_idx]
            end_col = start_col + num_to_add_orig
            
            num_to_add_final = num_to_add_orig
            
            # (重要) 边界检查和截断
            if end_col > max_len:
                num_to_add_final = max_len - start_col
                if num_to_add_final < 0:
                    num_to_add_final = 0
                
                tokens_to_add = tokens_to_add[:num_to_add_final]
                end_col = max_len # 澄清：num 设为最大值
            
            if num_to_add_final <= 0:
                # 即使没有添加（因为已满），也要确保 num 被设为 max_len (如果触发了截断)
                if end_col > max_len:
                    self.accepted_token_num[master_idx] = max_len
                continue 

            self.accepted_token[master_idx, start_col:end_col] = tokens_to_add
            self.accepted_token_num[master_idx] = end_col


    # ---
    # 版本二: 向量化实现
    # ---
    def update_from_verify_output_vectorized(self, verify_output: MockEagleVerifyOutput, req_pool_indices: torch.Tensor):
        
        device = self.src_req_pool_indices.device
        max_len = self.spec_draft_num

        accept_length_cpu = verify_output.accept_length_per_req_cpu
        # 处理 B_current = 0 的情况
        if not accept_length_cpu:
            return
            
        total_accepted_this_step = torch.tensor(accept_length_cpu, device=device, dtype=torch.long) + 1
        verified_tokens = verify_output.verified_id 
        
        B_current = total_accepted_this_step.shape[0]
        K_total = verified_tokens.shape[0]
        
        # 处理 K_total = 0 的情况
        if K_total == 0:
            return

        # 索引映射
        master = self.src_req_pool_indices.unsqueeze(1)
        current = req_pool_indices.unsqueeze(0)
        comparison = (master == current)
        row_indices, col_indices = comparison.nonzero(as_tuple=True)
        master_indices_for_current = torch.empty_like(row_indices)
        master_indices_for_current[col_indices] = row_indices

        # 向量化边界检查
        start_cols = self.accepted_token_num[master_indices_for_current]
        end_cols_proposed = start_cols + total_accepted_this_step
        # 澄清：num 设为最大值 (clamp 完美实现了这一点)
        end_cols_final = torch.clamp(end_cols_proposed, max=max_len)
        num_to_add_final = torch.clamp(end_cols_final - start_cols, min=0)

        # 过滤与索引生成
        relative_col_idx = torch.cat([
            torch.arange(n, device=device) for n in total_accepted_this_step.tolist()
        ])
        group_id_for_each_token = torch.repeat_interleave(
            torch.arange(B_current, device=device), 
            total_accepted_this_step
        )
        limit_per_token = num_to_add_final[group_id_for_each_token]
        mask = (relative_col_idx < limit_per_token)
        filtered_tokens = verified_tokens[mask]
        
        # 最终写入与更新
        if filtered_tokens.shape[0] > 0:
            flat_row_indices_orig = torch.repeat_interleave(
                master_indices_for_current, 
                total_accepted_this_step
            )
            flat_row_indices = flat_row_indices_orig[mask]

            start_cols_orig = torch.repeat_interleave(start_cols, total_accepted_this_step)
            flat_col_indices_orig = start_cols_orig + relative_col_idx
            flat_col_indices = flat_col_indices_orig[mask]

            self.accepted_token[flat_row_indices, flat_col_indices] = filtered_tokens
        
        # 向量化更新计数器
        self.accepted_token_num[master_indices_for_current] = end_cols_final


# ---------------------------------------------------
# 3. 测试执行器
# ---------------------------------------------------

def run_test_case(test_name: str, initial_cache: SubmissionCache, update_data: dict):
    """
    运行一个测试用例，对比 loop 和 vectorized 版本的结果。
    """
    print(f"--- 运行测试: {test_name} ---")
    
    # 解包更新数据
    req_pool_indices = update_data["req_pool_indices"]
    verify_output = update_data["verify_output"]

    # 1. 创建两个相同的初始状态
    cache_loop = initial_cache.clone()
    cache_vec = initial_cache.clone()
    
    # 2. 运行两个版本的函数
    try:
        cache_loop.update_from_verify_output_loop(verify_output, req_pool_indices)
    except Exception as e:
        print(f"[FAIL] Loop 版本运行时出错: {e}")
        return

    try:
        cache_vec.update_from_verify_output_vectorized(verify_output, req_pool_indices)
    except Exception as e:
        print(f"[FAIL] Vectorized 版本运行时出错: {e}")
        return
            
    # 3. 比较结果
    token_match = torch.all(cache_loop.accepted_token == cache_vec.accepted_token)
    num_match = torch.all(cache_loop.accepted_token_num == cache_vec.accepted_token_num)
    
    if token_match and num_match:
        print(f"✅ [PASS] 结果一致!")
        print(f"  最终 token_num: {cache_vec.accepted_token_num.cpu().tolist()}")
        print(f"  最终 token 状态:\n{cache_vec.accepted_token.cpu().numpy()}")
    else:
        print(f"❌ [FAIL] 结果不一致!")
        if not token_match:
            print("  accepted_token 不匹配:")
            print("    Loop版:\n", cache_loop.accepted_token.cpu().numpy())
            print("    Vec版:\n", cache_vec.accepted_token.cpu().numpy())
        if not num_match:
            print("  accepted_token_num 不匹配:")
            print("    Loop版:", cache_loop.accepted_token_num.cpu().tolist())
            print("    Vec版:", cache_vec.accepted_token_num.cpu().tolist())
    
    print("-" * 40 + "\n")


# ---------------------------------------------------
# 4. 主执行函数
# ---------------------------------------------------
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # ---
    # 定义基础状态 (5个请求, 最大长度为 5)
    # ---
    src_req_pool_indices = torch.tensor([100, 101, 102, 103, 104], device=device)
    spec_draft_num = 5
    base_cache = SubmissionCache(src_req_pool_indices, spec_draft_num, device)

    # ---
    # 测试用例 1: 正常更新
    # 更新请求 101 (idx 1) 和 103 (idx 3)
    # 101: 额外接受 1 (总共 2)
    # 103: 额外接受 2 (总共 3)
    # ---
    update_1_reqs = torch.tensor([101, 103], device=device)
    update_1_verify = MockEagleVerifyOutput(
        verified_id=torch.tensor([11, 12, 31, 32, 33], device=device), # 2 + 3 = 5 个 tokens
        accept_length_per_req_cpu=[1, 2] # [101], [103]
    )
    run_test_case(
        "Test 1: 正常更新", 
        base_cache, 
        {"req_pool_indices": update_1_reqs, "verify_output": update_1_verify}
    )

    # ---
    # 测试用例 2: 截断更新
    # 我们需要一个有预设值的 cache
    # ---
    cache_for_test_2 = base_cache.clone()
    # 102 (idx 2) 已经有 3 个 tokens
    cache_for_test_2.accepted_token[2] = torch.tensor([21, 22, 23, -1, -1], device=device)
    cache_for_test_2.accepted_token_num[2] = 3
    # 104 (idx 4) 已经有 4 个 tokens
    cache_for_test_2.accepted_token[4] = torch.tensor([41, 42, 43, 44, -1], device=device)
    cache_for_test_2.accepted_token_num[4] = 4

    # 更新请求 102 (idx 2) 和 104 (idx 4)
    # 102: 额外接受 1 (总共 2). start=3. end=5. 截断=False. 添加 [91, 92]
    # 104: 额外接受 2 (总共 3). start=4. end=7. 截断=True. num=5. 只添加 [93]
    update_2_reqs = torch.tensor([102, 104], device=device)
    update_2_verify = MockEagleVerifyOutput(
        verified_id=torch.tensor([91, 92, 93, 94, 95], device=device), # 2 + 3 = 5 个 tokens
        accept_length_per_req_cpu=[1, 2] # [102], [104]
    )
    run_test_case(
        "Test 2: 截断更新",
        cache_for_test_2,
        {"req_pool_indices": update_2_reqs, "verify_output": update_2_verify}
    )

    # ---
    # 测试用例 3: 更新已满的请求
    # ---
    cache_for_test_3 = base_cache.clone()
    # 101 (idx 1) 已经满了
    cache_for_test_3.accepted_token[1] = torch.tensor([1, 2, 3, 4, 5], device=device)
    cache_for_test_3.accepted_token_num[1] = 5

    # 更新请求 101 (idx 1)
    # 101: 额外接受 1 (总共 2). start=5. end=7. 截断=True. num=5. 添加 0 个.
    update_3_reqs = torch.tensor([101], device=device)
    update_3_verify = MockEagleVerifyOutput(
        verified_id=torch.tensor([98, 99], device=device), # 2 个 tokens
        accept_length_per_req_cpu=[1] # [101]
    )
    run_test_case(
        "Test 3: 更新已满的请求",
        cache_for_test_3,
        {"req_pool_indices": update_3_reqs, "verify_output": update_3_verify}
    )

    # ---
    # 测试用例 4: 空更新 (B_current = 0)
    # ---
    update_4_reqs = torch.tensor([], device=device, dtype=torch.long)
    update_4_verify = MockEagleVerifyOutput(
        verified_id=torch.tensor([], device=device, dtype=torch.long),
        accept_length_per_req_cpu=[]
    )
    run_test_case(
        "Test 4: 空更新 (B_current=0)",
        base_cache,
        {"req_pool_indices": update_4_reqs, "verify_output": update_4_verify}
    )


    # ---------------------------------------------------
    # 测试用例 5: 高 Batch Size 性能测试
    # ---------------------------------------------------
    print(f"--- 运行测试: Test 5: 高 Batch Size 性能测试 ---")
    
    B_TOTAL = 5000       # 总请求数
    B_CURRENT = 4000     # 本次更新的请求数
    SPEC_DRAFT_NUM = 32  # 最大长度
    N_REPEATS = 10       # 重复次数
    N_WARMUP = 2         # 预热次数

    print(f"测试配置: B_total={B_TOTAL}, B_current={B_CURRENT}, Spec_draft_num={SPEC_DRAFT_NUM}, Repeats={N_REPEATS}")

    # 1. 创建大规模的基础数据 (只创建一次)
    try:
        perf_indices = torch.arange(B_TOTAL, device=device)
        perf_base_cache = SubmissionCache(perf_indices, SPEC_DRAFT_NUM, device)
        
        # 随机选择 B_CURRENT 个请求进行更新
        perf_reqs = torch.randperm(B_TOTAL, device=device)[:B_CURRENT]
        
        # 模拟接受长度 (0-4个额外token)
        perf_accept_lens_cpu = torch.randint(0, 5, (B_CURRENT,)).tolist()
        perf_total_accepted = torch.tensor(perf_accept_lens_cpu, device=device) + 1
        
        K_total = perf_total_accepted.sum().item()
        
        # 模拟所有被接受的 token
        perf_verified_id = torch.randint(1000, 30000, (K_total,), device=device, dtype=torch.long)
        
        perf_verify_output = MockEagleVerifyOutput(
            verified_id=perf_verified_id,
            accept_length_per_req_cpu=perf_accept_lens_cpu
        )
        print(f"数据生成完毕。本次更新总token数 (K_total): {K_total}")

    except Exception as e:
        print(f"❌ [FAIL] 性能测试数据生成失败: {e}")
        if "out of memory" in str(e):
            print("  错误：GPU 显存不足。请尝试减小 B_TOTAL 或 B_CURRENT。")
        exit() # 数据生成失败，无法继续


    # 2. 计时 - 循环版本
    print("\n正在测试 Loop 版本...")
    loop_timings = []
    # 预热
    for _ in range(N_WARMUP):
        cache_copy = perf_base_cache.clone()
        cache_copy.update_from_verify_output_loop(perf_verify_output, perf_reqs)
        if device.type == 'cuda': torch.cuda.synchronize()
        
    # 正式计时
    start_time_loop = time.perf_counter()
    for i in range(N_REPEATS):
        cache_copy = perf_base_cache.clone() # 重置状态
        
        iter_start = time.perf_counter()
        cache_copy.update_from_verify_output_loop(perf_verify_output, perf_reqs)
        if device.type == 'cuda': 
            torch.cuda.synchronize() # 强制等待GPU
        iter_end = time.perf_counter()
        loop_timings.append(iter_end - iter_start)
        
    avg_time_loop = sum(loop_timings) / N_REPEATS
    print(f"Loop 版本平均耗时: {avg_time_loop * 1000:.4f} ms")


    # 3. 计时 - 向量化版本
    print("\n正在测试 Vectorized 版本...")
    vec_timings = []
    # 预热
    for _ in range(N_WARMUP):
        cache_copy = perf_base_cache.clone()
        cache_copy.update_from_verify_output_vectorized(perf_verify_output, perf_reqs)
        if device.type == 'cuda': torch.cuda.synchronize()

    # 正式计时
    start_time_vec = time.perf_counter()
    for i in range(N_REPEATS):
        cache_copy = perf_base_cache.clone() # 重置状态
        
        iter_start = time.perf_counter()
        cache_copy.update_from_verify_output_vectorized(perf_verify_output, perf_reqs)
        if device.type == 'cuda': 
            torch.cuda.synchronize() # 强制等待GPU
        iter_end = time.perf_counter()
        vec_timings.append(iter_end - iter_start)

    avg_time_vec = sum(vec_timings) / N_REPEATS
    print(f"Vectorized 版本平均耗时: {avg_time_vec * 1000:.4f} ms")

    # 4. 结果对比
    print("\n--- 性能对比结果 ---")
    if avg_time_vec > 0:
        speedup = avg_time_loop / avg_time_vec
        print(f"✅ [PASS] 向量化版本是循环版本的 {speedup:.2f} 倍。")
    else:
        print("ℹ️ Vectorized 版本运行时间过短，无法计算加速比。")
    print("-" * 40 + "\n")


    # ---------------------------------------------------
    # 测试用例 6: 中 Batch Size 性能测试
    # ---------------------------------------------------
    print(f"--- 运行测试: Test 5: 高 Batch Size 性能测试 ---")
    
    B_TOTAL = 128       # 总请求数
    B_CURRENT = 30     # 本次更新的请求数
    SPEC_DRAFT_NUM = 6  # 最大长度
    N_REPEATS = 10       # 重复次数
    N_WARMUP = 2         # 预热次数

    print(f"测试配置: B_total={B_TOTAL}, B_current={B_CURRENT}, Spec_draft_num={SPEC_DRAFT_NUM}, Repeats={N_REPEATS}")

    # 1. 创建大规模的基础数据 (只创建一次)
    try:
        perf_indices = torch.arange(B_TOTAL, device=device)
        perf_base_cache = SubmissionCache(perf_indices, SPEC_DRAFT_NUM, device)
        
        # 随机选择 B_CURRENT 个请求进行更新
        perf_reqs = torch.randperm(B_TOTAL, device=device)[:B_CURRENT]
        
        # 模拟接受长度 (0-4个额外token)
        perf_accept_lens_cpu = torch.randint(0, 5, (B_CURRENT,)).tolist()
        perf_total_accepted = torch.tensor(perf_accept_lens_cpu, device=device) + 1
        
        K_total = perf_total_accepted.sum().item()
        
        # 模拟所有被接受的 token
        perf_verified_id = torch.randint(1000, 30000, (K_total,), device=device, dtype=torch.long)
        
        perf_verify_output = MockEagleVerifyOutput(
            verified_id=perf_verified_id,
            accept_length_per_req_cpu=perf_accept_lens_cpu
        )
        print(f"数据生成完毕。本次更新总token数 (K_total): {K_total}")

    except Exception as e:
        print(f"❌ [FAIL] 性能测试数据生成失败: {e}")
        if "out of memory" in str(e):
            print("  错误：GPU 显存不足。请尝试减小 B_TOTAL 或 B_CURRENT。")
        exit() # 数据生成失败，无法继续


    # 2. 计时 - 循环版本
    print("\n正在测试 Loop 版本...")
    loop_timings = []
    # 预热
    for _ in range(N_WARMUP):
        cache_copy = perf_base_cache.clone()
        cache_copy.update_from_verify_output_loop(perf_verify_output, perf_reqs)
        if device.type == 'cuda': torch.cuda.synchronize()
        
    # 正式计时
    start_time_loop = time.perf_counter()
    for i in range(N_REPEATS):
        cache_copy = perf_base_cache.clone() # 重置状态
        
        iter_start = time.perf_counter()
        cache_copy.update_from_verify_output_loop(perf_verify_output, perf_reqs)
        if device.type == 'cuda': 
            torch.cuda.synchronize() # 强制等待GPU
        iter_end = time.perf_counter()
        loop_timings.append(iter_end - iter_start)
        
    avg_time_loop = sum(loop_timings) / N_REPEATS
    print(f"Loop 版本平均耗时: {avg_time_loop * 1000:.4f} ms")


    # 3. 计时 - 向量化版本
    print("\n正在测试 Vectorized 版本...")
    vec_timings = []
    # 预热
    for _ in range(N_WARMUP):
        cache_copy = perf_base_cache.clone()
        cache_copy.update_from_verify_output_vectorized(perf_verify_output, perf_reqs)
        if device.type == 'cuda': torch.cuda.synchronize()

    # 正式计时
    start_time_vec = time.perf_counter()
    for i in range(N_REPEATS):
        cache_copy = perf_base_cache.clone() # 重置状态
        
        iter_start = time.perf_counter()
        cache_copy.update_from_verify_output_vectorized(perf_verify_output, perf_reqs)
        if device.type == 'cuda': 
            torch.cuda.synchronize() # 强制等待GPU
        iter_end = time.perf_counter()
        vec_timings.append(iter_end - iter_start)

    avg_time_vec = sum(vec_timings) / N_REPEATS
    print(f"Vectorized 版本平均耗时: {avg_time_vec * 1000:.4f} ms")

    # 4. 结果对比
    print("\n--- 性能对比结果 ---")
    if avg_time_vec > 0:
        speedup = avg_time_loop / avg_time_vec
        print(f"✅ [PASS] 向量化版本是循环版本的 {speedup:.2f} 倍。")
    else:
        print("ℹ️ Vectorized 版本运行时间过短，无法计算加速比。")
    print("-" * 40 + "\n")