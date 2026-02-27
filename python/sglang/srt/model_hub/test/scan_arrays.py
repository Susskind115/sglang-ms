


import torch
import logging
logger = logging.getLogger(__name__)

def check_consistency(drafted_id, verify_output, accept_length, accept_token):
    """
    对比 Drafted ID (按 accept_length 截取后) 与 Verify Output 是否一致。
    """
    
    # --- 步骤 1: 将 drafted_id "压扁" 成一维 (Flatten with Mask) ---
    
    batch_size, max_len = drafted_id.shape
    
    # 1.1 创建列索引矩阵 [0, 1, 2, ..., max_len-1]
    col_indices = torch.arange(max_len, device=drafted_id.device).unsqueeze(0)
    
    # 1.2 扩展 accept_length 维度以进行广播
    lengths = accept_length.unsqueeze(1)
    
    # 1.3 生成掩码：只保留列号小于 accept_length 的位置
    # mask shape: [batch_size, max_len]
    mask = col_indices < lengths
    
    # 1.4 使用掩码提取数据
    # flat_draft 将是一个一维张量，包含了所有被接受的 token，顺序与 verify_output 理论上一致
    flat_draft = drafted_id[mask]
    
    
    # --- 步骤 2: 基础维度检查 ---
    
    if flat_draft.shape[0] != verify_output.shape[0]:
        print(f"❌ 严重错误: 元素总数不匹配!")
        print(f"Draft 有效总数: {flat_draft.shape[0]} (根据 accept_length 计算)")
        print(f"Verify 输出总数: {verify_output.shape[0]}")
        return

    # --- 步骤 3: 数值对比与错误定位 ---
    
    # 比较两个一维张量
    mismatch_mask = (flat_draft != verify_output)
    
    if not mismatch_mask.any():
        print("✅ 验证通过: Draft 与 Verify Output 完全一致。")
    else:
        print("❌ 发现数据不一致！开始定位错误...")
        
        # 为了给出友好的 Row/Col 提示，我们需要还原回 Batch 视角
        # 利用 cumsum 计算 verify_output 中每一段数据的起始位置
        # print(f"drafted_id: {drafted_id}, verify_output: {verify_output}, accept_length: {accept_length}, accept_token: {accept_token}")
        offsets = torch.zeros_like(accept_length)
        offsets[1:] = torch.cumsum(accept_length[:-1], dim=0)
        
        # 遍历每一行检查 (只在出错时执行，不影响正常性能)
        for i in range(batch_size):
            length = accept_length[i].item()
            if length == 0: continue
            
            # 取出 verify_output 中对应这一行的片段
            start = offsets[i].item()
            end = start + length
            verify_segment = verify_output[start:end]
            
            # 取出 drafted_id 中对应这一行的片段
            draft_segment = drafted_id[i, :length]
            
            # 对比片段
            if not torch.equal(draft_segment, verify_segment):
                print(f"\n[Row {i}] 发生不匹配:")
                print(f"  Expect (Draft) : {draft_segment.tolist()}")
                print(f"  Actual (Verify): {verify_segment.tolist()}")
                
                # 找出具体的列号
                diff_indices = torch.nonzero(draft_segment != verify_segment).flatten().tolist()
                print(f"  错误列号 (Col): {diff_indices}")


def flatten_accepted_data(drafted_id, out_loc_cache, accept_length):
    """
    根据 accept_length 提取 drafted_id 和 out_loc_cache 中的有效数据，
    并将其展平为一维张量。
    """
    # 1. 获取维度信息
    batch_size, max_len = drafted_id.shape
    
    # 2. 生成布尔掩码 (Boolean Mask)
    # col_indices: [0, 1, 2, ..., max_len-1] (扩充为 [1, max_len])
    col_indices = torch.arange(max_len, device=drafted_id.device).unsqueeze(0)
    
    # lengths: [batch_size, 1]
    lengths = accept_length.unsqueeze(1)
    
    # mask: [batch_size, max_len]
    # 逻辑：如果列索引 < accept_length，则为 True (保留)，否则为 False (丢弃)
    mask = col_indices < lengths
    
    # 3. 应用掩码提取数据 (Flattening)
    # PyTorch 的 mask 索引会自动返回一维结果
    flat_draft_tokens = drafted_id[mask]
    flat_out_locs = out_loc_cache[mask]
    
    return flat_draft_tokens, flat_out_locs



def find_diff_indices(tensor_a, tensor_b, atol=1e-5, log_info=True):
    """
    对比两个 [N, H, D] 张量，找出哪个 N (个体) 有区别
    :param atol: 绝对容差 (absolute tolerance)，浮点数比较建议留一点余地
    """
    if tensor_a.ndim == 2:
        tensor_a = tensor_a.unsqueeze(0)
        tensor_b = tensor_b.unsqueeze(0)
    # 1. 计算两个张量的绝对差值
    diff = torch.abs(tensor_a - tensor_b)
    
    # 2. 降维检查：将后面两个维度 (4, 64) 展平，并计算每个个体的最大差异值
    # view(6, -1) 把 [6, 4, 64] 变成了 [6, 256]
    # max(dim=1).values 得到形状 [6]，表示每个个体内部最大的那个误差值
    max_diff_per_item = diff.view(diff.shape[0], -1).max(dim=1).values
    
    # 3. 找出差异超过容差的索引 (Indices)
    # torch.nonzero 返回非零元素的坐标，squeeze 把多余的维度去掉
    diff_indices = torch.nonzero(max_diff_per_item > atol).squeeze()
    
    # --- 打印结果 ---
    if diff_indices.numel() == 0:
        if log_info:
            logger.info("✅ 两个张量完全一致 (所有个体误差均在容差范围内)。")
        return True
    else:
        # 统一转成 list 方便查看，处理只有一个差异的情况
        indices_list = diff_indices.tolist()
        if isinstance(indices_list, int): indices_list = [indices_list]
        if log_info:
            logger.info(f"❌ 发现不一致的个体索引: {indices_list}")
        
            # 顺便打印出具体差别有多大
            for idx in indices_list:   
                logger.info(f"   -> Index {idx}: 最大误差 = {max_diff_per_item[idx].item():.6f}")
        return False

# def check_draft_alignment(drafted_id, accepted_token):
#     """
#     对比 drafted_id 和 accepted_token。
#     逻辑：在 drafted_id != 0 的位置，检查两个张量的值是否相等。
#     """
    
#     # 1. 维度对齐 (Slicing)
#     # accepted_token 可能比 drafted_id 长，我们只截取和 drafted_id 相同的列数进行对比
#     cols = drafted_id.shape[1]
    
#     # 防御性编程：确保 accepted_token 至少覆盖了 drafted_id 的长度
#     # 如果 accepted 比 draft 短，说明逻辑有问题，直接报错或截断
#     if accepted_token.shape[1] < cols:
#         raise ValueError(f"Accepted token width ({accepted_token.shape[1]}) is smaller than draft ({cols})")
        
#     # 截取 accepted_token 的前 cols 列
#     accepted_slice = accepted_token[:, :cols]
    
#     # 2. 创建有效掩码 (Masking)
#     # 只有 drafted_id 不为 0 的地方才需要对比
#     valid_mask = (drafted_id != 0)
    
#     # 3. 计算不匹配的地方 (Mismatch Detection)
#     # 逻辑：(数值不相等) AND (draft 是有效值)
#     # 结果是一个布尔矩阵，True 表示 "这个位置 Draft 预测了某个词，但被拒绝了"
#     mismatch_mask = (drafted_id != accepted_slice) & valid_mask
    
#     # 4. 得出结论
    
#     # 方式 A: 全局判断 (只要有一个位置对不上，就返回 False)
#     is_all_aligned = not mismatch_mask.any()
    
#     # 方式 B: 按行判断 (返回一个 shape=[batch_size] 的布尔张量，指示哪些行是完全对上的)
#     # 只要某一行存在 mismatch (any(dim=1))，该行就是 False (未对齐)
#     row_is_aligned = ~mismatch_mask.any(dim=1)
    
#     return is_all_aligned, row_is_aligned, mismatch_mask


def report_mismatches(drafted_id, accepted_token):
    # 1. 维度对齐：截取 accepted_token 以匹配 drafted_id 的宽度
    cols = drafted_id.shape[1]
    target = accepted_token[:, :cols]
    
    # 2. 生成错误掩码 (Error Mask)
    # 逻辑：(draft 不是 padding) AND (draft 不等于 accepted)
    # 这是一个布尔矩阵，True 代表出错的位置
    mismatch_mask = (drafted_id != 0) & (drafted_id != target)
    
    # 3. 提取坐标 (Extract Coordinates)
    # torch.nonzero 返回一个形状为 [N, 2] 的张量，每一行都是一个 (row, col) 坐标
    # as_tuple=False 确保返回二维张量
    error_indices = torch.nonzero(mismatch_mask, as_tuple=False)
    
    # 4. 报告错误
    if error_indices.shape[0] == 0:
        print("✅ Perfect Match! 没有发现错误。")
        return
    
    print(f"❌ 发现 {error_indices.shape[0]} 处不匹配:")
    print(f"{'Row':<5} | {'Col':<5} | {'Draft':<10} | {'Accept':<10}")
    print("-" * 40)
    
    # 遍历错误索引并打印详情
    for idx in error_indices:
        row, col = idx[0].item(), idx[1].item()
        
        draft_val = drafted_id[row, col].item()
        accept_val = target[row, col].item()
        
        print(f"{row:<5} | {col:<5} | {draft_val:<10} | {accept_val:<10}")
