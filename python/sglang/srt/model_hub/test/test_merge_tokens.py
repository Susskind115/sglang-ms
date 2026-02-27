import torch

# ==========================================
# 1. 准备数据 (模拟您的输入)
# ==========================================
# 假设总长度 12，Batch Size 4
drafted_id = torch.tensor([13, 29930, 450, 6511, 278, 396, 4910, 680, 278, 6590, 3161, 736], device='cuda:0', dtype=torch.long)
verified_id = torch.tensor([29930, 450, 421, 278, 740, 4910, 680, 278, 6590, 3161, 310, 1051], device='cuda:0', dtype=torch.long)
# accept_length 之和必须等于 drafted_id 的长度 (3+2+6+1 = 12)
accept_length = torch.tensor([3, 2, 6, 1], device='cuda:0', dtype=torch.int32)

# ==========================================
# 2. 高效融合逻辑 (Core Logic)
# ==========================================

# A. 计算元数据
batch_size = accept_length.size(0)
total_draft_len = drafted_id.size(0)
# 新的总长度 = 原长度 + 每个请求多出的1个红利Token
total_new_len = total_draft_len + batch_size 

# B. 预分配输出显存 (避免动态扩容)
# 使用 empty 分配最快，不需要 zero 初始化
final_tensor = torch.empty(total_new_len, device=drafted_id.device, dtype=drafted_id.dtype)

# C. 定位 "红利 Token" 在 verified_id 中的位置 (Source Indices)
# 计算累加和，得到每个片段的结束位置
# accept_cumsum: [3, 5, 11, 12]
accept_cumsum = torch.cumsum(accept_length, dim=0)
# 减1得到下标: [2, 4, 10, 11]
bonus_src_indices = accept_cumsum - 1 

# D. 定位 "红利 Token" 在 新张量 中的位置 (Destination Indices)
# 新的每个请求长度 = accept_length + 1
# new_cumsum: [4, 7, 14, 16]
new_cumsum = torch.cumsum(accept_length + 1, dim=0)
# 减1得到新张量中的落座下标: [3, 6, 13, 15]
bonus_dest_indices = new_cumsum - 1

# E. 填入 "红利 Token" (Scatter)
# 直接从 verified_id 拿出数据填入 final_tensor 的指定位置
# 这一步只需一次 Kernel Launch
final_tensor[bonus_dest_indices] = verified_id[bonus_src_indices]

# F. 填入 "Draft Token" (Masked Fill)
# 我们需要把 drafted_id 填入剩下的空位。
# 创建一个全 True 的掩码
mask = torch.ones(total_new_len, device=drafted_id.device, dtype=torch.bool)
# 把刚才填了红利 Token 的位置设为 False
mask[bonus_dest_indices] = False
# 利用布尔索引一次性填入所有 draft token
final_tensor[mask] = drafted_id

# ==========================================
# 3. 验证结果
# ==========================================
print("融合后的张量:", final_tensor)
print("预期总长度:", total_new_len)
print("实际总长度:", final_tensor.shape[0])

# 验证逻辑：
# 请求1 (len 3+1): 13, 29930, 450 (Draft) + 421 (Verified尾部) -> 正确
# 请求2 (len 2+1): 6511, 278 (Draft) + 740 (Verified尾部) -> 正确
# ...