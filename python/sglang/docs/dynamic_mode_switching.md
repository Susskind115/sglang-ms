# 动态推理模式切换指南

本文档介绍如何在SGLang中实现投机推理（Speculative）和自回归推理（Autoregressive）之间的动态切换。

## 概述

SGLang现在支持在运行时动态切换推理模式，无需重启服务器。这对于以下场景特别有用：

- **负载自适应**：根据当前负载选择最优模式
- **性能调优**：实时比较不同模式的性能
- **资源管理**：根据内存使用情况调整并发策略

## 核心设计

### 内存分配策略

为了支持动态切换，系统采用**保守内存分配策略**：

1. **初始化时**：按照两种模式中更大的内存需求分配资源
2. **运行时**：通过动态调整`max_micro_batch_size`控制实际并发度
3. **模式切换**：不重新分配内存，只调整调度参数

### 并发控制

| 模式 | max_running_requests | 内存分配 | 实际并发控制 |
|------|---------------------|----------|-------------|
| Autoregressive | 4097 | 4097 | max_micro_batch_size |
| Speculative | 48 | 4097 | max_micro_batch_size = 48 |

## 使用方法

### 1. 启动服务器

启动时需要启用动态切换功能：

```bash
python -m sglang.launch_server \
    --model-path your_model_path \
    --enable-dynamic-mode-switching \
    --max-running-requests 4097
```

### 2. API调用

#### 切换推理模式

```bash
curl -X POST http://localhost:30000/switch_inference_mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "speculative"}'
```

响应：
```json
{
  "success": true,
  "message": "Successfully switched to speculative mode",
  "current_mode": "speculative"
}
```

#### 获取当前状态

```bash
curl http://localhost:30000/get_server_info
```

### 3. Python客户端

```python
import requests

# 切换到投机推理模式
response = requests.post(
    "http://localhost:30000/switch_inference_mode",
    json={"mode": "speculative"}
)

if response.json()["success"]:
    print("Successfully switched to speculative mode")

# 切换到自回归模式
response = requests.post(
    "http://localhost:30000/switch_inference_mode", 
    json={"mode": "autoregressive"}
)
```

## 性能特征

### Autoregressive模式
- **高吞吐量**：支持更多并发请求（4097）
- **稳定延迟**：每个token的生成时间相对固定
- **内存效率**：单个请求内存使用较低

### Speculative模式  
- **低延迟**：通过投机执行减少生成时间
- **有限并发**：支持较少并发请求（48）
- **内存密集**：需要额外的draft model内存

## 最佳实践

### 1. 模式选择策略

```python
def choose_mode(current_load, avg_request_length):
    """根据负载和请求长度选择最优模式"""
    if current_load > 100:
        return "autoregressive"  # 高负载时优先吞吐量
    elif avg_request_length > 1000:
        return "speculative"     # 长文本时优先延迟
    else:
        return "autoregressive"  # 默认选择
```

### 2. 监控和切换

```python
class AdaptiveModeController:
    def __init__(self):
        self.metrics_window = []
        
    def should_switch_mode(self, current_metrics):
        """基于性能指标决定是否切换模式"""
        # 收集最近的性能数据
        self.metrics_window.append(current_metrics)
        if len(self.metrics_window) > 10:
            self.metrics_window.pop(0)
            
        # 分析趋势并决定切换
        avg_latency = sum(m['latency'] for m in self.metrics_window) / len(self.metrics_window)
        current_load = current_metrics['active_requests']
        
        if avg_latency > 2.0 and current_load < 50:
            return "speculative"
        elif current_load > 200:
            return "autoregressive"
        return None
```

### 3. 错误处理

```python
def safe_mode_switch(mode):
    """安全的模式切换，包含重试和回滚"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:30000/switch_inference_mode",
                json={"mode": mode},
                timeout=10
            )
            
            if response.status_code == 200 and response.json()["success"]:
                return True
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    
    return False
```

## 限制和注意事项

### 1. 内存开销
- 系统始终按最大需求分配内存
- 在投机模式下可能存在内存浪费

### 2. 切换延迟
- 模式切换需要等待当前批次完成
- 频繁切换可能影响性能

### 3. 配置约束
- 必须在启动时启用动态切换功能
- 某些高级配置可能不支持动态调整

## 故障排除

### 常见问题

1. **切换失败**
   ```
   Error: Dynamic mode switching is not enabled
   ```
   解决：启动时添加`--enable-dynamic-mode-switching`参数

2. **内存不足**
   ```
   Error: alloc_req_slots runs out of memory
   ```
   解决：减少`--max-running-requests`或增加GPU内存

3. **性能下降**
   ```
   Warning: Frequent mode switching detected
   ```
   解决：增加切换间隔或优化切换策略

### 调试工具

```bash
# 查看当前状态
curl http://localhost:30000/get_server_info | jq '.internal_states'

# 监控内存使用
curl http://localhost:30000/get_server_info | jq '.max_total_num_tokens'
```

## 示例代码

完整的使用示例请参考：`examples/dynamic_mode_switching.py`

该示例展示了：
- 基本的模式切换操作
- 性能基准测试
- 自适应模式选择
- 错误处理和重试机制
