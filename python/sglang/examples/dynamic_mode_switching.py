#!/usr/bin/env python3
"""
动态推理模式切换示例

这个示例展示了如何在投机推理和自回归推理之间动态切换，
同时处理不同模式下的内存使用和请求上限差异。
"""

import requests
import time
import json
from typing import Dict, Any


class SGLangModeController:
    """SGLang推理模式控制器"""
    
    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url
        self.current_mode = None
        
    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        response = requests.get(f"{self.base_url}/get_server_info")
        response.raise_for_status()
        return response.json()
    
    def switch_mode(self, mode: str) -> bool:
        """切换推理模式
        
        Args:
            mode: 'speculative' 或 'autoregressive'
            
        Returns:
            bool: 切换是否成功
        """
        if mode not in ["speculative", "autoregressive"]:
            raise ValueError("Mode must be 'speculative' or 'autoregressive'")
            
        print(f"Switching to {mode} mode...")
        
        response = requests.post(
            f"{self.base_url}/switch_inference_mode",
            json={"mode": mode}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                self.current_mode = mode
                print(f"✅ Successfully switched to {mode} mode")
                return True
            else:
                print(f"❌ Failed to switch: {result.get('message')}")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return False
    
    def generate_text(self, prompt: str, max_new_tokens: int = 50) -> str:
        """生成文本"""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.7
                }
            }
        )
        response.raise_for_status()
        return response.json()["text"]
    
    def benchmark_mode(self, mode: str, prompts: list, iterations: int = 3) -> Dict[str, float]:
        """基准测试特定模式的性能"""
        if not self.switch_mode(mode):
            return {"error": "Failed to switch mode"}
            
        total_time = 0
        total_tokens = 0
        
        for i in range(iterations):
            for prompt in prompts:
                start_time = time.time()
                result = self.generate_text(prompt)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                # 简单估算token数量
                total_tokens += len(result.split())
                
        avg_time_per_request = total_time / (len(prompts) * iterations)
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        return {
            "mode": mode,
            "avg_time_per_request": avg_time_per_request,
            "tokens_per_second": tokens_per_second,
            "total_requests": len(prompts) * iterations
        }


def main():
    """主函数：演示动态模式切换"""
    
    # 初始化控制器
    controller = SGLangModeController()
    
    # 测试提示词
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "How does photosynthesis work?",
        "What are the benefits of renewable energy?"
    ]
    
    print("🚀 SGLang Dynamic Mode Switching Demo")
    print("=" * 50)
    
    # 获取服务器信息
    try:
        server_info = controller.get_server_info()
        print(f"📊 Server Info:")
        print(f"   Model: {server_info.get('model_path', 'Unknown')}")
        print(f"   Version: {server_info.get('version', 'Unknown')}")
        print()
    except Exception as e:
        print(f"❌ Failed to get server info: {e}")
        return
    
    # 测试自回归模式
    print("🔄 Testing Autoregressive Mode")
    print("-" * 30)
    autoregressive_results = controller.benchmark_mode("autoregressive", test_prompts)
    if "error" not in autoregressive_results:
        print(f"   Average time per request: {autoregressive_results['avg_time_per_request']:.3f}s")
        print(f"   Tokens per second: {autoregressive_results['tokens_per_second']:.2f}")
        print(f"   Total requests: {autoregressive_results['total_requests']}")
    else:
        print(f"   Error: {autoregressive_results['error']}")
    print()
    
    # 测试投机推理模式
    print("🔄 Testing Speculative Mode")
    print("-" * 30)
    speculative_results = controller.benchmark_mode("speculative", test_prompts)
    if "error" not in speculative_results:
        print(f"   Average time per request: {speculative_results['avg_time_per_request']:.3f}s")
        print(f"   Tokens per second: {speculative_results['tokens_per_second']:.2f}")
        print(f"   Total requests: {speculative_results['total_requests']}")
    else:
        print(f"   Error: {speculative_results['error']}")
    print()
    
    # 性能对比
    if "error" not in autoregressive_results and "error" not in speculative_results:
        print("📈 Performance Comparison")
        print("-" * 30)
        
        auto_tps = autoregressive_results['tokens_per_second']
        spec_tps = speculative_results['tokens_per_second']
        
        if spec_tps > auto_tps:
            improvement = ((spec_tps - auto_tps) / auto_tps) * 100
            print(f"   Speculative mode is {improvement:.1f}% faster")
        else:
            degradation = ((auto_tps - spec_tps) / auto_tps) * 100
            print(f"   Autoregressive mode is {degradation:.1f}% faster")
        print()
    
    # 演示实时切换
    print("🔀 Real-time Mode Switching Demo")
    print("-" * 30)
    
    modes = ["autoregressive", "speculative", "autoregressive"]
    prompt = "Tell me about artificial intelligence."
    
    for mode in modes:
        print(f"Switching to {mode} mode...")
        if controller.switch_mode(mode):
            start_time = time.time()
            result = controller.generate_text(prompt, max_new_tokens=30)
            end_time = time.time()
            
            print(f"   Generated in {end_time - start_time:.3f}s")
            print(f"   Result: {result[:100]}...")
        else:
            print(f"   Failed to switch to {mode} mode")
        print()
    
    print("✅ Demo completed!")


if __name__ == "__main__":
    main()
