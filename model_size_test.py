#!/usr/bin/env python3
"""
模型大小测试脚本 - 比较不同配置的参数量
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from signllm_model import SignLLM, ModelConfig, CONFIG
import torch

def test_model_sizes():
    """测试不同模型大小的参数量"""
    sizes = ["tiny", "small", "medium", "large"]
    
    print("🔍 SignLLM模型大小比较")
    print("=" * 60)
    
    for size in sizes:
        print(f"\n📊 {size.upper()} 模型:")
        
        # 更新配置
        global CONFIG
        CONFIG.__init__(size)
        
        # 创建模型
        model = SignLLM(languages=["ASL"])
        
        # 计算实际参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  实际参数量: {total_params:,} ({total_params/1_000_000:.1f}M)")
        print(f"  可训练参数: {trainable_params:,} ({trainable_params/1_000_000:.1f}M)")
        
        # 估算显存使用 (粗略)
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        print(f"  模型大小: {model_size_mb:.1f} MB")
        
        del model  # 释放内存
        
    print("\n💡 建议:")
    print("  - tiny: 快速原型开发和调试")
    print("  - small: 平衡性能和速度")
    print("  - medium: 更好的性能")
    print("  - large: 最佳性能（需要更多资源）")

if __name__ == "__main__":
    test_model_sizes() 