#!/usr/bin/env python3
"""
SignLLM推理脚本示例
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from signllm_model import SignLLM, ModelConfig, CONFIG


def load_model_for_inference(checkpoint_path: str, model_size: str = "tiny", languages=["ASL"]):
    """
    加载训练好的模型用于推理
    
    Args:
        checkpoint_path: 检查点文件路径
        model_size: 模型大小配置
        languages: 支持的语言列表
    Returns:
        model: 加载好的模型
    """
    # 1. 重建配置
    global CONFIG
    CONFIG.__init__(model_size)
    
    # 2. 重建模型结构
    model = SignLLM(languages=languages)
    
    # 3. 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 4. 设置为推理模式
    model.eval()
    
    print(f"✅ 模型已加载：{checkpoint_path}")
    print(f"📊 训练轮次：{checkpoint.get('epoch', 'Unknown')}")
    print(f"📉 训练损失：{checkpoint.get('loss', 'Unknown'):.6f}")
    
    return model


def inference_demo():
    """推理演示"""
    print("🚀 SignLLM推理演示")
    print("=" * 50)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
    # 加载模型
    checkpoint_path = "checkpoints/eggroll_train/epoch_10.pth"  # 修改为实际路径
    model = load_model_for_inference(
        checkpoint_path=checkpoint_path,
        model_size="tiny",           # 必须与训练时一致
        languages=["ASL"]            # 必须与训练时一致
    )
    model.to(device)
    
    # 推理测试
    print("\n🔍 开始推理...")
    with torch.no_grad():
        test_texts = [
            "Hello world",
            "How are you?",
            "Nice to meet you"
        ]
        
        for text in test_texts:
            print(f"\n📝 输入文本: '{text}'")
            
            # 生成手语姿态
            poses, quality_scores = model(
                texts=[text],
                language="ASL",
                mode="mlsf",
                max_length=100  # 可自定义帧数
            )
            
            print(f"📊 生成姿态形状: {poses.shape}")
            print(f"📏 生成帧数: {poses.shape[1]}")
            print(f"🎯 平均质量分数: {quality_scores.mean().item():.4f}")


def save_minimal_checkpoint(full_checkpoint_path: str, output_path: str):
    """
    保存最小化的推理检查点（只保留必要的模型权重）
    
    Args:
        full_checkpoint_path: 完整训练检查点路径
        output_path: 输出的最小检查点路径
    """
    # 加载完整检查点
    full_checkpoint = torch.load(full_checkpoint_path, map_location='cpu')
    
    # 创建最小检查点（只保留推理必需的信息）
    minimal_checkpoint = {
        'model_state_dict': full_checkpoint['model_state_dict'],
        'model_config': {
            'model_size': 'tiny',  # 需要手动指定或从训练记录中获取
            'languages': ['ASL']
        },
        'training_info': {
            'epoch': full_checkpoint.get('epoch'),
            'final_loss': full_checkpoint.get('loss')
        }
    }
    
    # 保存最小检查点
    torch.save(minimal_checkpoint, output_path)
    
    # 计算大小差异
    import os
    full_size = os.path.getsize(full_checkpoint_path) / 1024 / 1024  # MB
    minimal_size = os.path.getsize(output_path) / 1024 / 1024  # MB
    
    print(f"📦 完整检查点: {full_size:.2f} MB")
    print(f"📦 最小检查点: {minimal_size:.2f} MB")
    print(f"💾 节省空间: {full_size - minimal_size:.2f} MB ({(full_size-minimal_size)/full_size*100:.1f}%)")


if __name__ == "__main__":
    inference_demo()
    
    # 可选：创建最小化检查点
    # save_minimal_checkpoint(
    #     "checkpoints/eggroll_train/epoch_10.pth",
    #     "checkpoints/eggroll_train/minimal_epoch_10.pth"
    # ) 