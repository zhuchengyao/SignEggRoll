#!/usr/bin/env python3
"""
SignLLM 快速训练脚本 - 用于验证训练流程
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from signllm_model import SignLLM
from data_processor import MultilingualSignDataset


def main():
    print("🚀 SignLLM 快速训练验证")
    print("=" * 50)
    
    # 检查环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 检查数据
    data_path = Path("datasets/signllm_data_complete")
    if not data_path.exists():
        print(f"❌ 数据路径不存在: {data_path}")
        print("请先运行数据转换脚本")
        return
    
    asl_dev_path = data_path / "ASL" / "dev"
    if not asl_dev_path.exists():
        print(f"❌ ASL dev数据不存在: {asl_dev_path}")
        return
    
    sample_count = len([d for d in asl_dev_path.iterdir() if d.is_dir()])
    print(f"✅ 找到 {sample_count} 个ASL样本")
    
    # 创建模型
    print("\n📦 创建模型...")
    model = SignLLM(
        languages=["ASL"],
        gloss_vocab_size=1000,
        hidden_dim=256,
        pose_dim=150
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型参数: {total_params:,}")
    
    # 创建数据集
    print("\n📚 创建数据集...")
    try:
        dataset = MultilingualSignDataset(
            data_dirs={"ASL": str(data_path)},
            languages=["ASL"],
            split="dev",
            max_sequence_length=200,
            pose_dim=150
        )
        print(f"✅ 数据集创建成功: {len(dataset)} 样本")
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return
    
    # 测试数据加载
    print("\n🔍 测试数据加载...")
    try:
        sample = dataset[0]
        print(f"✅ 样本加载成功:")
        print(f"   文本: {sample['text'][:50]}...")
        print(f"   姿态形状: {sample['pose_sequence'].shape}")
        print(f"   语言: {sample['language']}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 快速训练
    print("\n🎯 开始快速训练（3个epoch）...")
    model.train()
    
    # 创建输出目录
    checkpoint_dir = Path("checkpoints/quick_train")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(3):
        print(f"\n📅 Epoch {epoch+1}/3")
        epoch_loss = 0
        num_batches = 0
        
        # 使用前20个样本进行快速训练
        max_samples = min(20, len(dataset))
        batch_size = 2
        
        for i in tqdm(range(0, max_samples, batch_size), desc=f"Epoch {epoch+1}"):
            try:
                # 手动创建批次
                batch_texts = []
                batch_poses = []
                
                for j in range(batch_size):
                    if i + j < max_samples:
                        sample = dataset[i + j]
                        batch_texts.append(sample['text'])
                        batch_poses.append(sample['pose_sequence'])
                
                if len(batch_texts) == 0:
                    continue
                
                # 填充到相同长度
                max_len = max(pose.size(0) for pose in batch_poses)
                padded_poses = []
                
                for pose in batch_poses:
                    if pose.size(0) < max_len:
                        padding = torch.zeros(max_len - pose.size(0), pose.size(1))
                        padded_pose = torch.cat([pose, padding], dim=0)
                    else:
                        padded_pose = pose[:max_len]
                    padded_poses.append(padded_pose)
                
                target_poses = torch.stack(padded_poses).to(device)
                
                # 前向传播
                pred_poses, quality_scores = model(
                    texts=batch_texts,
                    language="ASL",
                    mode="mlsf",
                    max_length=target_poses.size(1)
                )
                
                # 计算损失
                loss = criterion(pred_poses, target_poses)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # 统计
                epoch_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"❌ 批次 {i//batch_size} 失败: {e}")
                continue
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"📊 Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        
        checkpoint_path = checkpoint_dir / f"epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 检查点已保存: {checkpoint_path}")
    
    print("\n✅ 快速训练完成！")
    
    # 推理测试
    print("\n🔍 推理测试...")
    model.eval()
    with torch.no_grad():
        try:
            test_texts = ["Hello world", "How are you"]
            test_poses, test_quality = model(
                texts=test_texts,
                language="ASL",
                mode="mlsf",
                max_length=50
            )
            print(f"✅ 推理成功！")
            print(f"   输入文本: {test_texts}")
            print(f"   生成姿态形状: {test_poses.shape}")
            print(f"   质量分数: {test_quality.mean().item():.4f}")
        except Exception as e:
            print(f"❌ 推理失败: {e}")
    
    print("\n🎉 快速训练验证完成！")
    print("如果所有步骤都成功，您可以开始完整训练：")
    print("python start_training.py --config configs/signllm_your_data_config.json")


if __name__ == "__main__":
    main() 