#!/usr/bin/env python3
"""
最小化SignLLM训练脚本 - 解决数据格式问题
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from signllm_model import SignLLM, ModelConfig, CONFIG
from data_processor import MultilingualSignDataset


def main():
    print("🚀 最小化SignLLM训练")
    print("=" * 50)
    
    # 设置模型大小 - 只需要改这里！
    # 可选: "tiny", "small", "medium", "large"
    MODEL_SIZE = "tiny"  # 使用最小的模型
    
    # 更新全局配置
    global CONFIG
    CONFIG.__init__(MODEL_SIZE)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
    # 创建简化模型
    print("📦 创建模型...")
    model = SignLLM(languages=["ASL"]).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 实际模型参数: {total_params:,} ({total_params/1_000_000:.1f}M)")
    
    # 创建数据集
    print("📚 创建数据集...")
    dataset = MultilingualSignDataset(
        data_dirs={"ASL": "datasets/signllm_data_complete"},
        languages=["ASL"],
        split="dev",
        max_sequence_length=256,
        pose_dim=CONFIG.pose_dim
    )
    
    print(f"📊 数据集大小: {len(dataset)} 样本")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 训练循环
    print("\n🎯 开始训练...")
    model.train()
    epoch_num = 10
    
    for epoch in range(epoch_num):
        print(f"\n📅 Epoch {epoch+1}/{epoch_num}")
        epoch_loss = 0
        num_batches = 0
        
        # 手动批处理
        batch_size = 2
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Epoch {epoch+1}"):
            try:
                # 手动创建批次
                batch_texts = []
                batch_poses = []
                
                for j in range(batch_size):
                    if i + j < len(dataset):
                        sample = dataset[i + j]
                        batch_texts.append(sample['text'])
                        batch_poses.append(sample['pose_sequence'])
                
                if len(batch_texts) == 0:
                    continue
                
                # 转换为张量
                target_poses = torch.stack(batch_poses).to(device)
                
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
        if (epoch + 1) % 1 == 0:
            checkpoint_dir = Path("checkpoints/eggroll_train")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            
            checkpoint_path = checkpoint_dir / f"epoch_{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 检查点已保存: {checkpoint_path}")
    
    print("\n✅ 训练完成！")
    
    # 简单推理测试
    print("\n🔍 推理测试...")
    model.eval()
    with torch.no_grad():
        test_texts = ["Hello world"]
        test_poses, test_quality = model(
            texts=test_texts,
            language="ASL",
            mode="mlsf"
        )
        print(f"✅ 推理成功！生成姿态形状: {test_poses.shape}")
        print(f"📏 生成帧数: {test_poses.shape[1]}")


if __name__ == "__main__":
    main() 