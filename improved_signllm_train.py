#!/usr/bin/env python3
"""
改进的 SignLLM 训练脚本
- 添加验证集评估
- 复合损失函数
- 学习率调度
- 早停机制
- 更详细的日志记录
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from improved_signllm_model import ImprovedSignLLM, ModelConfig
from data_processor import MultilingualSignDataset, collate_fn
from improved_evaluation import ImprovedSignLLMEvaluator
from pose_consistency_loss import PoseConsistencyLoss


class PoseLoss(nn.Module):
    """复合姿态损失函数 - 集成了骨架结构约束"""
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3):
        super().__init__()
        self.alpha = alpha  # MSE权重
        self.beta = beta    # 运动平滑度权重  
        self.gamma = gamma  # 姿态一致性权重
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
        # 集成专门的姿态一致性损失
        self.pose_consistency = PoseConsistencyLoss(
            bone_length_weight=0.2,
            joint_angle_weight=0.1,
            symmetry_weight=0.05,
            temporal_weight=0.1
        )
        
    def forward(self, pred_poses: torch.Tensor, target_poses: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 基础重建损失
        mse_loss = self.mse(pred_poses, target_poses)
        l1_loss = self.l1(pred_poses, target_poses)
        
        # 运动平滑度损失（相邻帧差异）
        pred_diff = pred_poses[:, 1:] - pred_poses[:, :-1]
        target_diff = target_poses[:, 1:] - target_poses[:, :-1]
        motion_loss = self.mse(pred_diff, target_diff)
        
        # 姿态一致性损失（使用完整的骨架约束）
        consistency_losses = self.pose_consistency(pred_poses, target_poses)
        consistency_loss = consistency_losses['total']
        
        total_loss = (self.alpha * mse_loss + 
                     self.beta * motion_loss + 
                     self.gamma * consistency_loss)
        
        # 返回详细的损失信息
        result = {
            'total': total_loss,
            'mse': mse_loss,
            'motion': motion_loss,
            'consistency': consistency_loss,
            'l1': l1_loss
        }
        
        # 添加详细的一致性损失信息
        for key, value in consistency_losses.items():
            if key != 'total':
                result[f'consistency_{key}'] = value
        
        return result


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


def validate_model(model: ImprovedSignLLM, val_loader: DataLoader, 
                  criterion: PoseLoss, evaluator: ImprovedSignLLMEvaluator,
                  device: torch.device) -> Dict[str, float]:
    """验证模型性能"""
    model.eval()
    total_losses = {
        'total': 0, 'mse': 0, 'motion': 0, 'consistency': 0,
        'consistency_bone_length': 0, 'consistency_joint_angle': 0,
        'consistency_symmetry': 0, 'consistency_temporal': 0,
        'consistency_supervised': 0
    }
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            texts = batch["texts"]
            target_poses = batch["pose_sequences"].to(device)
            
            # Teacher Forcing模式验证
            results = model(
                texts=texts,
                language="ASL",
                target_poses=target_poses
            )
            pred_poses = results['predicted_poses']
            
            losses = criterion(pred_poses, target_poses)
            
            # 累积所有损失项
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            
            # 收集预测和目标用于详细评估
            all_predictions.extend(pred_poses.cpu().numpy())
            all_targets.extend(target_poses.cpu().numpy())
    
    # 计算平均损失
    avg_losses = {k: v / len(val_loader) for k, v in total_losses.items()}
    
    # 详细评估指标
    eval_metrics = evaluator.evaluate_poses(all_predictions, all_targets)
    
    return {**avg_losses, **eval_metrics}


def main():
    print("🚀 改进的 SignLLM 训练")
    print("=" * 60)
    
    # 配置
    MODEL_SIZE = "large"
    BATCH_SIZE = 4  # 调回到4，8可能导致内存问题
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5
    PATIENCE = 10
    
    # 设备和目录
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path("checkpoints/improved_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard日志
    writer = SummaryWriter(log_dir=output_dir / "logs")
    
    # 模型配置
    config = ModelConfig(MODEL_SIZE)
    
    # 创建模型
    print("📦 创建模型...")
    model = ImprovedSignLLM(languages=["ASL"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 参数量: {total_params:,} ({total_params/1_000_000:.1f}M)")
    
    # 数据集
    print("📚 加载数据集...")
    train_dataset = MultilingualSignDataset(
        data_dirs={"ASL": "datasets/signllm_data_complete"},
        languages=["ASL"],
        split="dev",
        max_sequence_length=config.default_max_frames,
        pose_dim=config.pose_dim,
    )
    
    val_dataset = MultilingualSignDataset(
        data_dirs={"ASL": "datasets/signllm_data_complete"},
        languages=["ASL"],
        split="test",  # 暂时使用dev作为验证集（因为test不存在）
        max_sequence_length=config.default_max_frames,
        pose_dim=config.pose_dim,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )
    
    print(f"📊 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 优化器和损失
    criterion = PoseLoss(alpha=1.0, beta=0.5, gamma=1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    scaler = torch.amp.GradScaler(device.type)
    
    # 评估器和早停
    evaluator = ImprovedSignLLMEvaluator()
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    # 训练循环
    best_val_loss = float('inf')
    print("\n🎯 开始训练...")
    
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        epoch_losses = {
            'total': 0, 'mse': 0, 'motion': 0, 'consistency': 0,
            'consistency_bone_length': 0, 'consistency_joint_angle': 0,
            'consistency_symmetry': 0, 'consistency_temporal': 0
        }
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch_idx, batch in enumerate(train_pbar):
            texts = batch["texts"]
            target_poses = batch["pose_sequences"].to(device)
            
            with torch.amp.autocast(device.type):
                results = model(
                    texts=texts,
                    language="ASL",
                    target_poses=target_poses
                )
                pred_poses = results['predicted_poses']
                losses = criterion(pred_poses, target_poses)
            
            optimizer.zero_grad()
            scaler.scale(losses['total']).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # 记录损失
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # 更新进度条
            current_loss = losses['total'].item()
            train_pbar.set_postfix({'loss': f'{current_loss:.4f}'})
            
            # 记录到TensorBoard
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', current_loss, step)
        
        # 计算平均训练损失
        avg_train_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        
        # 验证阶段
        val_metrics = validate_model(model, val_loader, criterion, evaluator, device)
        val_loss = val_metrics['total']
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录到TensorBoard
        for key, value in avg_train_losses.items():
            writer.add_scalar(f'Train/{key.capitalize()}Loss', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'Val/{key.capitalize()}', value, epoch)
        
        # 打印结果
        print(f"\n📊 Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  训练损失: {avg_train_losses['total']:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"  DTW分数: {val_metrics.get('dtw_score', 0):.4f}")
        print(f"  姿态相似度: {val_metrics.get('pose_similarity', 0):.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config.__dict__,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print("💾 保存最佳模型")
        
        # 早停检查
        if early_stopping(val_loss):
            print(f"🛑 早停触发 (epoch {epoch+1})")
            break
    
    writer.close()
    print("\n✅ 训练完成！")
    
    # 最终测试
    print("\n🔍 最终推理测试...")
    model.eval()
    model.set_inference_mode(True)
    with torch.no_grad():
        test_texts = ["Hello world", "How are you?"]
        results = model(texts=test_texts, language="ASL")
        test_poses = results['predicted_poses']
        print(f"✅ 推理成功，输出形状: {test_poses.shape}")


if __name__ == "__main__":
    main() 