#!/usr/bin/env python3
"""
集成配置管理的训练脚本 - 支持配置文件和命令行参数
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

from signllm_model import SignLLM
from data_processor import MultilingualSignDataset, collate_fn
from evaluation import SignLLMEvaluator
from config_manager import ConfigManager
from improved_signllm_train import PoseLoss, EarlyStopping, validate_model


def main():
    print("🚀 SignLLM 配置化训练")
    print("=" * 60)
    
    # 解析配置
    config_manager = ConfigManager()
    config = config_manager.parse_args_and_config()
    
    print("🔧 训练配置:")
    print(f"  模型大小: {config.model_size}")
    print(f"  隐藏维度: {config.hidden_dim}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  训练轮数: {config.num_epochs}")
    print(f"  设备: {config.device_auto}")
    
    # 内存估算
    memory_info = config.estimate_memory_usage()
    print(f"  估算内存: {memory_info['total_estimated_mb']:.1f} MB")
    
    # 设备
    device = torch.device(config.device_auto)
    
    # 创建输出目录
    output_dir = Path("checkpoints") / f"config_training_{config.model_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存使用的配置
    config_manager.save_to_file(config, output_dir / "training_config.yaml")
    
    # TensorBoard日志
    writer = SummaryWriter(log_dir=output_dir / "logs")
    
    # 创建模型
    print("📦 创建模型...")
    model = SignLLM(languages=["ASL"]).to(device)
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
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    print(f"📊 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 优化器和损失
    criterion = PoseLoss(
        alpha=config.loss_alpha, 
        beta=config.loss_beta, 
        gamma=config.loss_gamma
    )
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 混合精度训练
    scaler = torch.amp.GradScaler(device.type) if config.mixed_precision else None
    
    # 评估器和早停
    evaluator = SignLLMEvaluator()
    early_stopping = EarlyStopping(patience=config.patience)
    
    # 训练循环
    best_val_loss = float('inf')
    print("\n🎯 开始训练...")
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        epoch_losses = {'total': 0, 'mse': 0, 'motion': 0, 'consistency': 0}
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_idx, batch in enumerate(train_pbar):
            texts = batch["texts"]
            target_poses = batch["pose_sequences"].to(device)
            
            # 前向传播
            if config.mixed_precision and scaler:
                with torch.amp.autocast(device.type):
                    pred_poses, _ = model(
                        texts=texts,
                        language="ASL", 
                        mode="mlsf",
                        target_poses=target_poses
                    )
                    losses = criterion(pred_poses, target_poses)
                
                optimizer.zero_grad()
                scaler.scale(losses['total']).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_poses, _ = model(
                    texts=texts,
                    language="ASL", 
                    mode="mlsf",
                    target_poses=target_poses
                )
                losses = criterion(pred_poses, target_poses)
                
                optimizer.zero_grad()
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # 记录损失
            for key in epoch_losses:
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
        print(f"\n📊 Epoch {epoch+1}/{config.num_epochs}:")
        print(f"  训练损失: {avg_train_losses['total']:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"  DTW分数: {val_metrics.get('dtw_score', 0):.4f}")
        print(f"  姿态相似度: {val_metrics.get('pose_similarity', 0):.4f}")
        print(f"  当前学习率: {scheduler.optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config.to_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print("💾 保存最佳模型")
        
        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config.to_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # 早停检查
        if early_stopping(val_loss):
            print(f"🛑 早停触发 (epoch {epoch+1})")
            break
    
    writer.close()
    print("\n✅ 训练完成！")
    print(f"📁 模型保存在: {output_dir}")
    
    # 最终测试
    print("\n🔍 最终推理测试...")
    model.eval()
    with torch.no_grad():
        test_texts = ["Hello world", "How are you?"]
        test_poses, _ = model(texts=test_texts, language="ASL", mode="mlsf")
        print(f"✅ 推理成功，输出形状: {test_poses.shape}")


if __name__ == "__main__":
    main() 