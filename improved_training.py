#!/usr/bin/env python3
"""
改进的SignLLM训练脚本
包含多种优化技术以突破训练瓶颈
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import numpy as np
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from signllm_model import SignLLM, ModelConfig, CONFIG, AdvancedRLLoss
from data_processor import SignLanguageDataset  # 假设你有这个数据加载器


class ImprovedTrainer:
    """改进的训练器"""
    
    def __init__(self, model_size: str = "medium", use_advanced_loss: bool = True):
        """
        Args:
            model_size: 模型大小 ("small", "medium", "large", "xl")
            use_advanced_loss: 是否使用改进的损失函数
        """
        # 更新配置
        global CONFIG
        CONFIG.__init__(model_size)
        CONFIG.print_config()
        
        # 初始化模型
        self.model = SignLLM(languages=["ASL"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 损失函数
        if use_advanced_loss:
            self.criterion = AdvancedRLLoss(
                alpha=0.15,   # 质量权重
                beta=0.1,     # 多样性权重  
                gamma=0.08,   # 平滑性权重
                delta=0.03    # 一致性权重
            )
            print("✅ 使用改进的AdvancedRLLoss损失函数")
        else:
            self.criterion = nn.MSELoss()
            print("✅ 使用标准MSE损失函数")
        
        # 设置优化器 - 使用AdamW + 分层学习率
        self.setup_optimizer()
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # 设置日志
        self.setup_logging()
        
    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        # 分层学习率：BERT用较小学习率，新参数用较大学习率
        bert_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder.encoder' in name:  # BERT参数
                bert_params.append(param)
            else:  # 其他参数
                other_params.append(param)
        
        # 基础学习率
        base_lr = self.get_optimal_lr()
        
        # 参数组
        param_groups = [
            {'params': bert_params, 'lr': base_lr * 0.1, 'weight_decay': CONFIG.weight_decay * 0.5},  # BERT用小学习率
            {'params': other_params, 'lr': base_lr, 'weight_decay': CONFIG.weight_decay}  # 其他参数
        ]
        
        # AdamW优化器
        self.optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True  # 提高稳定性
        )
        
        print(f"✅ 优化器设置完成")
        print(f"   BERT学习率: {base_lr * 0.1:.2e}")
        print(f"   其他参数学习率: {base_lr:.2e}")
        print(f"   权重衰减: {CONFIG.weight_decay}")
        
    def get_optimal_lr(self) -> float:
        """根据模型大小获取最优学习率"""
        lr_map = {
            "tiny": 3e-4,
            "small": 2e-4, 
            "medium": 1e-4,
            "large": 8e-5,
            "xl": 5e-5
        }
        return lr_map.get(CONFIG.model_size, 1e-4)
    
    def setup_scheduler(self, total_steps: int):
        """设置学习率调度器"""
        # 使用OneCycleLR进行学习率调度
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[group['lr'] for group in self.optimizer.param_groups],
            total_steps=total_steps,
            pct_start=0.1,  # 10%的步数用于预热
            anneal_strategy='cos',
            div_factor=25.0,  # 初始学习率 = max_lr / div_factor
            final_div_factor=1000.0  # 最终学习率 = max_lr / final_div_factor
        )
        
        print(f"✅ 学习率调度器设置完成")
        print(f"   总步数: {total_steps}")
        print(f"   预热比例: 10%")
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('improved_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def compute_loss(self, pred_poses: torch.Tensor, target_poses: torch.Tensor, 
                    quality_scores: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """计算损失"""
        if isinstance(self.criterion, AdvancedRLLoss):
            return self.criterion(pred_poses, target_poses, quality_scores, mask)
        else:
            return self.criterion(pred_poses, target_poses)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 解包数据
            texts = batch['text']
            target_poses = batch['poses'].to(self.device)  # [batch_size, seq_len, 150]
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            try:
                # 模型预测
                pred_poses, quality_scores = self.model(
                    texts=texts, 
                    language="ASL", 
                    mode="mlsf",
                    max_length=target_poses.size(1)
                )
                
                # 计算损失
                loss = self.compute_loss(pred_poses, target_poses, quality_scores, mask)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG.gradient_clip)
                
                # 优化器步进
                self.optimizer.step()
                
                # 学习率调度
                if hasattr(self, 'scheduler'):
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.learning_rates.append(current_lr)
                
                # 累积损失
                total_loss += loss.item()
                
                # 更新进度条
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{avg_loss:.6f}',
                    'lr': f'{current_lr:.2e}' if hasattr(self, 'scheduler') else 'N/A'
                })
                
            except RuntimeError as e:
                self.logger.error(f"训练错误: {e}")
                continue
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="验证中"):
                texts = batch['text']
                target_poses = batch['poses'].to(self.device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(self.device)
                
                try:
                    # 模型预测
                    pred_poses, quality_scores = self.model(
                        texts=texts,
                        language="ASL",
                        mode="mlsf", 
                        max_length=target_poses.size(1)
                    )
                    
                    # 计算损失
                    loss = self.compute_loss(pred_poses, target_poses, quality_scores, mask)
                    total_loss += loss.item()
                    
                except RuntimeError as e:
                    self.logger.error(f"验证错误: {e}")
                    continue
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': CONFIG.__dict__
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = str(Path(filepath).parent / "best_model.pth")
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 保存最佳模型: {best_path}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        self.logger.info(f"✅ 加载检查点: epoch {self.current_epoch}, best_loss: {self.best_loss:.6f}")
    
    def plot_training_curves(self, save_path: str = "training_curves.png"):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        if self.train_losses:
            axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        if self.val_losses:
            axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 学习率曲线
        if self.learning_rates:
            axes[0, 1].plot(self.learning_rates, color='green')
            axes[0, 1].set_title('学习率变化')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True)
        
        # 损失对比 (最近的epochs)
        if len(self.train_losses) > 10:
            recent_train = self.train_losses[-10:]
            recent_val = self.val_losses[-10:] if len(self.val_losses) >= 10 else self.val_losses
            axes[1, 0].plot(recent_train, 'o-', label='Recent Train', color='blue')
            if recent_val:
                axes[1, 0].plot(recent_val, 'o-', label='Recent Val', color='red')
            axes[1, 0].set_title('最近10轮损失')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 训练统计
        if self.train_losses:
            min_loss = min(self.train_losses)
            best_epoch = self.train_losses.index(min_loss)
            improvement = self.train_losses[0] - min_loss if len(self.train_losses) > 1 else 0
            
            stats_text = f"""训练统计:
最佳训练损失: {min_loss:.6f}
最佳轮次: {best_epoch}
总改进: {improvement:.6f}
当前轮次: {self.current_epoch}
模型大小: {CONFIG.model_size}"""
            
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 1].set_title('训练统计')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 训练曲线已保存: {save_path}")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None,
              num_epochs: int = 50, save_dir: str = "improved_checkpoints"):
        """主训练循环"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # 计算总步数并设置调度器
        total_steps = len(train_dataloader) * num_epochs
        self.setup_scheduler(total_steps)
        
        self.logger.info(f"🚀 开始改进训练")
        self.logger.info(f"   模型大小: {CONFIG.model_size}")
        self.logger.info(f"   总轮次: {num_epochs}")
        self.logger.info(f"   总步数: {total_steps}")
        self.logger.info(f"   保存目录: {save_dir}")
        
        patience = 10  # 早停耐心
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch(train_dataloader)
            
            # 验证
            val_loss = None
            if val_dataloader:
                val_loss = self.validate(val_dataloader)
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            else:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")
            
            # 检查是否为最佳模型
            current_loss = val_loss if val_loss is not None else train_loss
            is_best = current_loss < self.best_loss
            
            if is_best:
                self.best_loss = current_loss
                patience_counter = 0
                self.logger.info(f"🎉 新的最佳损失: {self.best_loss:.6f}")
            else:
                patience_counter += 1
            
            # 保存检查点
            checkpoint_path = save_dir / f"epoch_{epoch}.pth"
            self.save_checkpoint(str(checkpoint_path), is_best)
            
            # 绘制训练曲线
            if epoch % 5 == 0:
                self.plot_training_curves(str(save_dir / f"training_curves_epoch_{epoch}.png"))
            
            # 早停检查
            if patience_counter >= patience:
                self.logger.info(f"⏹️  早停触发，连续{patience}轮无改进")
                break
        
        # 最终保存
        final_path = save_dir / "final_model.pth"
        self.save_checkpoint(str(final_path))
        self.plot_training_curves(str(save_dir / "final_training_curves.png"))
        
        self.logger.info(f"✅ 训练完成！最佳损失: {self.best_loss:.6f}")


def main():
    """主函数 - 使用改进的训练器"""
    
    # 训练配置
    BATCH_SIZE = 8  # 根据显存调整
    NUM_EPOCHS = 100
    MODEL_SIZE = "medium"  # 尝试更大的模型
    
    print("🔧 设置改进训练")
    
    # 创建训练器
    trainer = ImprovedTrainer(
        model_size=MODEL_SIZE,
        use_advanced_loss=True
    )
    
    # TODO: 加载你的数据集
    # train_dataset = SignLanguageDataset("datasets/signllm_data_complete/ASL/train/")
    # val_dataset = SignLanguageDataset("datasets/signllm_data_complete/ASL/dev/")
    # 
    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print("⚠️  请根据你的数据集路径修改数据加载部分")
    print("   然后运行: trainer.train(train_dataloader, val_dataloader, NUM_EPOCHS)")
    
    # 示例：如果你有检查点要继续训练
    # if Path("improved_checkpoints/epoch_10.pth").exists():
    #     trainer.load_checkpoint("improved_checkpoints/epoch_10.pth")
    
    # 开始训练
    # trainer.train(train_dataloader, val_dataloader, NUM_EPOCHS)


if __name__ == "__main__":
    main() 