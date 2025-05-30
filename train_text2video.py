import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import os
import argparse
from tqdm import tqdm
import wandb
from text2video_model import PoseUNet1D, TextToPoseDiffusion
from video_dataset import create_asl_dataloader
import matplotlib.pyplot as plt
from datetime import datetime
import glob

class ASLTextToPoseTrainer:
    """ASL Text-to-Pose Diffusion Model训练器"""
    
    def __init__(
        self,
        model: PoseUNet1D,
        diffusion: TextToPoseDiffusion,
        data_dir: str,
        device: str = "cuda",
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        num_epochs: int = 1000,
        save_interval: int = 5,
        log_interval: int = 10,
        sample_interval: int = 200,
        use_wandb: bool = True,
        project_name: str = "asl-text2pose-diffusion",
        num_frames: int = 100,
        pose_dim: int = 120,
        max_samples: int = None,
        resume_from: str = None,
        lr_scheduler_type: str = "plateau"  # "plateau" 或 "cosine"
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device
        self.num_epochs = num_epochs
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.num_frames = num_frames
        self.pose_dim = pose_dim
        self.lr_scheduler_type = lr_scheduler_type
        
        # 训练状态跟踪
        self.start_epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.best_loss = float('inf')
        self.lr_patience_counter = 0
        
        # 确保扩散过程的张量在正确的设备上
        self.diffusion.betas = self.diffusion.betas.to(device)
        self.diffusion.alphas = self.diffusion.alphas.to(device)
        self.diffusion.alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
        self.diffusion.alphas_cumprod_prev = self.diffusion.alphas_cumprod_prev.to(device)
        self.diffusion.posterior_variance = self.diffusion.posterior_variance.to(device)
        
        # 创建数据集和加载器
        print("📊 创建数据集...")
        self.dataloader, self.dataset = create_asl_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if os.name == 'nt' else 2,  # Windows使用0
            num_frames=num_frames,
            max_samples=max_samples,
            normalize=True,
            augment=False
        )
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-6
        )
        
        # 学习率调度器
        if lr_scheduler_type == "plateau":
            # 基于验证损失的动态调整
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
                threshold=1e-4,
                min_lr=learning_rate * 0.001
            )
        else:
            # 基于epoch的余弦退火
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=learning_rate * 0.01
            )
        
        # 创建保存目录（如果是resume，使用原目录）
        if resume_from:
            self.save_dir = os.path.dirname(resume_from)
            print(f"🔄 继续训练模式，使用目录: {self.save_dir}")
        else:
            self.save_dir = f"checkpoints/asl_text2pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"🆕 新训练模式，创建目录: {self.save_dir}")
        
        # 载入检查点（如果提供）
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
        elif resume_from:
            print(f"⚠️  检查点文件不存在: {resume_from}")
            print("🆕 从头开始训练...")
        
        # Wandb初始化
        if use_wandb:
            wandb_config = {
                "model_channels": model.model_channels,
                "num_timesteps": diffusion.num_timesteps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "num_frames": num_frames,
                "pose_dim": pose_dim,
                "dataset_size": len(self.dataset),
                "lr_scheduler": lr_scheduler_type,
                "resume_from": resume_from is not None
            }
            
            if resume_from:
                # Resume existing run
                wandb.init(
                    project=project_name,
                    config=wandb_config,
                    resume="allow"
                )
            else:
                # New run
                wandb.init(
                    project=project_name,
                    config=wandb_config
                )
        
        print(f"ASL Text-to-Pose训练器初始化完成:")
        print(f"  设备: {device}")
        print(f"  数据集大小: {len(self.dataset)}")
        print(f"  批次大小: {batch_size}")
        print(f"  初始学习率: {learning_rate}")
        print(f"  学习率调度: {lr_scheduler_type}")
        print(f"  训练轮数: {num_epochs}")
        print(f"  开始epoch: {self.start_epoch}")
        print(f"  姿态规格: {num_frames}帧 x {pose_dim}维")
        print(f"  保存目录: {self.save_dir}")
        
    def train_step(self, batch):
        """执行一个训练步骤"""
        pose_sequences = batch['pose_sequences'].to(self.device)
        captions = batch['captions']
        
        # 随机采样时间步
        t = torch.randint(
            0, self.diffusion.num_timesteps, 
            (pose_sequences.shape[0],), 
            device=self.device
        )
        
        # 计算损失
        loss = self.diffusion.p_losses(self.model, pose_sequences, t, captions)
        
        return loss
    
    def validate(self, num_samples=50):
        """验证模型性能"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i * self.dataloader.batch_size >= num_samples:
                    break
                    
                pose_sequences = batch['pose_sequences'].to(self.device)
                captions = batch['captions']
                
                t = torch.randint(
                    0, self.diffusion.num_timesteps,
                    (pose_sequences.shape[0],), device=self.device
                ).long()
                
                loss = self.diffusion.p_losses(self.model, pose_sequences, t, captions)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0
    
    def sample_poses(self, text_prompts):
        """生成姿态序列样本"""
        self.model.eval()
        
        with torch.no_grad():
            # 生成样本
            samples = self.diffusion.sample(
                self.model, 
                text_prompts=text_prompts,
                num_frames=self.num_frames,
                pose_dim=self.pose_dim
            )
            
            # 使用最后一步的结果
            final_samples = samples[-1]  # (batch_size, T, 120)
            
            # 反标准化
            if hasattr(self.dataset, 'denormalize_pose'):
                denormalized_samples = []
                for i in range(final_samples.shape[0]):
                    denorm_pose = self.dataset.denormalize_pose(final_samples[i].numpy())
                    denormalized_samples.append(denorm_pose)
                final_samples = np.stack(denormalized_samples)
            else:
                final_samples = final_samples.numpy()
        
        self.model.train()
        return final_samples
    
    def visualize_pose_sequence(self, pose_sequence, title="Pose Sequence"):
        """可视化姿态序列"""
        # pose_sequence: (T, 120)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 身体关键点轨迹
        body_coords = pose_sequence[:, :36].reshape(-1, 18, 2)  # (T, 18, 2)
        for joint_idx in range(18):
            if np.any(np.abs(body_coords[:, joint_idx, :]) > 1e-6):
                axes[0, 0].plot(body_coords[:, joint_idx, 0], label=f'Joint {joint_idx}')
        axes[0, 0].set_title('身体关键点 X 坐标轨迹')
        axes[0, 0].set_xlabel('帧数')
        axes[0, 0].set_ylabel('X 坐标')
        
        # 2. 左手关键点轨迹
        left_hand_coords = pose_sequence[:, 36:78].reshape(-1, 21, 2)  # (T, 21, 2)
        for joint_idx in range(0, 21, 5):  # 只显示部分关键点
            if np.any(np.abs(left_hand_coords[:, joint_idx, :]) > 1e-6):
                axes[0, 1].plot(left_hand_coords[:, joint_idx, 0], label=f'Joint {joint_idx}')
        axes[0, 1].set_title('左手关键点 X 坐标轨迹')
        axes[0, 1].set_xlabel('帧数')
        axes[0, 1].set_ylabel('X 坐标')
        
        # 3. 右手关键点轨迹  
        right_hand_coords = pose_sequence[:, 78:120].reshape(-1, 21, 2)  # (T, 21, 2)
        for joint_idx in range(0, 21, 5):  # 只显示部分关键点
            if np.any(np.abs(right_hand_coords[:, joint_idx, :]) > 1e-6):
                axes[1, 0].plot(right_hand_coords[:, joint_idx, 0], label=f'Joint {joint_idx}')
        axes[1, 0].set_title('右手关键点 X 坐标轨迹')
        axes[1, 0].set_xlabel('帧数')
        axes[1, 0].set_ylabel('X 坐标')
        
        # 4. 特征热力图
        im = axes[1, 1].imshow(pose_sequence.T, aspect='auto', cmap='viridis')
        axes[1, 1].set_title('姿态特征热力图')
        axes[1, 1].set_xlabel('帧数')
        axes[1, 1].set_ylabel('特征维度')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def visualize_samples(self, epoch, sample_prompts=None):
        """可视化生成的样本"""
        if sample_prompts is None:
            sample_prompts = [
                "hello",
                "thank you", 
                "good morning",
                "how are you"
            ]
        
        # 限制样本数量以避免内存问题
        sample_prompts = sample_prompts[:2]
        
        try:
            samples = self.sample_poses(sample_prompts)  # (B, T, 120)
            
            for i, (pose_sequence, prompt) in enumerate(zip(samples, sample_prompts)):
                # 可视化姿态序列
                fig = self.visualize_pose_sequence(pose_sequence, f"生成样本: {prompt}")
                
                # 保存图片
                save_path = os.path.join(self.save_dir, f'pose_sample_epoch_{epoch}_{i}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                
                if self.use_wandb:
                    wandb.log({f"generated_pose_{i}": wandb.Image(save_path)}, step=epoch)
                
                plt.close()
            
            print(f"生成样本可视化已保存")
            
        except Exception as e:
            print(f"生成样本时出错: {e}")
            
    def load_checkpoint(self, checkpoint_path):
        """载入检查点继续训练"""
        print(f"📂 载入检查点: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 载入模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ 模型权重载入成功")
            
            # 载入优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ 优化器状态载入成功")
            
            # 载入调度器状态
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ 学习率调度器状态载入成功")
            
            # 载入训练状态
            self.start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个epoch开始
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.global_step = checkpoint.get('global_step', 0)
            
            # 载入损失历史（如果有的话）
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
                print(f"✅ 载入 {len(self.train_losses)} 个epoch的损失历史")
            
            # 验证模型配置匹配
            config = checkpoint.get('config', {})
            if config:
                model_channels = config.get('model_channels', self.model.model_channels)
                num_frames = config.get('num_frames', self.num_frames)
                pose_dim = config.get('pose_dim', self.pose_dim)
                
                if (model_channels != self.model.model_channels or 
                    num_frames != self.num_frames or 
                    pose_dim != self.pose_dim):
                    print("⚠️  模型配置不匹配，可能导致问题:")
                    print(f"   检查点: channels={model_channels}, frames={num_frames}, pose_dim={pose_dim}")
                    print(f"   当前:   channels={self.model.model_channels}, frames={self.num_frames}, pose_dim={self.pose_dim}")
            
            print(f"🎯 继续训练状态:")
            print(f"   开始epoch: {self.start_epoch}")
            print(f"   全局步数: {self.global_step}")
            print(f"   当前最佳损失: {self.best_loss:.6f}")
            print(f"   当前学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            
        except Exception as e:
            print(f"❌ 载入检查点失败: {e}")
            print("🆕 将从头开始训练...")
            self.start_epoch = 0
            self.global_step = 0
            self.best_loss = float('inf')
            
    def save_checkpoint(self, epoch, loss):
        """保存检查点 - 空间节省策略：只保留3个固定文件"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,  # 保存损失历史
            'lr_scheduler_type': self.lr_scheduler_type,
            'config': {
                'num_frames': self.num_frames,
                'pose_dim': self.pose_dim,
                'model_channels': self.model.model_channels,
                'num_timesteps': self.diffusion.num_timesteps,
            },
            'dataset_stats': {
                'pose_mean': self.dataset.pose_mean if hasattr(self.dataset, 'pose_mean') else None,
                'pose_std': self.dataset.pose_std if hasattr(self.dataset, 'pose_std') else None,
            }
        }
        
        # 1. 始终保存最新检查点（覆盖）
        latest_path = os.path.join(self.save_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 2. 如果是最佳模型，保存最佳检查点（覆盖）
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(self.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"🏆 保存最佳模型 (损失: {loss:.6f} -> {self.best_loss:.6f})")
        
        # 3. 每50个epoch保存一次备份（覆盖）
        if epoch % 50 == 0 and epoch > 0:
            backup_path = os.path.join(self.save_dir, 'backup.pth')
            torch.save(checkpoint, backup_path)
            print(f"💾 保存备份检查点 (Epoch {epoch})")
        
        # 显示文件大小信息
        try:
            latest_size = os.path.getsize(latest_path) / (1024*1024)  # MB
            best_path = os.path.join(self.save_dir, 'best.pth')
            backup_path = os.path.join(self.save_dir, 'backup.pth')
            
            print(f"💿 检查点文件状态:")
            print(f"   latest.pth: {latest_size:.1f}MB (Epoch {epoch})")
            
            if os.path.exists(best_path):
                best_size = os.path.getsize(best_path) / (1024*1024)
                print(f"   best.pth: {best_size:.1f}MB (损失: {self.best_loss:.6f})")
            
            if os.path.exists(backup_path):
                backup_size = os.path.getsize(backup_path) / (1024*1024)
                backup_epoch = torch.load(backup_path, map_location='cpu')['epoch']
                print(f"   backup.pth: {backup_size:.1f}MB (Epoch {backup_epoch})")
                
        except Exception as e:
            print(f"⚠️  无法获取文件大小信息: {e}")
    
    def train(self):
        """主训练循环"""
        print("开始训练ASL Text-to-Pose模型...")
        
        self.model.train()
        
        # 如果是继续训练，显示恢复信息
        if self.start_epoch > 0:
            print(f"🔄 从第 {self.start_epoch+1} 个epoch继续训练")
            print(f"   当前最佳损失: {self.best_loss:.6f}")
            print(f"   全局步数: {self.global_step}")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_losses = []
            
            # 训练一个epoch
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
            
            for batch in pbar:
                loss = self.train_step(batch)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                # 将损失转换为CPU标量
                loss_value = loss.item()
                epoch_losses.append(loss_value)
                self.global_step += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss_value:.6f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'best': f'{self.best_loss:.6f}'
                })
                
                # 记录日志
                if self.global_step % self.log_interval == 0 and self.use_wandb:
                    wandb.log({
                        'train_loss': loss_value,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'best_loss': self.best_loss
                    }, step=self.global_step)
            
            # 计算epoch平均损失
            avg_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_loss)
            
            # 验证和学习率调整
            val_loss = None
            if epoch % 10 == 0:  # 每10个epoch验证一次
                val_loss = self.validate()
                print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
                
                # 根据调度器类型调整学习率
                if self.lr_scheduler_type == "plateau":
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(val_loss)  # 基于验证损失
                    new_lr = self.optimizer.param_groups[0]['lr']
                    
                    if new_lr < old_lr:
                        print(f"📉 学习率降低: {old_lr:.2e} -> {new_lr:.2e}")
                        self.lr_patience_counter = 0
                    else:
                        self.lr_patience_counter += 1
                        
                else:
                    self.scheduler.step()  # 基于epoch
                
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'avg_train_loss': avg_loss,
                        'val_loss': val_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'lr_patience': self.lr_patience_counter
                    }, step=self.global_step)
            else:
                # 非验证epoch，仍需调整学习率（如果是cosine调度）
                if self.lr_scheduler_type == "cosine":
                    self.scheduler.step()
            
            # 早停检查（如果学习率过小且很久没有改善）
            current_lr = self.optimizer.param_groups[0]['lr']
            if (self.lr_scheduler_type == "plateau" and 
                current_lr < 1e-7 and 
                self.lr_patience_counter > 20):
                print(f"🛑 学习率过小 ({current_lr:.2e}) 且长时间无改善，提前停止训练")
                break
            
            # 生成样本
            if epoch % self.sample_interval == 0:
                print(f"🎨 生成姿态样本 (Epoch {epoch+1})")
                self.visualize_samples(epoch)
            
            # 保存检查点
            if epoch % self.save_interval == 0 or epoch == self.num_epochs - 1:
                self.save_checkpoint(epoch, avg_loss)
                
            # 简单打印（非验证epoch）
            if epoch % 10 != 0:
                print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}, LR = {current_lr:.2e}")
        
        print("训练完成!")
        print(f"🏆 最佳损失: {self.best_loss:.6f}")
        print(f"📊 总训练步数: {self.global_step}")
        
        if self.use_wandb:
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='训练ASL Text-to-Pose Diffusion Model')
    parser.add_argument('--data_dir', type=str, default='datasets/signllm_training_data/ASL./dev', help='ASL数据目录路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--model_channels', type=int, default=256, help='模型通道数')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='扩散步数')
    parser.add_argument('--num_frames', type=int, default=100, help='序列帧数')
    parser.add_argument('--pose_dim', type=int, default=120, help='姿态特征维度')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--no_wandb', action='store_true', help='禁用wandb日志记录')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数量（用于调试）')
    parser.add_argument('--resume_from', type=str, default=None, help='从哪个检查点继续训练')
    parser.add_argument('--lr_scheduler_type', type=str, default="plateau", help='学习率调度器类型')
    
    args = parser.parse_args()
    
    # 🔍 自动检测现有检查点
    print("🔍 检查现有训练检查点...")
    checkpoint_dirs = glob.glob("checkpoints/asl_text2pose_*")
    
    if args.resume_from is None and checkpoint_dirs:
        # 找到最新的检查点目录
        latest_checkpoint_dir = max(checkpoint_dirs, key=os.path.getmtime)
        
        # 优先级：latest.pth > best.pth > backup.pth
        possible_checkpoints = [
            os.path.join(latest_checkpoint_dir, 'latest.pth'),
            os.path.join(latest_checkpoint_dir, 'best.pth'),
            os.path.join(latest_checkpoint_dir, 'backup.pth')
        ]
        
        for checkpoint_path in possible_checkpoints:
            if os.path.exists(checkpoint_path):
                args.resume_from = checkpoint_path
                print(f"🎯 自动检测到检查点: {checkpoint_path}")
                break
    
    if args.resume_from:
        print(f"📂 将从检查点继续训练: {args.resume_from}")
    else:
        print("🆕 未找到检查点，将从头开始训练")
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"❌ 数据目录不存在: {args.data_dir}")
        print("📁 请检查数据路径是否正确")
        return
    
    # 显示训练配置
    print(f"\n📋 训练配置:")
    print(f"   数据目录: {args.data_dir}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   学习率: {args.learning_rate}")
    print(f"   训练轮数: {args.num_epochs}")
    print(f"   模型通道数: {args.model_channels}")
    print(f"   扩散步数: {args.num_timesteps}")
    print(f"   序列长度: {args.num_frames}帧")
    print(f"   姿态维度: {args.pose_dim}维")
    print(f"   学习率调度: {args.lr_scheduler_type}")
    print(f"   设备: {args.device}")
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，切换到CPU")
        args.device = 'cpu'
    elif args.device == 'cuda':
        print(f"🚀 使用GPU: {torch.cuda.get_device_name()}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 创建模型
    print("\n🏗️  创建模型...")
    model = PoseUNet1D(
        pose_dim=args.pose_dim,
        model_channels=args.model_channels,
        num_frames=args.num_frames
    )
    
    # 创建扩散过程
    diffusion = TextToPoseDiffusion(
        num_timesteps=args.num_timesteps,
        beta_schedule='cosine'
    )
    
    # 移动diffusion的参数到设备
    diffusion.betas = diffusion.betas.to(args.device)
    diffusion.alphas = diffusion.alphas.to(args.device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(args.device)
    diffusion.alphas_cumprod_prev = diffusion.alphas_cumprod_prev.to(args.device)
    diffusion.posterior_variance = diffusion.posterior_variance.to(args.device)
    
    # 创建训练器
    trainer = ASLTextToPoseTrainer(
        model=model,
        diffusion=diffusion,
        data_dir=args.data_dir,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_frames=args.num_frames,
        pose_dim=args.pose_dim,
        max_samples=args.max_samples,
        resume_from=args.resume_from,
        lr_scheduler_type=args.lr_scheduler_type,
        use_wandb=not args.no_wandb
    )
    
    # 如果是调试模式，限制数据量
    if args.max_samples:
        trainer.dataset.data_paths = trainer.dataset.data_paths[:args.max_samples]
        trainer.dataset.captions = trainer.dataset.captions[:args.max_samples]
        print(f"🐛 调试模式：限制数据集大小为 {len(trainer.dataset)} 个样本")
    
    # 显示数据集信息
    print(f"\n📊 数据集信息:")
    print(f"   总样本数: {len(trainer.dataset)}")
    print(f"   每批次: {args.batch_size} 个样本")
    print(f"   总批次数: {len(trainer.dataloader)}")
    print(f"   预计每epoch时间: ~{len(trainer.dataloader) * args.batch_size / 100:.1f}分钟")
    
    # 开始训练
    print(f"\n🚀 开始训练...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        print("💾 正在保存当前状态...")
        # 保存中断时的检查点
        if hasattr(trainer, 'train_losses') and trainer.train_losses:
            current_epoch = trainer.start_epoch + len(trainer.train_losses) - 1
            current_loss = trainer.train_losses[-1]
            trainer.save_checkpoint(current_epoch, current_loss)
            print("✅ 检查点已保存，可以使用 --resume_from 继续训练")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 训练脚本执行完成！")

if __name__ == "__main__":
    main() 