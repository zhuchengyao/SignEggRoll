import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import argparse
from tqdm import tqdm
import wandb
from diffusion_model import UNet1D, GaussianDiffusion
from pose_dataset import create_dataloader
import matplotlib.pyplot as plt
from datetime import datetime

class DiffusionTrainer:
    """Diffusion Model训练器"""
    
    def __init__(
        self,
        model: UNet1D,
        diffusion: GaussianDiffusion,
        data_dir: str,
        device: str = "cuda",
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 1000,
        save_interval: int = 100,
        log_interval: int = 10,
        sample_interval: int = 200,
        use_wandb: bool = True,
        project_name: str = "pose-diffusion"
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device
        self.num_epochs = num_epochs
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        
        # 创建数据加载器
        self.dataloader, self.dataset = create_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            normalize=True,
            augment=True
        )
        
        # 优化器和调度器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-6
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01
        )
        
        # 损失跟踪
        self.train_losses = []
        self.best_loss = float('inf')
        
        # 创建保存目录
        self.save_dir = f"checkpoints/diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Wandb初始化
        if use_wandb:
            wandb.init(
                project=project_name,
                config={
                    "model_channels": model.model_channels,
                    "num_timesteps": diffusion.num_timesteps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "dataset_size": len(self.dataset)
                }
            )
        self.use_wandb = use_wandb
        
        print(f"训练器初始化完成:")
        print(f"  设备: {device}")
        print(f"  数据集大小: {len(self.dataset)}")
        print(f"  批次大小: {batch_size}")
        print(f"  学习率: {learning_rate}")
        print(f"  训练轮数: {num_epochs}")
        print(f"  保存目录: {self.save_dir}")
        
    def train_step(self, batch):
        """单步训练"""
        poses = batch.to(self.device)  # (batch_size, 67, 3)
        
        # 随机采样时间步
        t = torch.randint(
            0, self.diffusion.num_timesteps, 
            (poses.shape[0],), device=self.device
        ).long()
        
        # 计算损失
        loss = self.diffusion.p_losses(self.model, poses, t)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, num_samples=100):
        """验证模型性能"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i * self.dataloader.batch_size >= num_samples:
                    break
                    
                poses = batch.to(self.device)
                t = torch.randint(
                    0, self.diffusion.num_timesteps,
                    (poses.shape[0],), device=self.device
                ).long()
                
                loss = self.diffusion.p_losses(self.model, poses, t)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0
    
    def sample_poses(self, num_samples=4):
        """生成姿态样本"""
        self.model.eval()
        
        with torch.no_grad():
            # 生成样本
            samples = self.diffusion.sample(
                self.model, 
                num_samples=num_samples,
                num_keypoints=67
            )
            
            # 使用最后一步的结果
            final_samples = samples[-1]  # (num_samples, 67, 3)
            
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
    
    def visualize_samples(self, epoch, num_samples=4):
        """可视化生成的样本"""
        samples = self.sample_poses(num_samples)
        
        fig = plt.figure(figsize=(16, 4))
        
        for i in range(num_samples):
            pose = samples[i]  # (67, 3)
            
            ax = fig.add_subplot(1, num_samples, i+1, projection='3d')
            
            # 绘制关键点
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], 
                      c='blue', marker='o', s=30, alpha=0.7)
            
            # 设置标题和坐标轴
            ax.set_title(f'生成样本 {i+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # 设置相同的缩放比例
            max_range = np.array([
                pose[:,0].max()-pose[:,0].min(),
                pose[:,1].max()-pose[:,1].min(),
                pose[:,2].max()-pose[:,2].min()
            ]).max() / 2.0
            
            mid_x = (pose[:,0].max()+pose[:,0].min()) * 0.5
            mid_y = (pose[:,1].max()+pose[:,1].min()) * 0.5
            mid_z = (pose[:,2].max()+pose[:,2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f'samples_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if self.use_wandb:
            wandb.log({"generated_samples": wandb.Image(save_path)}, step=epoch)
        
        plt.close()
        
    def save_checkpoint(self, epoch, loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'dataset_stats': {
                'mean': self.dataset.mean if hasattr(self.dataset, 'mean') else None,
                'std': self.dataset.std if hasattr(self.dataset, 'std') else None,
            }
        }
        
        # 保存最新检查点
        save_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, save_path)
        
        # 保存最佳模型
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型 (损失: {loss:.6f})")
        
        # 定期保存检查点
        if epoch % self.save_interval == 0:
            epoch_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """主训练循环"""
        print("开始训练...")
        
        self.model.train()
        global_step = 0
        
        for epoch in range(self.num_epochs):
            epoch_losses = []
            
            # 训练一个epoch
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
            
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                global_step += 1
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss:.6f}'})
                
                # 记录日志
                if global_step % self.log_interval == 0 and self.use_wandb:
                    wandb.log({
                        'train_loss': loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }, step=global_step)
            
            # 计算epoch平均损失
            avg_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_loss)
            
            # 更新学习率
            self.scheduler.step()
            
            # 验证
            if epoch % 50 == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
                
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'avg_train_loss': avg_loss,
                        'val_loss': val_loss
                    }, step=global_step)
            
            # 生成样本
            if epoch % self.sample_interval == 0:
                print(f"生成样本 (Epoch {epoch+1})")
                self.visualize_samples(epoch)
            
            # 保存检查点
            if epoch % self.save_interval == 0 or epoch == self.num_epochs - 1:
                self.save_checkpoint(epoch, avg_loss)
        
        print("训练完成!")
        
        if self.use_wandb:
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='训练Diffusion Model for 3D Pose Generation')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--model_channels', type=int, default=128, help='模型通道数')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='扩散步数')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--no_wandb', action='store_true', help='不使用wandb')
    parser.add_argument('--max_files', type=int, default=None, help='最大文件数量（调试用）')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    # 创建模型
    model = UNet1D(
        model_channels=args.model_channels,
        num_keypoints=67
    )
    
    # 创建扩散过程
    diffusion = GaussianDiffusion(
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
    trainer = DiffusionTrainer(
        model=model,
        diffusion=diffusion,
        data_dir=args.data_dir,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        use_wandb=not args.no_wandb
    )
    
    # 如果是调试模式，限制数据量
    if args.max_files:
        trainer.dataset.poses = trainer.dataset.poses[:args.max_files * 100]  # 假设每个文件100帧
        print(f"调试模式：限制数据集大小为 {len(trainer.dataset)} 个样本")
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 