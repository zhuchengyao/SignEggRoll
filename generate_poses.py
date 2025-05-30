import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diffusion_model import UNet1D, GaussianDiffusion
import json
from datetime import datetime

class PoseGenerator:
    """姿态生成器"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 提取模型配置（如果保存了的话）
        model_config = checkpoint.get('model_config', {})
        
        # 创建模型
        self.model = UNet1D(
            model_channels=model_config.get('model_channels', 128),
            num_keypoints=67
        )
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # 创建扩散过程
        self.diffusion = GaussianDiffusion(
            num_timesteps=model_config.get('num_timesteps', 1000),
            beta_schedule='cosine'
        )
        
        # 移动diffusion参数到设备
        self.diffusion.betas = self.diffusion.betas.to(device)
        self.diffusion.alphas = self.diffusion.alphas.to(device)
        self.diffusion.alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
        self.diffusion.alphas_cumprod_prev = self.diffusion.alphas_cumprod_prev.to(device)
        self.diffusion.posterior_variance = self.diffusion.posterior_variance.to(device)
        
        # 加载数据集统计信息
        self.dataset_stats = checkpoint.get('dataset_stats', {})
        self.mean = self.dataset_stats.get('mean')
        self.std = self.dataset_stats.get('std')
        
        print(f"模型加载完成:")
        print(f"  设备: {device}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  损失: {checkpoint['loss']:.6f}")
        print(f"  标准化: {'是' if self.mean is not None else '否'}")
        
    def denormalize_pose(self, pose):
        """反标准化姿态"""
        if self.mean is not None and self.std is not None:
            return pose * self.std + self.mean
        return pose
    
    def generate_poses(self, num_samples: int = 1, return_trajectory: bool = False):
        """生成姿态"""
        print(f"正在生成 {num_samples} 个姿态样本...")
        
        with torch.no_grad():
            if return_trajectory:
                # 返回完整的去噪轨迹
                samples_trajectory = self.diffusion.p_sample_loop(
                    self.model, 
                    shape=(num_samples, 67, 3)
                )
                
                # 反标准化每一步
                if self.mean is not None:
                    denorm_trajectory = []
                    for step_samples in samples_trajectory:
                        denorm_step = []
                        for i in range(step_samples.shape[0]):
                            denorm_pose = self.denormalize_pose(step_samples[i].numpy())
                            denorm_step.append(denorm_pose)
                        denorm_trajectory.append(np.stack(denorm_step))
                    return denorm_trajectory
                else:
                    return [step.numpy() for step in samples_trajectory]
            else:
                # 只返回最终结果
                final_samples = self.diffusion.sample(
                    self.model,
                    num_samples=num_samples,
                    num_keypoints=67
                )[-1]  # 取最后一步
                
                # 反标准化
                if self.mean is not None:
                    denorm_samples = []
                    for i in range(final_samples.shape[0]):
                        denorm_pose = self.denormalize_pose(final_samples[i].numpy())
                        denorm_samples.append(denorm_pose)
                    return np.stack(denorm_samples)
                else:
                    return final_samples.numpy()
    
    def visualize_poses(self, poses, save_path: str = None, title: str = "Generated Poses"):
        """可视化生成的姿态"""
        num_poses = poses.shape[0]
        
        # 计算网格布局
        cols = min(4, num_poses)
        rows = (num_poses + cols - 1) // cols
        
        fig = plt.figure(figsize=(4 * cols, 4 * rows))
        
        for i in range(num_poses):
            pose = poses[i]  # (67, 3)
            
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            
            # 绘制关键点
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], 
                      c='blue', marker='o', s=30, alpha=0.7)
            
            # 设置标题和坐标轴
            ax.set_title(f'样本 {i+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # 设置相同的缩放比例
            if not np.allclose(pose, 0):
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
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果保存到: {save_path}")
        
        plt.show()
    
    def save_poses(self, poses, save_path: str, format: str = 'skels'):
        """保存生成的姿态"""
        if format == 'skels':
            # 保存为.skels格式
            with open(save_path, 'w') as f:
                for pose in poses:
                    # 将(67, 3)展平为(201,)
                    flattened = pose.flatten()
                    line = ' '.join([f'{x:.6f}' for x in flattened])
                    f.write(line + '\n')
        
        elif format == 'npy':
            # 保存为numpy格式
            np.save(save_path, poses)
        
        elif format == 'json':
            # 保存为JSON格式
            poses_list = poses.tolist()
            with open(save_path, 'w') as f:
                json.dump(poses_list, f, indent=2)
        
        print(f"姿态数据保存到: {save_path} (格式: {format})")
    
    def interpolate_poses(self, pose1, pose2, num_steps: int = 10):
        """在两个姿态之间插值"""
        alphas = np.linspace(0, 1, num_steps)
        interpolated = []
        
        for alpha in alphas:
            interp_pose = (1 - alpha) * pose1 + alpha * pose2
            interpolated.append(interp_pose)
        
        return np.stack(interpolated)
    
    def generate_animation_frames(self, num_frames: int = 30, interpolate: bool = True):
        """生成动画帧"""
        if interpolate:
            # 生成关键帧
            num_keyframes = max(2, num_frames // 10)
            keyframes = self.generate_poses(num_keyframes)
            
            # 在关键帧之间插值
            frames = []
            frames_per_segment = num_frames // (num_keyframes - 1)
            
            for i in range(num_keyframes - 1):
                interp_frames = self.interpolate_poses(
                    keyframes[i], keyframes[i+1], frames_per_segment
                )
                frames.extend(interp_frames[:-1])  # 避免重复最后一帧
            
            frames.append(keyframes[-1])  # 添加最后一帧
            return np.stack(frames[:num_frames])
        else:
            # 直接生成独立帧
            return self.generate_poses(num_frames)

def create_gif_from_poses(poses, save_path: str, duration: float = 0.1):
    """从姿态序列创建GIF动画"""
    import matplotlib.animation as animation
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算所有帧的边界
    all_poses = np.concatenate(poses, axis=0)
    max_range = np.array([
        all_poses[:,0].max()-all_poses[:,0].min(),
        all_poses[:,1].max()-all_poses[:,1].min(),
        all_poses[:,2].max()-all_poses[:,2].min()
    ]).max() / 2.0
    
    mid_x = (all_poses[:,0].max()+all_poses[:,0].min()) * 0.5
    mid_y = (all_poses[:,1].max()+all_poses[:,1].min()) * 0.5
    mid_z = (all_poses[:,2].max()+all_poses[:,2].min()) * 0.5
    
    def animate(frame):
        ax.clear()
        pose = poses[frame]
        
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], 
                  c='blue', marker='o', s=30, alpha=0.7)
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'帧 {frame+1}/{len(poses)}')
    
    ani = animation.FuncAnimation(fig, animate, frames=len(poses), 
                                 interval=duration*1000, blit=False)
    
    ani.save(save_path, writer='pillow', fps=1/duration)
    plt.close()
    print(f"GIF动画保存到: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='从训练好的Diffusion Model生成3D姿态')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--num_samples', type=int, default=8, help='生成样本数量')
    parser.add_argument('--output_dir', type=str, default='generated_poses', help='输出目录')
    parser.add_argument('--format', type=str, choices=['skels', 'npy', 'json'], 
                       default='skels', help='保存格式')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    parser.add_argument('--animation', action='store_true', help='生成动画')
    parser.add_argument('--num_frames', type=int, default=30, help='动画帧数')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建生成器
    generator = PoseGenerator(args.checkpoint, args.device)
    
    if args.animation:
        # 生成动画
        print(f"生成动画 ({args.num_frames}帧)...")
        frames = generator.generate_animation_frames(args.num_frames, interpolate=True)
        
        # 保存动画帧
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        frames_path = os.path.join(args.output_dir, f'animation_frames_{timestamp}.{args.format}')
        generator.save_poses(frames, frames_path, args.format)
        
        # 创建GIF
        gif_path = os.path.join(args.output_dir, f'pose_animation_{timestamp}.gif')
        create_gif_from_poses(frames, gif_path, duration=0.1)
        
        # 可视化几个关键帧
        if args.visualize:
            key_frames = frames[::max(1, len(frames)//8)]  # 选择8个关键帧
            viz_path = os.path.join(args.output_dir, f'animation_keyframes_{timestamp}.png')
            generator.visualize_poses(key_frames, viz_path, "Animation Key Frames")
    
    else:
        # 生成静态姿态
        poses = generator.generate_poses(args.num_samples)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(args.output_dir, f'generated_poses_{timestamp}.{args.format}')
        generator.save_poses(poses, save_path, args.format)
        
        # 可视化
        if args.visualize:
            viz_path = os.path.join(args.output_dir, f'generated_poses_{timestamp}.png')
            generator.visualize_poses(poses, viz_path)
        
        print(f"成功生成 {len(poses)} 个姿态样本")

if __name__ == "__main__":
    main() 