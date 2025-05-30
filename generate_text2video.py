import torch
import numpy as np
import os
import argparse
import json
from text2video_model import PoseUNet1D, TextToPoseDiffusion
import matplotlib.pyplot as plt
from datetime import datetime

class ASLTextToPoseGenerator:
    """ASL Text-to-Pose生成器"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # 加载检查点
        print(f"从 {checkpoint_path} 加载模型...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 获取配置
        config = checkpoint.get('config', {})
        self.num_frames = config.get('num_frames', 100)
        self.pose_dim = config.get('pose_dim', 120)
        model_channels = config.get('model_channels', 256)
        num_timesteps = config.get('num_timesteps', 1000)
        
        # 创建模型
        self.model = PoseUNet1D(
            pose_dim=self.pose_dim,
            model_channels=model_channels,
            num_frames=self.num_frames
        ).to(device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 创建扩散过程
        self.diffusion = TextToPoseDiffusion(
            num_timesteps=num_timesteps,
            beta_schedule='cosine'
        )
        
        # 移动diffusion的参数到设备
        self.diffusion.betas = self.diffusion.betas.to(device)
        self.diffusion.alphas = self.diffusion.alphas.to(device)
        self.diffusion.alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
        self.diffusion.alphas_cumprod_prev = self.diffusion.alphas_cumprod_prev.to(device)
        self.diffusion.posterior_variance = self.diffusion.posterior_variance.to(device)
        
        # 加载数据集统计信息用于反标准化
        dataset_stats = checkpoint.get('dataset_stats', {})
        self.pose_mean = dataset_stats.get('pose_mean')
        self.pose_std = dataset_stats.get('pose_std')
        
        if self.pose_mean is not None:
            self.pose_mean = torch.tensor(self.pose_mean, device=device)
            self.pose_std = torch.tensor(self.pose_std, device=device)
        
        print(f"模型加载完成:")
        print(f"  设备: {device}")
        print(f"  姿态规格: {self.num_frames}帧 x {self.pose_dim}维")
        print(f"  模型通道数: {model_channels}")
        print(f"  扩散步数: {num_timesteps}")
    
    @torch.no_grad()
    def generate_pose(self, text_prompt: str, num_inference_steps: int = None):
        """生成单个姿态序列"""
        if num_inference_steps is None:
            num_inference_steps = self.diffusion.num_timesteps
        
        print(f"正在生成姿态序列: '{text_prompt}'")
        
        # 生成姿态序列
        pose_sequences = self.diffusion.sample(
            self.model,
            text_prompts=[text_prompt],
            num_frames=self.num_frames,
            pose_dim=self.pose_dim
        )
        
        # 获取最终结果
        final_pose = pose_sequences[-1][0]  # (T, 120)
        
        # 反标准化
        if self.pose_mean is not None and self.pose_std is not None:
            final_pose = final_pose * self.pose_std + self.pose_mean
        
        return final_pose.cpu().numpy()
    
    @torch.no_grad()
    def generate_batch(self, text_prompts: list, num_inference_steps: int = None):
        """批量生成姿态序列"""
        if num_inference_steps is None:
            num_inference_steps = self.diffusion.num_timesteps
        
        print(f"正在批量生成 {len(text_prompts)} 个姿态序列")
        
        # 生成姿态序列
        pose_sequences = self.diffusion.sample(
            self.model,
            text_prompts=text_prompts,
            num_frames=self.num_frames,
            pose_dim=self.pose_dim
        )
        
        # 获取最终结果
        final_poses = pose_sequences[-1]  # (B, T, 120)
        
        # 反标准化
        if self.pose_mean is not None and self.pose_std is not None:
            final_poses = final_poses * self.pose_std + self.pose_mean
        
        return final_poses.cpu().numpy()
    
    def save_pose_sequence(self, pose_array: np.ndarray, save_path: str, format: str = 'json'):
        """保存姿态序列为文件"""
        # pose_array: (T, 120)
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if format == 'json':
            # 转换为ASL数据格式
            poses_data = self.convert_to_asl_format(pose_array)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(poses_data, f, indent=2)
                
        elif format == 'npy':
            np.save(save_path, pose_array)
            
        elif format == 'csv':
            import pandas as pd
            # 创建列名
            columns = []
            # 身体关键点
            for i in range(18):
                columns.extend([f'body_{i}_x', f'body_{i}_y'])
            # 左手关键点
            for i in range(21):
                columns.extend([f'left_hand_{i}_x', f'left_hand_{i}_y'])
            # 右手关键点
            for i in range(21):
                columns.extend([f'right_hand_{i}_x', f'right_hand_{i}_y'])
            
            df = pd.DataFrame(pose_array, columns=columns)
            df.to_csv(save_path, index=False)
        
        print(f"姿态序列已保存到: {save_path}")
    
    def convert_to_asl_format(self, pose_array: np.ndarray):
        """转换为ASL数据格式"""
        poses_data = {'poses': []}
        
        for frame_idx in range(pose_array.shape[0]):
            frame_data = pose_array[frame_idx]  # (120,)
            
            # 分离身体、左手、右手数据
            body_data = frame_data[:36].reshape(18, 2)  # (18, 2)
            left_hand_data = frame_data[36:78].reshape(21, 2)  # (21, 2)
            right_hand_data = frame_data[78:120].reshape(21, 2)  # (21, 2)
            
            # 构建OpenPose格式
            pose_keypoints_2d = []
            for i in range(18):
                pose_keypoints_2d.extend([float(body_data[i, 0]), float(body_data[i, 1]), 1.0])
            
            hand_left_keypoints_2d = []
            for i in range(21):
                hand_left_keypoints_2d.extend([float(left_hand_data[i, 0]), float(left_hand_data[i, 1]), 1.0])
            
            hand_right_keypoints_2d = []
            for i in range(21):
                hand_right_keypoints_2d.extend([float(right_hand_data[i, 0]), float(right_hand_data[i, 1]), 1.0])
            
            frame_pose = {
                'pose_keypoints_2d': pose_keypoints_2d,
                'hand_left_keypoints_2d': hand_left_keypoints_2d,
                'hand_right_keypoints_2d': hand_right_keypoints_2d,
                'face_keypoints_2d': [0.0] * 210  # 空的面部关键点
            }
            
            poses_data['poses'].append(frame_pose)
        
        return poses_data
    
    def visualize_pose_sequence(self, pose_array: np.ndarray, text_prompt: str, save_path: str = None):
        """可视化姿态序列"""
        # pose_array: (T, 120)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 分离不同部分的数据
        body_coords = pose_array[:, :36].reshape(-1, 18, 2)  # (T, 18, 2)
        left_hand_coords = pose_array[:, 36:78].reshape(-1, 21, 2)  # (T, 21, 2)
        right_hand_coords = pose_array[:, 78:120].reshape(-1, 21, 2)  # (T, 21, 2)
        
        # 1. 身体关键点X坐标轨迹
        for joint_idx in [1, 2, 5]:  # 显示躯干和肩膀
            if np.any(np.abs(body_coords[:, joint_idx, :]) > 1e-6):
                axes[0, 0].plot(body_coords[:, joint_idx, 0], label=f'Joint {joint_idx}')
        axes[0, 0].set_title('身体关键点 X 坐标轨迹')
        axes[0, 0].set_xlabel('帧数')
        axes[0, 0].set_ylabel('X 坐标')
        axes[0, 0].legend()
        
        # 2. 身体关键点Y坐标轨迹
        for joint_idx in [1, 2, 5]:
            if np.any(np.abs(body_coords[:, joint_idx, :]) > 1e-6):
                axes[0, 1].plot(body_coords[:, joint_idx, 1], label=f'Joint {joint_idx}')
        axes[0, 1].set_title('身体关键点 Y 坐标轨迹')
        axes[0, 1].set_xlabel('帧数')
        axes[0, 1].set_ylabel('Y 坐标')
        axes[0, 1].legend()
        
        # 3. 左手关键点轨迹
        for joint_idx in [0, 4, 8, 12, 16, 20]:  # 手腕和指尖
            if np.any(np.abs(left_hand_coords[:, joint_idx, :]) > 1e-6):
                axes[0, 2].plot(left_hand_coords[:, joint_idx, 0], label=f'Joint {joint_idx}')
        axes[0, 2].set_title('左手关键点 X 坐标轨迹')
        axes[0, 2].set_xlabel('帧数')
        axes[0, 2].set_ylabel('X 坐标')
        axes[0, 2].legend()
        
        # 4. 右手关键点轨迹
        for joint_idx in [0, 4, 8, 12, 16, 20]:  # 手腕和指尖
            if np.any(np.abs(right_hand_coords[:, joint_idx, :]) > 1e-6):
                axes[1, 0].plot(right_hand_coords[:, joint_idx, 0], label=f'Joint {joint_idx}')
        axes[1, 0].set_title('右手关键点 X 坐标轨迹')
        axes[1, 0].set_xlabel('帧数')
        axes[1, 0].set_ylabel('X 坐标')
        axes[1, 0].legend()
        
        # 5. 特征热力图
        im = axes[1, 1].imshow(pose_array.T, aspect='auto', cmap='viridis')
        axes[1, 1].set_title('姿态特征热力图')
        axes[1, 1].set_xlabel('帧数')
        axes[1, 1].set_ylabel('特征维度')
        plt.colorbar(im, ax=axes[1, 1])
        
        # 6. 手部轨迹2D可视化（取中间帧）
        mid_frame = len(pose_array) // 2
        left_hand_frame = left_hand_coords[mid_frame]  # (21, 2)
        right_hand_frame = right_hand_coords[mid_frame]  # (21, 2)
        
        # 绘制手形
        axes[1, 2].scatter(left_hand_frame[:, 0], -left_hand_frame[:, 1], 
                          c='blue', s=30, alpha=0.7, label='左手')
        axes[1, 2].scatter(right_hand_frame[:, 0], -right_hand_frame[:, 1], 
                          c='red', s=30, alpha=0.7, label='右手')
        axes[1, 2].set_title(f'手部姿态 (第{mid_frame}帧)')
        axes[1, 2].set_xlabel('X 坐标')
        axes[1, 2].set_ylabel('Y 坐标')
        axes[1, 2].legend()
        axes[1, 2].set_aspect('equal')
        
        plt.suptitle(f"生成的ASL姿态: {text_prompt}", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化图片已保存到: {save_path}")
        
        plt.show()
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='ASL Text-to-Pose姿态生成')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--prompts', type=str, nargs='+', required=True, help='文本提示（可以多个）')
    parser.add_argument('--output_dir', type=str, default='./generated_poses', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--format', type=str, choices=['json', 'npy', 'csv'], default='json', help='保存格式')
    parser.add_argument('--visualize', action='store_true', help='可视化生成结果')
    parser.add_argument('--num_inference_steps', type=int, default=None, help='推理步数（默认使用训练时的步数）')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    # 创建生成器
    generator = ASLTextToPoseGenerator(args.checkpoint, args.device)
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"generation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成姿态序列
    if len(args.prompts) == 1:
        # 单个姿态生成
        text_prompt = args.prompts[0]
        pose_sequence = generator.generate_pose(text_prompt, args.num_inference_steps)
        
        # 保存姿态序列
        safe_prompt = text_prompt[:30].replace(' ', '_').replace('/', '_')
        pose_path = os.path.join(output_dir, f"pose_{safe_prompt}.{args.format}")
        generator.save_pose_sequence(pose_sequence, pose_path, args.format)
        
        # 可视化（如果需要）
        if args.visualize:
            vis_path = os.path.join(output_dir, f"visualization_{safe_prompt}.png")
            generator.visualize_pose_sequence(pose_sequence, text_prompt, vis_path)
    
    else:
        # 批量生成
        pose_sequences = generator.generate_batch(args.prompts, args.num_inference_steps)
        
        for i, (pose_sequence, text_prompt) in enumerate(zip(pose_sequences, args.prompts)):
            # 保存姿态序列
            safe_prompt = text_prompt[:30].replace(' ', '_').replace('/', '_')
            pose_path = os.path.join(output_dir, f"pose_{i:03d}_{safe_prompt}.{args.format}")
            generator.save_pose_sequence(pose_sequence, pose_path, args.format)
            
            # 可视化（如果需要）
            if args.visualize:
                vis_path = os.path.join(output_dir, f"visualization_{i:03d}_{safe_prompt}.png")
                generator.visualize_pose_sequence(pose_sequence, text_prompt, vis_path)
    
    print(f"\n生成完成！所有文件已保存到: {output_dir}")

def interactive_generation():
    """交互式生成模式"""
    print("=== ASL Text-to-Pose 交互式生成 ===")
    
    # 获取模型路径
    checkpoint_path = input("请输入模型检查点路径: ").strip()
    if not os.path.exists(checkpoint_path):
        print("模型文件不存在！")
        return
    
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建生成器
    try:
        generator = ASLTextToPoseGenerator(checkpoint_path, device)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 创建输出目录
    output_dir = "./interactive_pose_generation"
    os.makedirs(output_dir, exist_ok=True)
    
    while True:
        print("\n" + "="*50)
        text_prompt = input("请输入ASL文本描述 (输入'quit'退出): ").strip()
        
        if text_prompt.lower() == 'quit':
            break
        
        if not text_prompt:
            print("文本提示不能为空！")
            continue
        
        try:
            # 生成姿态序列
            print("正在生成ASL姿态序列...")
            pose_sequence = generator.generate_pose(text_prompt)
            
            # 保存姿态序列
            timestamp = datetime.now().strftime('%H%M%S')
            safe_prompt = text_prompt[:30].replace(' ', '_').replace('/', '_')
            
            # 保存为JSON格式
            pose_path = os.path.join(output_dir, f"{timestamp}_{safe_prompt}.json")
            generator.save_pose_sequence(pose_sequence, pose_path, 'json')
            
            # 保存为NumPy格式
            npy_path = os.path.join(output_dir, f"{timestamp}_{safe_prompt}.npy")
            generator.save_pose_sequence(pose_sequence, npy_path, 'npy')
            
            # 可视化
            vis_path = os.path.join(output_dir, f"{timestamp}_{safe_prompt}_visualization.png")
            generator.visualize_pose_sequence(pose_sequence, text_prompt, vis_path)
            
            print(f"✅ 生成完成！文件保存在: {output_dir}")
            print(f"   - 姿态数据: {pose_path}")
            print(f"   - NumPy数据: {npy_path}")
            print(f"   - 可视化: {vis_path}")
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 如果没有命令行参数，启动交互模式
        interactive_generation()
    else:
        # 使用命令行参数
        main() 