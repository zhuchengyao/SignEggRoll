#!/usr/bin/env python3

# ASL Text-to-Pose最小训练示例
import torch
from text2video_model import PoseUNet1D, TextToPoseDiffusion
from train_text2video import ASLTextToPoseTrainer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建小模型用于快速测试
    model = PoseUNet1D(
        pose_dim=120,
        model_channels=32,
        num_frames=30
    )
    
    diffusion = TextToPoseDiffusion(num_timesteps=100)
    
    # 创建训练器
    trainer = ASLTextToPoseTrainer(
        model=model,
        diffusion=diffusion,
        data_dir="datasets/signllm_training_data/ASL/dev",
        device=device,
        batch_size=2,
        learning_rate=1e-3,
        num_epochs=5,
        num_frames=30,
        use_wandb=False  # 关闭wandb用于测试
    )
    
    # 限制数据集大小用于快速测试
    trainer.dataset.data_paths = trainer.dataset.data_paths[:5]
    trainer.dataset.captions = trainer.dataset.captions[:5]
    
    print(f"开始训练 (数据集大小: {len(trainer.dataset)})")
    trainer.train()

if __name__ == "__main__":
    main()
