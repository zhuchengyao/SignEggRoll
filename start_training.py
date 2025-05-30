#!/usr/bin/env python3
"""
ASL Text-to-Pose训练启动脚本
一键启动完整训练流程
"""

import argparse
import os
import torch

def main():
    print("🚀 ASL Text-to-Pose训练启动器")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description='ASL Text-to-Pose训练启动器')
    parser.add_argument('--mode', type=str, choices=['test', 'small', 'full'], default='small',
                       help='训练模式: test(测试), small(小规模), full(完整)')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--no_wandb', action='store_true', help='禁用wandb')
    
    args = parser.parse_args()
    
    # 设备检查
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 使用设备: {device}")
    if device == "cpu":
        print("⚠️  警告: 使用CPU训练会非常慢，建议使用GPU")
    
    # 数据目录
    data_dir = "datasets/signllm_training_data/ASL/dev"
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请确保ASL数据在正确位置")
        return
    
    print(f"✅ 数据目录: {data_dir}")
    
    # 根据模式设置参数
    if args.mode == 'test':
        print("🧪 测试模式 - 快速验证")
        epochs = args.epochs or 5
        batch_size = args.batch_size or 2
        max_samples = 10
        model_channels = 64
        num_timesteps = 100
        
    elif args.mode == 'small':
        print("🏃 小规模训练 - 适合调试和验证")
        epochs = args.epochs or 50
        batch_size = args.batch_size or 4
        max_samples = 1000
        model_channels = 128
        num_timesteps = 500
        
    elif args.mode == 'full':
        print("🚀 完整训练 - 使用全部数据")
        epochs = args.epochs or 1000
        batch_size = args.batch_size or 8
        max_samples = None
        model_channels = 256
        num_timesteps = 1000
    
    print(f"📊 训练配置:")
    print(f"   轮数: {epochs}")
    print(f"   批次大小: {batch_size}")
    print(f"   学习率: {args.lr}")
    print(f"   模型通道: {model_channels}")
    print(f"   扩散步数: {num_timesteps}")
    print(f"   最大样本: {max_samples or '全部'}")
    
    # 构建训练命令
    cmd_parts = [
        "python train_text2video.py",
        f"--data_dir {data_dir}",
        f"--batch_size {batch_size}",
        f"--learning_rate {args.lr}",
        f"--num_epochs {epochs}",
        f"--model_channels {model_channels}",
        f"--num_timesteps {num_timesteps}",
        f"--device {device}"
    ]
    
    if max_samples:
        cmd_parts.append(f"--max_samples {max_samples}")
    
    if args.no_wandb:
        cmd_parts.append("--no_wandb")
    
    command = " ".join(cmd_parts)
    
    print(f"\n🔧 执行命令:")
    print(f"   {command}")
    
    # 确认启动
    confirm = input("\n继续启动训练? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 训练已取消")
        return
    
    print("\n🚀 启动训练...")
    print("=" * 50)
    
    # 执行训练
    import subprocess
    try:
        subprocess.run(command, shell=True, check=True)
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练出错: {e}")
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")

if __name__ == "__main__":
    main() 