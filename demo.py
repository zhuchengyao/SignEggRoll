#!/usr/bin/env python3
"""
Diffusion Model for 3D Pose Generation - 快速演示脚本

这个脚本演示如何：
1. 加载和处理姿态数据
2. 训练一个简单的diffusion model
3. 生成新的姿态样本

使用方法:
python demo.py --data_dir ./datasets/processed --quick_test
"""

import os
import sys
import argparse
import torch
import numpy as np
from pose_dataset import PoseDataset, visualize_pose_data
from diffusion_model import UNet1D, GaussianDiffusion
from train_diffusion import DiffusionTrainer
from generate_poses import PoseGenerator

def check_data_directory(data_dir):
    """检查数据目录是否存在和有效"""
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    # 查找.skels文件
    import glob
    skels_files = glob.glob(os.path.join(data_dir, "**", "*.skels"), recursive=True)
    
    if len(skels_files) == 0:
        print(f"❌ 在目录 {data_dir} 中未找到 .skels 文件")
        print("请确保数据目录包含 .skels 格式的姿态数据文件")
        return False
    
    print(f"✅ 找到 {len(skels_files)} 个 .skels 数据文件")
    return True

def test_data_loading(data_dir, max_files=2):
    """测试数据加载"""
    print("\n🔍 测试数据加载...")
    
    try:
        # 创建数据集（限制文件数量用于快速测试）
        dataset = PoseDataset(
            data_dir=data_dir, 
            max_files=max_files,
            normalize=True,
            augment=False
        )
        
        if len(dataset) == 0:
            print("❌ 数据集为空")
            return None
        
        print(f"✅ 成功加载 {len(dataset)} 个姿态帧")
        
        # 测试单个样本
        sample = dataset[0]
        print(f"✅ 样本形状: {sample.shape}")
        print(f"✅ 数值范围: [{sample.min():.4f}, {sample.max():.4f}]")
        
        # 可视化几个样本
        print("📊 生成数据可视化...")
        visualize_pose_data(dataset, num_samples=min(3, len(dataset)))
        
        return dataset
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def quick_train_test(dataset, device='cuda', epochs=50):
    """快速训练测试"""
    print(f"\n🚀 开始快速训练测试 ({epochs} epochs)...")
    
    # 创建小型模型用于快速测试
    model = UNet1D(
        model_channels=64,  # 减小模型以加快训练
        num_keypoints=67,
        channel_mult=(1, 2, 4)  # 减少层数
    )
    
    # 创建扩散过程
    diffusion = GaussianDiffusion(
        num_timesteps=500,  # 减少时间步以加快训练
        beta_schedule='cosine'
    )
    
    # 移动diffusion参数到设备
    diffusion.betas = diffusion.betas.to(device)
    diffusion.alphas = diffusion.alphas.to(device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    diffusion.alphas_cumprod_prev = diffusion.alphas_cumprod_prev.to(device)
    diffusion.posterior_variance = diffusion.posterior_variance.to(device)
    
    # 创建临时数据目录
    temp_data_dir = "temp_demo_data"
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # 保存数据集的一小部分用于训练
    sample_data_path = os.path.join(temp_data_dir, "demo_samples.skels")
    with open(sample_data_path, 'w') as f:
        for i in range(min(100, len(dataset))):  # 最多100个样本
            pose = dataset.poses[i]  # 获取原始未标准化的数据
            flattened = pose.flatten()
            line = ' '.join([f'{x:.6f}' for x in flattened])
            f.write(line + '\n')
    
    print(f"💾 保存演示数据到: {sample_data_path}")
    
    try:
        # 创建训练器
        trainer = DiffusionTrainer(
            model=model,
            diffusion=diffusion,
            data_dir=temp_data_dir,
            device=device,
            batch_size=8,  # 小批次
            learning_rate=2e-4,
            num_epochs=epochs,
            save_interval=25,
            sample_interval=25,
            use_wandb=False  # 关闭wandb以简化
        )
        
        print("✅ 训练器创建成功")
        
        # 快速训练
        trainer.train()
        
        print("✅ 训练完成!")
        return trainer.save_dir
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return None
    
    finally:
        # 清理临时文件
        if os.path.exists(sample_data_path):
            os.remove(sample_data_path)
        if os.path.exists(temp_data_dir):
            os.rmdir(temp_data_dir)

def test_generation(checkpoint_dir, device='cuda'):
    """测试姿态生成"""
    print("\n🎨 测试姿态生成...")
    
    # 查找最佳模型检查点
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    latest_model_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    
    checkpoint_path = best_model_path if os.path.exists(best_model_path) else latest_model_path
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 找不到模型检查点: {checkpoint_path}")
        return False
    
    try:
        # 创建生成器
        generator = PoseGenerator(checkpoint_path, device)
        print("✅ 生成器创建成功")
        
        # 生成几个样本
        print("🎯 生成姿态样本...")
        poses = generator.generate_poses(num_samples=4)
        print(f"✅ 成功生成 {len(poses)} 个姿态样本")
        
        # 可视化结果
        generator.visualize_poses(poses, "demo_generated_poses.png", "演示：生成的姿态")
        
        # 保存结果
        generator.save_poses(poses, "demo_generated_poses.skels", "skels")
        
        print("✅ 生成测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 生成测试失败: {e}")
        return False

def print_usage_instructions():
    """打印使用说明"""
    print("\n" + "="*60)
    print("🎉 演示完成! 下面是完整的使用说明:")
    print("="*60)
    
    print("\n📁 1. 数据准备:")
    print("   将你的 .skels 文件放在数据目录中")
    print("   每行应包含 67*3=201 个浮点数（67个关键点的x,y,z坐标）")
    
    print("\n🚀 2. 训练模型:")
    print("   python train_diffusion.py --data_dir ./datasets/processed \\")
    print("                            --batch_size 32 \\")
    print("                            --num_epochs 1000 \\")
    print("                            --learning_rate 1e-4")
    
    print("\n🎨 3. 生成姿态:")
    print("   python generate_poses.py --checkpoint ./checkpoints/best_model.pth \\")
    print("                            --num_samples 8 \\")
    print("                            --visualize")
    
    print("\n🎬 4. 生成动画:")
    print("   python generate_poses.py --checkpoint ./checkpoints/best_model.pth \\")
    print("                            --animation \\")
    print("                            --num_frames 30 \\")
    print("                            --visualize")
    
    print("\n📊 5. 数据分析:")
    print("   python pose_dataset.py  # 测试数据加载和可视化")
    
    print("\n💡 提示:")
    print("   - 使用 --no_wandb 禁用 wandb 日志记录")
    print("   - 使用 --device cpu 在CPU上运行")
    print("   - 使用 --max_files N 限制加载的文件数量（调试用）")
    
    print("\n📦 项目结构:")
    print("   diffusion_model.py    - 扩散模型核心实现")
    print("   pose_dataset.py       - 数据加载和预处理")
    print("   train_diffusion.py    - 训练脚本")
    print("   generate_poses.py     - 生成脚本")
    print("   demo.py              - 演示脚本")
    print("   requirements.txt     - 依赖包列表")

def main():
    parser = argparse.ArgumentParser(description='Diffusion Model 3D姿态生成 - 演示脚本')
    parser.add_argument('--data_dir', type=str, default='./datasets/processed', 
                       help='数据目录路径')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--quick_test', action='store_true', 
                       help='执行快速训练测试')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='快速测试的训练轮数')
    parser.add_argument('--skip_training', action='store_true', 
                       help='跳过训练，只测试数据加载')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    print("🤖 Diffusion Model 3D姿态生成 - 演示")
    print(f"设备: {args.device}")
    print(f"数据目录: {args.data_dir}")
    
    # 1. 检查数据目录
    if not check_data_directory(args.data_dir):
        print("\n💡 建议:")
        print("   1. 确保数据目录存在")
        print("   2. 确保目录中有 .skels 格式的文件")
        print("   3. 每个 .skels 文件每行应有 201 个浮点数")
        return
    
    # 2. 测试数据加载
    dataset = test_data_loading(args.data_dir, max_files=2)
    if dataset is None:
        return
    
    # 3. 如果不跳过训练，进行快速训练测试
    checkpoint_dir = None
    if not args.skip_training and args.quick_test:
        checkpoint_dir = quick_train_test(dataset, args.device, args.epochs)
        
        # 4. 测试生成
        if checkpoint_dir:
            test_generation(checkpoint_dir, args.device)
    
    # 5. 打印使用说明
    print_usage_instructions()

if __name__ == "__main__":
    main() 