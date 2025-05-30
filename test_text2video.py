#!/usr/bin/env python3
"""
ASL Text-to-Pose项目快速测试脚本
用于验证整个流程是否正常工作
"""

import os
import sys
import torch
import subprocess
import shutil
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    print("📦 检查依赖...")
    
    required_packages = [
        'torch', 'torchvision', 'transformers', 
        'numpy', 'matplotlib', 'json'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'json':
                import json
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下依赖包：{missing_packages}")
        print("请运行: pip install -r requirements_text2video.txt")
        return False
    
    print("✅ 所有依赖检查通过")
    return True

def check_asl_data():
    """检查ASL数据格式"""
    print("\n🔍 检查ASL数据格式...")
    
    # 查找ASL数据目录
    possible_dirs = [
        "datasets/signllm_training_data/ASL/dev",
        "./datasets/signllm_training_data/ASL/dev", 
        "./asl_data",
        "./datasets",
        "./data",
        "."
    ]
    
    asl_dirs = []
    for base_dir in possible_dirs:
        if os.path.exists(base_dir):
            # 如果是signllm数据目录，直接检查是否有dev_*子目录
            if "signllm_training_data" in base_dir:
                dev_dirs = [d for d in os.listdir(base_dir) if d.startswith('dev_')]
                if dev_dirs:
                    # 检查前几个目录是否有pose.json和text.txt
                    for dev_dir in dev_dirs[:3]:
                        dev_path = os.path.join(base_dir, dev_dir)
                        pose_file = os.path.join(dev_path, 'pose.json')
                        text_file = os.path.join(dev_path, 'text.txt')
                        if os.path.exists(pose_file) and os.path.exists(text_file):
                            asl_dirs.append(dev_path)
                    if asl_dirs:
                        print(f"✅ 找到 {len(dev_dirs)} 个ASL数据目录")
                        return base_dir  # 返回父目录
            else:
                # 原有的检查逻辑
                for item in os.listdir(base_dir):
                    item_path = os.path.join(base_dir, item)
                    if os.path.isdir(item_path) and item.startswith('dev_'):
                        pose_file = os.path.join(item_path, 'pose.json')
                        text_file = os.path.join(item_path, 'text.txt')
                        if os.path.exists(pose_file) and os.path.exists(text_file):
                            asl_dirs.append(item_path)
                
                if asl_dirs:
                    print(f"✅ 找到 {len(asl_dirs)} 个ASL数据目录")
                    return base_dir  # 返回父目录
    
    print("❌ 未找到ASL数据")
    print("请确保数据目录包含dev_*子目录，每个子目录有pose.json和text.txt文件")
    return None

def test_data_loading():
    """测试数据加载"""
    print("\n📊 测试数据加载...")
    
    try:
        from video_dataset import ASLTextPoseDataset
        
        # 查找数据目录
        data_dir = check_asl_data()
        if not data_dir:
            return False
        
        # 创建数据集
        dataset = ASLTextPoseDataset(
            data_dir=data_dir,
            num_frames=50,  # 减少帧数用于测试
            max_samples=5   # 限制样本数量
        )
        
        if len(dataset) == 0:
            print("❌ 数据集为空")
            return False
        
        # 测试加载一个样本
        sample = dataset[0]
        pose_sequence = sample['pose_sequence']
        caption = sample['caption']
        
        print(f"✅ 数据加载成功")
        print(f"   样本数量: {len(dataset)}")
        print(f"   姿态形状: {pose_sequence.shape}")
        print(f"   示例文本: {caption}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n🏗️ 测试模型创建...")
    
    try:
        from text2video_model import PoseUNet1D, TextToPoseDiffusion
        
        # 创建模型
        model = PoseUNet1D(
            pose_dim=120,
            model_channels=64,  # 减少通道数用于测试
            num_frames=50
        )
        
        # 创建扩散过程
        diffusion = TextToPoseDiffusion(
            num_timesteps=100,  # 减少步数用于测试
            beta_schedule='cosine'
        )
        
        print(f"✅ 模型创建成功")
        print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   扩散步数: {diffusion.num_timesteps}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

def test_forward_pass():
    """测试前向传播"""
    print("\n⚡ 测试前向传播...")
    
    try:
        from text2video_model import PoseUNet1D, TextToPoseDiffusion
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   使用设备: {device}")
        
        # 创建模型
        model = PoseUNet1D(
            pose_dim=120,
            model_channels=64,
            num_frames=50
        ).to(device)
        
        # 创建扩散过程
        diffusion = TextToPoseDiffusion(num_timesteps=100)
        diffusion.betas = diffusion.betas.to(device)
        diffusion.alphas = diffusion.alphas.to(device)
        diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
        diffusion.alphas_cumprod_prev = diffusion.alphas_cumprod_prev.to(device)
        diffusion.posterior_variance = diffusion.posterior_variance.to(device)
        
        # 创建测试输入
        batch_size = 2
        pose_sequence = torch.randn(batch_size, 50, 120).to(device)
        timesteps = torch.randint(0, 100, (batch_size,)).to(device)
        text_prompts = ["hello", "thank you"]
        
        # 前向传播
        with torch.no_grad():
            output = model(pose_sequence, timesteps, text_prompts)
        
        print(f"✅ 前向传播成功")
        print(f"   输入形状: {pose_sequence.shape}")
        print(f"   输出形状: {output.shape}")
        
        # 测试损失计算
        loss = diffusion.p_losses(model, pose_sequence, timesteps, text_prompts)
        print(f"   损失计算成功，损失值: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False

def test_training_step():
    """测试训练步骤"""
    print("\n🏋️ 测试训练步骤...")
    
    try:
        from train_text2video import ASLTextToPoseTrainer
        from text2video_model import PoseUNet1D, TextToPoseDiffusion
        
        # 查找数据目录
        data_dir = check_asl_data()
        if not data_dir:
            return False
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        model = PoseUNet1D(
            pose_dim=120,
            model_channels=32,  # 更小的模型用于测试
            num_frames=20       # 更短的序列
        )
        
        diffusion = TextToPoseDiffusion(num_timesteps=50)
        
        # 创建训练器
        trainer = ASLTextToPoseTrainer(
            model=model,
            diffusion=diffusion,
            data_dir=data_dir,
            device=device,
            batch_size=2,
            learning_rate=1e-3,
            num_epochs=2,
            num_frames=20,
            use_wandb=False
        )
        
        # 限制数据集大小
        trainer.dataset.data_paths = trainer.dataset.data_paths[:2]
        trainer.dataset.captions = trainer.dataset.captions[:2]
        
        # 测试一个训练步骤
        for batch in trainer.dataloader:
            loss = trainer.train_step(batch)
            print(f"✅ 训练步骤成功，损失: {loss:.6f}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ 训练步骤失败: {e}")
        return False

def test_generation():
    """测试生成"""
    print("\n🎨 测试姿态生成...")
    
    try:
        from text2video_model import PoseUNet1D, TextToPoseDiffusion
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        model = PoseUNet1D(
            pose_dim=120,
            model_channels=32,
            num_frames=20
        ).to(device)
        
        diffusion = TextToPoseDiffusion(num_timesteps=20)  # 很少的步数用于快速测试
        diffusion.betas = diffusion.betas.to(device)
        diffusion.alphas = diffusion.alphas.to(device)
        diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
        diffusion.alphas_cumprod_prev = diffusion.alphas_cumprod_prev.to(device)
        diffusion.posterior_variance = diffusion.posterior_variance.to(device)
        
        # 生成姿态序列
        with torch.no_grad():
            pose_sequences = diffusion.sample(
                model,
                text_prompts=["hello"],
                num_frames=20,
                pose_dim=120
            )
        
        final_pose = pose_sequences[-1][0].cpu().numpy()  # (T, 120)
        
        print(f"✅ 姿态生成成功")
        print(f"   生成形状: {final_pose.shape}")
        print(f"   数值范围: [{final_pose.min():.3f}, {final_pose.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 姿态生成失败: {e}")
        return False

def test_pose_visualization():
    """测试姿态可视化"""
    print("\n📈 测试姿态可视化...")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 创建测试姿态数据
        test_pose = np.random.randn(50, 120) * 10  # (T, 120)
        
        from generate_text2video import ASLTextToPoseGenerator
        
        # 创建一个临时生成器实例（不需要加载真实模型）
        class MockGenerator:
            def visualize_pose_sequence(self, pose_array, text_prompt, save_path=None):
                # 简化的可视化测试
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.imshow(pose_array.T, aspect='auto', cmap='viridis')
                ax.set_title(f"测试姿态: {text_prompt}")
                ax.set_xlabel('帧数')
                ax.set_ylabel('特征维度')
                
                if save_path:
                    plt.savefig(save_path)
                    print(f"   可视化保存到: {save_path}")
                
                plt.close()
                return True
        
        generator = MockGenerator()
        test_output = "./test_visualization.png"
        generator.visualize_pose_sequence(test_pose, "test pose", test_output)
        
        # 清理测试文件
        if os.path.exists(test_output):
            os.remove(test_output)
        
        print(f"✅ 姿态可视化成功")
        return True
        
    except Exception as e:
        print(f"❌ 姿态可视化失败: {e}")
        return False

def create_minimal_training_example():
    """创建最小训练示例"""
    print("\n📋 创建最小训练示例...")
    
    # 查找数据目录
    data_dir = check_asl_data()
    if not data_dir:
        print("❌ 需要ASL数据才能创建训练示例")
        return False
    
    script_content = f'''#!/usr/bin/env python3

# ASL Text-to-Pose最小训练示例
import torch
from text2video_model import PoseUNet1D, TextToPoseDiffusion
from train_text2video import ASLTextToPoseTrainer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {{device}}")
    
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
        data_dir="{data_dir}",
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
    
    print(f"开始训练 (数据集大小: {{len(trainer.dataset)}})")
    trainer.train()

if __name__ == "__main__":
    main()
'''
    
    with open("minimal_train_example.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ 最小训练示例已创建: minimal_train_example.py")
    print("   运行命令: python minimal_train_example.py")
    return True

def run_full_test():
    """运行完整测试流程"""
    print("🚀 开始ASL Text-to-Pose项目完整测试\n")
    
    tests = [
        ("依赖检查", check_dependencies),
        ("ASL数据检查", lambda: check_asl_data() is not None),
        ("数据加载测试", test_data_loading),
        ("模型创建测试", test_model_creation),
        ("前向传播测试", test_forward_pass),
        ("训练步骤测试", test_training_step),
        ("姿态生成测试", test_generation),
        ("可视化测试", test_pose_visualization),
        ("创建训练示例", create_minimal_training_example)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"⚠️ {test_name} 跳过或失败")
        except Exception as e:
            print(f"❌ {test_name} 出现异常: {e}")
    
    print(f"\n📊 测试总结:")
    print(f"   通过: {passed}/{total}")
    print(f"   成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 所有测试通过！ASL Text-to-Pose项目设置正确。")
        print("\n📝 下一步:")
        print("   1. 运行 python minimal_train_example.py 开始训练")
        print("   2. 使用完整数据集进行正式训练")
        print("   3. 调整超参数优化效果")
    else:
        print(f"\n⚠️ 有 {total-passed} 个测试失败，请检查相关组件。")
    
    return passed == total

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        test_map = {
            "deps": check_dependencies,
            "data": test_data_loading,
            "model": test_model_creation,
            "forward": test_forward_pass,
            "train": test_training_step,
            "generate": test_generation,
            "visualize": test_pose_visualization
        }
        
        if test_name in test_map:
            print(f"运行单项测试: {test_name}")
            success = test_map[test_name]()
            sys.exit(0 if success else 1)
        else:
            print(f"未知测试: {test_name}")
            print(f"可用测试: {list(test_map.keys())}")
            sys.exit(1)
    else:
        # 运行完整测试
        success = run_full_test()
        sys.exit(0 if success else 1) 