#!/usr/bin/env python3
"""
快速训练脚本 - 用于测试ASL text-to-pose系统
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import os
from tqdm import tqdm
from text2video_model import PoseUNet1D, TextToPoseDiffusion
from video_dataset import create_asl_dataloader
import matplotlib.pyplot as plt

def quick_test():
    """快速测试数据加载和模型前向传播"""
    print("🚀 开始快速测试...")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 数据路径
    data_dir = "datasets/signllm_training_data/ASL/dev"
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    print(f"✅ 数据目录存在: {data_dir}")
    
    try:
        # 创建数据加载器（小批次测试）
        print("📊 创建数据加载器...")
        dataloader, dataset = create_asl_dataloader(
            data_dir=data_dir,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # 避免multiprocessing问题
            num_frames=30,  # 较短序列用于测试
            max_samples=5,  # 只加载5个样本
            normalize=True,
            augment=False
        )
        
        print(f"✅ 数据加载成功，数据集大小: {len(dataset)}")
        
        # 测试一个批次
        print("🔍 测试数据批次...")
        for batch in dataloader:
            pose_sequences = batch['pose_sequences']
            captions = batch['captions']
            print(f"   姿态序列形状: {pose_sequences.shape}")
            print(f"   文本数量: {len(captions)}")
            print(f"   示例文本: {captions[0][:50]}...")
            break
        
        # 创建小模型进行测试
        print("🏗️ 创建模型...")
        model = PoseUNet1D(
            pose_dim=120,
            model_channels=64,  # 小一些的模型
            num_res_blocks=1,   # 减少层数
            num_frames=30,
            use_transformer=False  # 先不使用transformer
        ).to(device)
        
        print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建扩散过程
        diffusion = TextToPoseDiffusion(num_timesteps=50)  # 减少时间步
        diffusion.betas = diffusion.betas.to(device)
        diffusion.alphas = diffusion.alphas.to(device)
        diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
        diffusion.alphas_cumprod_prev = diffusion.alphas_cumprod_prev.to(device)
        diffusion.posterior_variance = diffusion.posterior_variance.to(device)
        
        # 测试前向传播
        print("⚡ 测试前向传播...")
        pose_sequences = pose_sequences.to(device)
        timesteps = torch.randint(0, 50, (pose_sequences.shape[0],)).to(device)
        
        with torch.no_grad():
            output = model(pose_sequences, timesteps, captions)
            print(f"✅ 前向传播成功！输出形状: {output.shape}")
        
        # 测试损失计算
        print("📉 测试损失计算...")
        loss = diffusion.p_losses(model, pose_sequences, timesteps, captions)
        print(f"✅ 损失计算成功！损失值: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_train():
    """快速训练几个步骤"""
    print("🏋️ 开始快速训练...")
    
    if not quick_test():
        print("❌ 基础测试失败，无法开始训练")
        return
    
    # 设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "datasets/signllm_training_data/ASL/dev"
    
    # 创建数据加载器
    dataloader, dataset = create_asl_dataloader(
        data_dir=data_dir,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        num_frames=30,
        max_samples=10,  # 只用10个样本训练
        normalize=True,
        augment=False
    )
    
    # 创建模型
    model = PoseUNet1D(
        pose_dim=120,
        model_channels=64,
        num_res_blocks=1,
        num_frames=30,
        use_transformer=False
    ).to(device)
    
    # 创建扩散过程
    diffusion = TextToPoseDiffusion(num_timesteps=50)
    diffusion.betas = diffusion.betas.to(device)
    diffusion.alphas = diffusion.alphas.to(device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    diffusion.alphas_cumprod_prev = diffusion.alphas_cumprod_prev.to(device)
    diffusion.posterior_variance = diffusion.posterior_variance.to(device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    # 训练几个步骤
    model.train()
    losses = []
    
    print("开始训练...")
    for epoch in range(3):  # 只训练3个epoch
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/3')
        
        for batch in pbar:
            pose_sequences = batch['pose_sequences'].to(device)
            captions = batch['captions']
            
            # 随机时间步
            t = torch.randint(0, 50, (pose_sequences.shape[0],)).to(device)
            
            # 计算损失
            loss = diffusion.p_losses(model, pose_sequences, t, captions)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(losses, 'b-o')
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 快速训练完成！")
    print("模型已经成功运行，现在可以开始完整训练。")
    
    # 保存模型
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'pose_dim': 120,
            'model_channels': 64,
            'num_frames': 30,
            'num_timesteps': 50
        }
    }
    torch.save(checkpoint, 'quick_test_model.pth')
    print("模型已保存为 quick_test_model.pth")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        quick_test()
    else:
        quick_train() 