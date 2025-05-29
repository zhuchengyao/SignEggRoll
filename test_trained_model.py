#!/usr/bin/env python3
"""
测试训练好的SignLLM模型
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from signllm_model import SignLLM


def load_trained_model(checkpoint_path):
    """加载训练好的模型"""
    # 创建模型
    model = SignLLM(
        languages=["ASL"],
        gloss_vocab_size=1000,
        hidden_dim=256,
        pose_dim=150
    )
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ 模型加载成功，训练epoch: {checkpoint['epoch']}, 损失: {checkpoint['loss']:.6f}")
    
    return model


def test_model_inference(model, test_texts):
    """测试模型推理"""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    results = []
    
    with torch.no_grad():
        for text in test_texts:
            print(f"\n🔍 测试文本: '{text}'")
            
            # MLSF模式
            poses_mlsf, quality_mlsf = model(
                texts=[text],
                language="ASL",
                mode="mlsf",
                max_length=30
            )
            
            # Prompt2LangGloss模式
            poses_p2lg, gloss_logits, quality_p2lg = model(
                texts=[text],
                language="ASL",
                mode="prompt2langgloss",
                max_pose_length=30,
                max_gloss_length=20
            )
            
            results.append({
                'text': text,
                'mlsf_poses': poses_mlsf.cpu().numpy(),
                'mlsf_quality': quality_mlsf.cpu().numpy(),
                'p2lg_poses': poses_p2lg.cpu().numpy(),
                'p2lg_gloss': gloss_logits.cpu().numpy(),
                'p2lg_quality': quality_p2lg.cpu().numpy()
            })
            
            print(f"  MLSF姿态形状: {poses_mlsf.shape}")
            print(f"  MLSF质量分数: {quality_mlsf.mean().item():.4f}")
            print(f"  P2LG姿态形状: {poses_p2lg.shape}")
            print(f"  P2LG质量分数: {quality_p2lg.mean().item():.4f}")
    
    return results


def visualize_poses(results, save_path="pose_analysis.png"):
    """可视化生成的姿态"""
    fig, axes = plt.subplots(2, len(results), figsize=(5*len(results), 8))
    if len(results) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, result in enumerate(results):
        text = result['text']
        mlsf_poses = result['mlsf_poses'][0]  # [frames, 150]
        p2lg_poses = result['p2lg_poses'][0]  # [frames, 150]
        
        # MLSF模式姿态轨迹
        ax1 = axes[0, i]
        # 显示前10个关键点的轨迹
        for j in range(min(10, mlsf_poses.shape[1]//3)):
            x_coords = mlsf_poses[:, j*3]
            y_coords = mlsf_poses[:, j*3+1]
            ax1.plot(x_coords, y_coords, alpha=0.7, label=f'Point {j}')
        
        ax1.set_title(f'MLSF模式\n"{text[:20]}..."')
        ax1.set_xlabel('X坐标')
        ax1.set_ylabel('Y坐标')
        ax1.grid(True)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Prompt2LangGloss模式姿态轨迹
        ax2 = axes[1, i]
        for j in range(min(10, p2lg_poses.shape[1]//3)):
            x_coords = p2lg_poses[:, j*3]
            y_coords = p2lg_poses[:, j*3+1]
            ax2.plot(x_coords, y_coords, alpha=0.7, label=f'Point {j}')
        
        ax2.set_title(f'Prompt2LangGloss模式\n"{text[:20]}..."')
        ax2.set_xlabel('X坐标')
        ax2.set_ylabel('Y坐标')
        ax2.grid(True)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 姿态可视化已保存: {save_path}")


def analyze_model_performance(results):
    """分析模型性能"""
    print("\n📈 模型性能分析")
    print("=" * 50)
    
    for result in results:
        text = result['text']
        mlsf_poses = result['mlsf_poses'][0]
        p2lg_poses = result['p2lg_poses'][0]
        
        print(f"\n📝 文本: '{text}'")
        
        # 运动幅度分析
        mlsf_motion = np.std(mlsf_poses, axis=0).mean()
        p2lg_motion = np.std(p2lg_poses, axis=0).mean()
        
        print(f"  MLSF运动幅度: {mlsf_motion:.4f}")
        print(f"  P2LG运动幅度: {p2lg_motion:.4f}")
        
        # 平滑度分析
        mlsf_smoothness = np.mean(np.abs(np.diff(mlsf_poses, axis=0)))
        p2lg_smoothness = np.mean(np.abs(np.diff(p2lg_poses, axis=0)))
        
        print(f"  MLSF平滑度: {mlsf_smoothness:.4f}")
        print(f"  P2LG平滑度: {p2lg_smoothness:.4f}")
        
        # 质量分数
        mlsf_quality = result['mlsf_quality'].mean()
        p2lg_quality = result['p2lg_quality'].mean()
        
        print(f"  MLSF质量分数: {mlsf_quality:.4f}")
        print(f"  P2LG质量分数: {p2lg_quality:.4f}")


def main():
    print("🧪 测试训练好的SignLLM模型")
    print("=" * 50)
    
    # 加载最新的模型
    checkpoint_path = "checkpoints/minimal_train/epoch_3.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return
    
    model = load_trained_model(checkpoint_path)
    
    # 测试文本
    test_texts = [
        "Hello world",
        "How are you?",
        "Thank you very much",
        "Good morning"
    ]
    
    print(f"\n🔍 测试 {len(test_texts)} 个文本...")
    
    # 推理测试
    results = test_model_inference(model, test_texts)
    
    # 性能分析
    analyze_model_performance(results)
    
    # 可视化
    print("\n🎨 生成可视化...")
    visualize_poses(results[:2])  # 只可视化前2个结果
    
    print("\n🎉 模型测试完成！")
    print("💡 您的SignLLM模型已经可以将文本转换为手语姿态序列了！")


if __name__ == "__main__":
    main() 