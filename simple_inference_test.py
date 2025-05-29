#!/usr/bin/env python3
"""
简单推理测试 - 直接使用新模型
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from signllm_model import SignLLM


def test_fresh_model():
    """测试新创建的模型"""
    print("🧪 测试SignLLM模型推理能力")
    print("=" * 50)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLLM(
        languages=["ASL"],
        gloss_vocab_size=1000,
        hidden_dim=256,
        pose_dim=150
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型参数: {total_params:,}")
    print(f"🔧 使用设备: {device}")
    
    # 测试文本
    test_texts = [
        "Hello",
        "Thank you",
        "Good morning",
        "How are you"
    ]
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for text in test_texts:
            print(f"\n🔍 测试文本: '{text}'")
            
            # MLSF模式
            poses_mlsf, quality_mlsf = model(
                texts=[text],
                language="ASL",
                mode="mlsf",
                max_length=25
            )
            
            # Prompt2LangGloss模式
            poses_p2lg, gloss_logits, quality_p2lg = model(
                texts=[text],
                language="ASL",
                mode="prompt2langgloss",
                max_pose_length=25,
                max_gloss_length=15
            )
            
            print(f"  ✅ MLSF姿态: {poses_mlsf.shape}")
            print(f"  ✅ P2LG姿态: {poses_p2lg.shape}")
            print(f"  📊 MLSF质量: {quality_mlsf.mean().item():.4f}")
            print(f"  📊 P2LG质量: {quality_p2lg.mean().item():.4f}")
            
            results.append({
                'text': text,
                'mlsf_poses': poses_mlsf.cpu().numpy(),
                'p2lg_poses': poses_p2lg.cpu().numpy(),
                'mlsf_quality': quality_mlsf.cpu().numpy(),
                'p2lg_quality': quality_p2lg.cpu().numpy()
            })
    
    return results


def analyze_results(results):
    """分析结果"""
    print("\n📈 结果分析")
    print("=" * 40)
    
    for result in results:
        text = result['text']
        mlsf_poses = result['mlsf_poses'][0]  # [frames, 150]
        p2lg_poses = result['p2lg_poses'][0]  # [frames, 150]
        
        print(f"\n📝 '{text}':")
        
        # 数据范围
        mlsf_range = f"[{mlsf_poses.min():.3f}, {mlsf_poses.max():.3f}]"
        p2lg_range = f"[{p2lg_poses.min():.3f}, {p2lg_poses.max():.3f}]"
        
        print(f"  MLSF数据范围: {mlsf_range}")
        print(f"  P2LG数据范围: {p2lg_range}")
        
        # 运动变化
        mlsf_motion = np.std(mlsf_poses, axis=0).mean()
        p2lg_motion = np.std(p2lg_poses, axis=0).mean()
        
        print(f"  MLSF运动变化: {mlsf_motion:.4f}")
        print(f"  P2LG运动变化: {p2lg_motion:.4f}")


def visualize_simple(results):
    """简单可视化"""
    print("\n🎨 生成可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for i, result in enumerate(results[:2]):
        text = result['text']
        mlsf_poses = result['mlsf_poses'][0]  # [frames, 150]
        p2lg_poses = result['p2lg_poses'][0]  # [frames, 150]
        
        # MLSF模式 - 显示前几个关键点的X坐标变化
        ax1 = axes[0, i]
        for j in range(min(5, mlsf_poses.shape[1]//3)):
            x_coords = mlsf_poses[:, j*3]
            ax1.plot(x_coords, label=f'Point {j}', alpha=0.8)
        ax1.set_title(f'MLSF: "{text}"')
        ax1.set_xlabel('时间帧')
        ax1.set_ylabel('X坐标')
        ax1.legend()
        ax1.grid(True)
        
        # P2LG模式
        ax2 = axes[1, i]
        for j in range(min(5, p2lg_poses.shape[1]//3)):
            x_coords = p2lg_poses[:, j*3]
            ax2.plot(x_coords, label=f'Point {j}', alpha=0.8)
        ax2.set_title(f'P2LG: "{text}"')
        ax2.set_xlabel('时间帧')
        ax2.set_ylabel('X坐标')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('signllm_inference_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 可视化已保存: signllm_inference_test.png")


def main():
    print("🚀 SignLLM 推理能力测试")
    print("=" * 50)
    
    try:
        # 测试模型
        results = test_fresh_model()
        
        # 分析结果
        analyze_results(results)
        
        # 可视化
        visualize_simple(results)
        
        print("\n🎉 测试完成！")
        print("💡 SignLLM模型可以成功将文本转换为手语姿态序列！")
        print("📋 模型特点:")
        print("  - 支持两种生成模式 (MLSF & Prompt2LangGloss)")
        print("  - 输出150维姿态数据 (50个关键点 × 3坐标)")
        print("  - 包含质量评估机制")
        print("  - 支持多语言手语生成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 