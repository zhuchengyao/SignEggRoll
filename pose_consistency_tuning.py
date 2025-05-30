#!/usr/bin/env python3
"""
姿态约束权重调优脚本
帮助找到最佳的约束权重配置，平衡约束效果和姿态相似度
"""

import torch
import numpy as np
from pose_consistency_loss import PoseConsistencyLoss
from improved_evaluation import ImprovedSignLLMEvaluator
from typing import Dict, List, Tuple


def generate_realistic_test_poses(batch_size=4, seq_len=20, pose_dim=150, 
                                pose_type="normal", add_noise=0.05):
    """生成更真实的测试姿态数据"""
    if pose_type == "normal":
        # 生成较为自然的姿态序列
        poses = torch.zeros(batch_size, seq_len, pose_dim)
        for b in range(batch_size):
            # 生成基础姿态
            base_pose = torch.randn(pose_dim) * 0.1
            for t in range(seq_len):
                # 添加时间变化和噪声
                temporal_variation = torch.sin(torch.tensor(t * 0.1)) * 0.02
                noise = torch.randn(pose_dim) * add_noise
                poses[b, t] = base_pose + temporal_variation + noise
                
    elif pose_type == "target":
        # 生成目标姿态（稍微不同但合理）
        poses = generate_realistic_test_poses(batch_size, seq_len, pose_dim, "normal", add_noise)
        # 添加小的系统性差异
        poses += torch.randn_like(poses) * 0.02
        
    return poses


def test_constraint_weights(weight_configs: List[Dict[str, float]], 
                          test_data: Tuple[torch.Tensor, torch.Tensor],
                          device: str = "auto") -> Dict[str, Dict[str, float]]:
    """测试不同的约束权重配置"""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pred_poses, target_poses = test_data
    pred_poses = pred_poses.to(device)
    target_poses = target_poses.to(device)
    
    evaluator = ImprovedSignLLMEvaluator()
    results = {}
    
    print("🧪 测试不同的约束权重配置")
    print("=" * 70)
    
    for i, config in enumerate(weight_configs):
        config_name = f"Config_{i+1}"
        print(f"\n🔍 {config_name}: {config}")
        
        # 创建损失函数
        consistency_loss = PoseConsistencyLoss(
            bone_length_weight=config.get('bone_length', 1.0),
            joint_angle_weight=config.get('joint_angle', 0.5),
            symmetry_weight=config.get('symmetry', 0.3),
            temporal_weight=config.get('temporal', 0.2),
            device=device
        )
        
        # 计算约束损失
        with torch.no_grad():
            constraint_losses = consistency_loss(pred_poses, target_poses)
        
        # 计算评估指标
        pred_np = [p.cpu().numpy() for p in pred_poses]
        target_np = [t.cpu().numpy() for t in target_poses]
        eval_metrics = evaluator.evaluate_poses(pred_np, target_np)
        
        # 合并结果
        combined_metrics = {
            # 约束损失
            'constraint_total': constraint_losses['total'].item(),
            'constraint_bone': constraint_losses['bone_length'].item(),
            'constraint_angle': constraint_losses['joint_angle'].item(),
            'constraint_symmetry': constraint_losses['symmetry'].item(),
            'constraint_temporal': constraint_losses['temporal'].item(),
            'constraint_supervised': constraint_losses['supervised'].item(),
            
            # 评估指标
            'cosine_similarity': eval_metrics['cosine_similarity'],
            'euclidean_similarity': eval_metrics['euclidean_similarity'],
            'weighted_similarity': eval_metrics['weighted_similarity'],
            'dtw_score': eval_metrics['dtw_score'],
            'motion_smoothness': eval_metrics['motion_smoothness'],
        }
        
        results[config_name] = combined_metrics
        
        # 打印关键指标
        print(f"  约束总损失: {combined_metrics['constraint_total']:.4f}")
        print(f"  加权相似度: {combined_metrics['weighted_similarity']:.4f}")
        print(f"  欧氏相似度: {combined_metrics['euclidean_similarity']:.4f}")
        print(f"  DTW分数:   {combined_metrics['dtw_score']:.4f}")
    
    return results


def analyze_results(results: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """分析测试结果，给出建议"""
    
    print("\n📊 结果分析")
    print("=" * 70)
    
    # 计算各指标的最佳配置
    metrics_to_maximize = [
        'weighted_similarity', 'euclidean_similarity', 
        'dtw_score', 'motion_smoothness'
    ]
    metrics_to_minimize = ['constraint_total']
    
    best_configs = {}
    
    for metric in metrics_to_maximize:
        values = {config: metrics[metric] for config, metrics in results.items()}
        best_config = max(values, key=values.get)
        best_configs[metric] = (best_config, values[best_config])
    
    for metric in metrics_to_minimize:
        values = {config: metrics[metric] for config, metrics in results.items()}
        best_config = min(values, key=values.get)
        best_configs[metric] = (best_config, values[best_config])
    
    print("\n🏆 各指标最佳配置:")
    for metric, (config, value) in best_configs.items():
        print(f"  {metric:20s}: {config} ({value:.4f})")
    
    # 计算综合评分
    print("\n📈 综合评分 (加权相似度 + DTW分数 - 约束损失/10):")
    composite_scores = {}
    for config, metrics in results.items():
        score = (metrics['weighted_similarity'] + 
                metrics['dtw_score'] - 
                metrics['constraint_total'] / 10)
        composite_scores[config] = score
        print(f"  {config}: {score:.4f}")
    
    best_overall = max(composite_scores, key=composite_scores.get)
    print(f"\n🎯 综合最佳配置: {best_overall}")
    
    return best_configs


def recommend_weights() -> List[Dict[str, float]]:
    """推荐的权重配置组合"""
    return [
        # 原始配置（偏重约束）
        {
            'bone_length': 1.0,
            'joint_angle': 0.5,
            'symmetry': 0.3,
            'temporal': 0.2
        },
        
        # 轻量约束
        {
            'bone_length': 0.3,
            'joint_angle': 0.2,
            'symmetry': 0.1,
            'temporal': 0.1
        },
        
        # 平衡配置
        {
            'bone_length': 0.5,
            'joint_angle': 0.3,
            'symmetry': 0.2,
            'temporal': 0.1
        },
        
        # 只关注重要约束
        {
            'bone_length': 0.8,
            'joint_angle': 0.1,
            'symmetry': 0.1,
            'temporal': 0.05
        },
        
        # 时间优先
        {
            'bone_length': 0.2,
            'joint_angle': 0.1,
            'symmetry': 0.1,
            'temporal': 0.4
        },
        
        # 极轻约束
        {
            'bone_length': 0.1,
            'joint_angle': 0.05,
            'symmetry': 0.05,
            'temporal': 0.05
        }
    ]


def main():
    """主函数"""
    print("🎛️  姿态约束权重调优")
    print("=" * 70)
    
    # 生成测试数据
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    pred_poses = generate_realistic_test_poses(4, 20, 150, "normal")
    target_poses = generate_realistic_test_poses(4, 20, 150, "target")
    test_data = (pred_poses, target_poses)
    
    # 获取推荐配置
    weight_configs = recommend_weights()
    
    # 测试权重配置
    results = test_constraint_weights(weight_configs, test_data, device)
    
    # 分析结果
    best_configs = analyze_results(results)
    
    # 生成建议
    print("\n💡 建议:")
    print("=" * 70)
    print("1. 如果姿态相似度很重要，选择轻量约束或极轻约束配置")
    print("2. 如果需要物理合理性，选择平衡配置")
    print("3. 如果动作平滑度重要，选择时间优先配置")
    print("4. 可以根据验证集表现进一步微调权重")
    
    print("\n🔧 推荐的训练脚本修改:")
    print("```python")
    print("# 在 improved_signllm_train.py 中")
    print("self.pose_consistency = PoseConsistencyLoss(")
    print("    bone_length_weight=0.3,  # 降低骨骼约束")
    print("    joint_angle_weight=0.2,  # 降低角度约束")
    print("    symmetry_weight=0.1,     # 降低对称约束")
    print("    temporal_weight=0.1      # 降低时间约束")
    print(")")
    print("```")


if __name__ == "__main__":
    main() 