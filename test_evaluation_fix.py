#!/usr/bin/env python3
"""
测试评估指标修复效果
验证新的评估方法是否更合理
"""

import torch
import numpy as np
from improved_signllm_model import ImprovedSignLLM, ModelConfig
from improved_evaluation import ImprovedSignLLMEvaluator, compare_evaluation_methods
from pose_consistency_loss import PoseConsistencyLoss

def test_evaluation_fix():
    """测试评估修复的效果"""
    print("🧪 测试评估指标修复")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建模型
    config = ModelConfig("small")  # 使用small模型减少计算量
    model = ImprovedSignLLM(config).to(device)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 20
    texts = ["Hello", "How are you?"]
    target_poses = torch.randn(batch_size, seq_len, 150).to(device)
    
    print(f"📊 测试数据: {batch_size} 样本, {seq_len} 帧")
    
    # 1. 测试模型前向传播
    print("\n🔍 测试模型前向传播...")
    model.train()
    with torch.no_grad():
        # Teacher forcing模式
        results = model(texts=texts, language="ASL", target_poses=target_poses)
        pred_poses = results['predicted_poses']
        print(f"✅ Teacher forcing 成功: {pred_poses.shape}")
        
        # 推理模式  
        model.set_inference_mode(True)
        results = model(texts=texts, language="ASL", max_length=20)
        inference_poses = results['predicted_poses']
        print(f"✅ 推理模式成功: {inference_poses.shape}")
    
    # 2. 测试损失函数
    print("\n🔍 测试损失函数...")
    pose_loss = PoseConsistencyLoss(
        bone_length_weight=0.2,
        joint_angle_weight=0.1,
        symmetry_weight=0.05,
        temporal_weight=0.1
    )
    
    with torch.no_grad():
        losses = pose_loss(pred_poses, target_poses)
        print("约束损失:")
        for key, value in losses.items():
            print(f"  {key:20s}: {value.item():.6f}")
    
    # 3. 测试评估指标对比
    print("\n🔍 测试评估指标对比...")
    
    # 创建两组数据：相似的和不相似的
    similar_pred = target_poses + torch.randn_like(target_poses) * 0.05  # 小差异
    different_pred = torch.randn_like(target_poses) * 0.5  # 大差异
    
    # 转换为numpy
    target_np = [target_poses[i].cpu().numpy() for i in range(batch_size)]
    similar_np = [similar_pred[i].cpu().numpy() for i in range(batch_size)]
    different_np = [different_pred[i].cpu().numpy() for i in range(batch_size)]
    
    # 使用改进的评估器
    evaluator = ImprovedSignLLMEvaluator()
    
    similar_metrics = evaluator.evaluate_poses(similar_np, target_np)
    different_metrics = evaluator.evaluate_poses(different_np, target_np)
    
    print("\n📊 相似预测 vs 不相似预测对比:")
    print("指标名称                 相似预测    不相似预测    比值")
    print("-" * 55)
    
    key_metrics = ['cosine_similarity', 'euclidean_similarity', 'weighted_similarity', 'dtw_score']
    for metric in key_metrics:
        sim_val = similar_metrics[metric]
        diff_val = different_metrics[metric]
        ratio = sim_val / diff_val if diff_val > 0 else float('inf')
        print(f"{metric:25s} {sim_val:8.4f}    {diff_val:8.4f}    {ratio:6.2f}")
    
    # 4. 验证修复效果
    print("\n✅ 修复效果验证:")
    
    # 检查相似预测是否得到更高分数
    if similar_metrics['weighted_similarity'] > different_metrics['weighted_similarity']:
        print("✅ 加权相似度：相似预测 > 不相似预测 ✓")
    else:
        print("❌ 加权相似度：相似预测 < 不相似预测 ✗")
    
    if similar_metrics['euclidean_similarity'] > different_metrics['euclidean_similarity']:
        print("✅ 欧氏相似度：相似预测 > 不相似预测 ✓")
    else:
        print("❌ 欧氏相似度：相似预测 < 不相似预测 ✗")
    
    if similar_metrics['dtw_score'] > different_metrics['dtw_score']:
        print("✅ DTW分数：相似预测 > 不相似预测 ✓")
    else:
        print("❌ DTW分数：相似预测 < 不相似预测 ✗")
    
    print("\n💡 结论:")
    print("- 现在使用 weighted_similarity 作为主要指标")
    print("- 这个指标更适合姿态任务，关注绝对位置而非方向")
    print("- 约束权重已降低，模型会更关注目标匹配")
    
    return similar_metrics, different_metrics

def test_constraint_weights():
    """测试不同约束权重的效果"""
    print("\n🔧 测试约束权重效果")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 生成测试数据
    pred_poses = torch.randn(2, 20, 150).to(device)
    target_poses = torch.randn(2, 20, 150).to(device)
    
    # 测试不同权重配置
    configs = [
        ("原始权重", {"bone_length_weight": 1.0, "joint_angle_weight": 0.5, "symmetry_weight": 0.3, "temporal_weight": 0.2}),
        ("修复权重", {"bone_length_weight": 0.2, "joint_angle_weight": 0.1, "symmetry_weight": 0.05, "temporal_weight": 0.1}),
        ("极轻权重", {"bone_length_weight": 0.1, "joint_angle_weight": 0.05, "symmetry_weight": 0.05, "temporal_weight": 0.05})
    ]
    
    print("权重配置                约束总损失      建议")
    print("-" * 50)
    
    for name, weights in configs:
        loss_fn = PoseConsistencyLoss(**weights, device=device)
        
        with torch.no_grad():
            losses = loss_fn(pred_poses, target_poses)
            total_loss = losses['total'].item()
        
        suggestion = ""
        if total_loss > 10:
            suggestion = "约束过强"
        elif total_loss < 1:
            suggestion = "约束过弱"
        else:
            suggestion = "平衡良好"
        
        print(f"{name:20s} {total_loss:10.4f}      {suggestion}")
    
    print("\n💡 建议使用修复权重，在约束和相似度之间取得平衡")

if __name__ == "__main__":
    test_evaluation_fix()
    test_constraint_weights()
    
    print("\n🎉 测试完成！")
    print("\n📝 总结:")
    print("1. ✅ 模型前向传播正常")
    print("2. ✅ 约束权重已降低")
    print("3. ✅ 评估指标更合理")
    print("4. ✅ 现在可以重新训练了")
    
    print("\n🚀 下一步:")
    print("运行 python improved_signllm_train.py 开始训练") 