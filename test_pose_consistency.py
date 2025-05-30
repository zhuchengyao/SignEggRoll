#!/usr/bin/env python3
"""
测试 PoseConsistencyLoss 的效果
比较有/无姿态约束的损失值
"""

import torch
import numpy as np
from pose_consistency_loss import PoseConsistencyLoss
import matplotlib.pyplot as plt

def generate_test_poses(batch_size=2, seq_len=10, pose_dim=150, pose_type="normal"):
    """生成测试姿态数据"""
    if pose_type == "normal":
        # 正常的姿态数据
        poses = torch.randn(batch_size, seq_len, pose_dim) * 0.1
    elif pose_type == "abnormal":
        # 异常的姿态数据（违反物理约束）
        poses = torch.randn(batch_size, seq_len, pose_dim) * 2.0  # 更大的变化
        # 人为制造一些物理上不合理的姿态
        poses = poses.view(batch_size, seq_len, 50, 3)
        # 让左右手距离过远
        poses[:, :, 8:29, 0] += 5.0   # 左手x坐标偏移很大
        poses[:, :, 29:50, 0] -= 5.0  # 右手x坐标偏移很大
        poses = poses.view(batch_size, seq_len, pose_dim)
    elif pose_type == "jerky":
        # 不平滑的姿态数据（时间不一致）
        poses = torch.zeros(batch_size, seq_len, pose_dim)
        for t in range(seq_len):
            poses[:, t] = torch.randn(batch_size, pose_dim) * 0.5  # 每帧随机
    
    return poses

def test_pose_consistency_loss():
    """测试姿态一致性损失"""
    print("🧪 测试 PoseConsistencyLoss")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建损失函数
    consistency_loss = PoseConsistencyLoss(
        bone_length_weight=1.0,
        joint_angle_weight=0.5,
        symmetry_weight=0.3,
        temporal_weight=0.2,
        device=device
    )
    
    # 测试不同类型的姿态
    test_cases = [
        ("正常姿态", "normal"),
        ("异常姿态", "abnormal"), 
        ("抖动姿态", "jerky")
    ]
    
    results = {}
    
    for case_name, pose_type in test_cases:
        print(f"\n🔍 测试 {case_name}:")
        
        # 生成测试数据
        pred_poses = generate_test_poses(
            batch_size=2, seq_len=10, pose_dim=150, pose_type=pose_type
        ).to(device)
        
        # 计算损失
        with torch.no_grad():
            losses = consistency_loss(pred_poses)
        
        results[case_name] = losses
        
        # 打印结果
        for key, value in losses.items():
            print(f"  {key:20s}: {value.item():.6f}")
    
    # 比较分析
    print("\n📊 比较分析:")
    print("-" * 50)
    
    normal_total = results["正常姿态"]["total"].item()
    abnormal_total = results["异常姿态"]["total"].item()
    jerky_total = results["抖动姿态"]["total"].item()
    
    print(f"正常姿态总损失: {normal_total:.6f}")
    print(f"异常姿态总损失: {abnormal_total:.6f} (是正常的 {abnormal_total/normal_total:.1f} 倍)")
    print(f"抖动姿态总损失: {jerky_total:.6f} (是正常的 {jerky_total/normal_total:.1f} 倍)")
    
    # 验证约束是否有效
    if abnormal_total > normal_total * 2:
        print("✅ 骨骼约束有效：异常姿态产生了更高的损失")
    else:
        print("❌ 骨骼约束可能需要调整")
    
    if jerky_total > normal_total * 1.5:
        print("✅ 时间约束有效：抖动姿态产生了更高的损失")
    else:
        print("❌ 时间约束可能需要调整")

def test_with_target_poses():
    """测试有目标姿态的监督损失"""
    print("\n🎯 测试监督学习损失")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    consistency_loss = PoseConsistencyLoss(device=device)
    
    # 生成相似和不相似的姿态对
    target_poses = generate_test_poses(2, 10, 150, "normal").to(device)
    
    # 相似的预测
    similar_pred = target_poses + torch.randn_like(target_poses) * 0.1
    
    # 不相似的预测
    different_pred = generate_test_poses(2, 10, 150, "abnormal").to(device)
    
    with torch.no_grad():
        similar_losses = consistency_loss(similar_pred, target_poses)
        different_losses = consistency_loss(different_pred, target_poses)
    
    print("相似预测的监督损失:")
    for key, value in similar_losses.items():
        if 'supervised' in key or key == 'total':
            print(f"  {key:20s}: {value.item():.6f}")
    
    print("\n不相似预测的监督损失:")
    for key, value in different_losses.items():
        if 'supervised' in key or key == 'total':
            print(f"  {key:20s}: {value.item():.6f}")
    
    supervised_ratio = different_losses['supervised'].item() / similar_losses['supervised'].item()
    print(f"\n监督损失比值: {supervised_ratio:.2f} (应该 > 1)")
    
    if supervised_ratio > 2:
        print("✅ 监督损失有效：不相似预测产生了更高的损失")
    else:
        print("❌ 监督损失可能需要调整")

if __name__ == "__main__":
    test_pose_consistency_loss()
    test_with_target_poses()
    
    print("\n🎉 测试完成！")
    print("\n💡 使用建议:")
    print("1. 如果正常姿态的损失过高，可以调低权重")
    print("2. 如果异常姿态的损失不够高，可以调高相应约束的权重")
    print("3. 可以根据具体应用场景调整各项损失的权重") 