#!/usr/bin/env python3
"""
SignLLM 姿态一致性损失函数
基于真实骨架结构和生理约束的完整实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


# 从项目中复用的骨架连接关系
REAL_SKELETON_STRUCTURE = [
    # head-neck
    (0, 1, 0),
    # left shoulder
    (1, 2, 1),
    # left arm
    (2, 3, 2), (3, 4, 3),
    # right shoulder
    (1, 5, 1),
    # right arm
    (5, 6, 2), (6, 7, 3),
    # left hand - wrist
    (7, 8, 4),
    # left hand - palm
    (8, 9, 5), (8, 13, 9), (8, 17, 13), (8, 21, 17), (8, 25, 21),
    # left hand - fingers
    (9, 10, 6), (10, 11, 7), (11, 12, 8),
    (13, 14, 10), (14, 15, 11), (15, 16, 12),
    (17, 18, 14), (18, 19, 15), (19, 20, 16),
    (21, 22, 18), (22, 23, 19), (23, 24, 20),
    (25, 26, 22), (26, 27, 23), (27, 28, 24),
    # right hand - wrist
    (4, 29, 4),
    # right hand - palm
    (29, 30, 5), (29, 34, 9), (29, 38, 13), (29, 42, 17), (29, 46, 21),
    # right hand - fingers
    (30, 31, 6), (31, 32, 7), (32, 33, 8),
    (34, 35, 10), (35, 36, 11), (36, 37, 12),
    (38, 39, 14), (39, 40, 15), (40, 41, 16),
    (42, 43, 18), (43, 44, 19), (44, 45, 20),
    (46, 47, 22), (47, 48, 23), (48, 49, 24),
]

REAL_CONNECTIONS = [(start, end) for start, end, _ in REAL_SKELETON_STRUCTURE]


class PoseConsistencyLoss(nn.Module):
    """
    SignLLM专用的姿态一致性损失函数
    基于真实骨架结构和生理约束
    """
    
    def __init__(self, 
                 bone_length_weight: float = 1.0,
                 joint_angle_weight: float = 0.5, 
                 symmetry_weight: float = 0.3,
                 temporal_weight: float = 0.2,
                 device: str = "auto"):
        super().__init__()
        
        self.bone_length_weight = bone_length_weight
        self.joint_angle_weight = joint_angle_weight
        self.symmetry_weight = symmetry_weight
        self.temporal_weight = temporal_weight
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 预计算骨骼长度约束
        self._setup_bone_constraints()
        
        # 预计算对称关系
        self._setup_symmetry_constraints()
        
    def _setup_bone_constraints(self):
        """设置骨骼长度约束"""
        # 定义不同骨骼的期望长度范围（归一化坐标系下）
        self.bone_length_ranges = {
            # 上身骨骼
            (0, 1): (0.05, 0.15),   # 头-颈
            (1, 2): (0.1, 0.2),     # 颈-左肩
            (1, 5): (0.1, 0.2),     # 颈-右肩
            (2, 3): (0.2, 0.35),    # 上臂
            (3, 4): (0.2, 0.35),    # 前臂
            (5, 6): (0.2, 0.35),    # 上臂
            (6, 7): (0.2, 0.35),    # 前臂
            (4, 8): (0.05, 0.15),   # 右腕-左手
            (7, 29): (0.05, 0.15),  # 左腕-右手
        }
        
        # 手部骨骼长度（更小的范围）
        for start, end in REAL_CONNECTIONS:
            if 8 <= start < 50 and 8 <= end < 50:  # 手部连接
                self.bone_length_ranges[(start, end)] = (0.01, 0.08)
    
    def _setup_symmetry_constraints(self):
        """设置对称性约束"""
        # 左右手对称关系
        self.symmetry_pairs = []
        for i in range(21):  # 21个手部关键点
            left_idx = 8 + i   # 左手: 8-28
            right_idx = 29 + i # 右手: 29-49
            self.symmetry_pairs.append((left_idx, right_idx))
        
        # 左右肩对称
        self.symmetry_pairs.extend([(2, 5), (3, 6), (4, 7)])
    
    def forward(self, pred_poses: torch.Tensor, target_poses: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        计算姿态一致性损失
        
        Args:
            pred_poses: 预测姿态 [batch, seq_len, 150]
            target_poses: 目标姿态 [batch, seq_len, 150] (可选，用于监督)
        
        Returns:
            Dict包含各项损失
        """
        batch_size, seq_len, pose_dim = pred_poses.shape
        
        # 重塑为关键点格式 [batch, seq_len, 50, 3]
        pred_joints = pred_poses.view(batch_size, seq_len, 50, 3)
        
        losses = {}
        
        # 1. 骨骼长度约束
        bone_loss = self._bone_length_loss(pred_joints)
        losses['bone_length'] = bone_loss
        
        # 2. 关节角度约束
        angle_loss = self._joint_angle_loss(pred_joints)
        losses['joint_angle'] = angle_loss
        
        # 3. 对称性约束
        symmetry_loss = self._symmetry_loss(pred_joints)
        losses['symmetry'] = symmetry_loss
        
        # 4. 时间一致性约束
        if seq_len > 1:
            temporal_loss = self._temporal_consistency_loss(pred_joints)
            losses['temporal'] = temporal_loss
        else:
            losses['temporal'] = torch.tensor(0.0, device=pred_poses.device)
        
        # 5. 如果有目标姿态，添加监督损失
        if target_poses is not None:
            target_joints = target_poses.view(batch_size, seq_len, 50, 3)
            supervised_loss = self._supervised_consistency_loss(pred_joints, target_joints)
            losses['supervised'] = supervised_loss
        
        # 计算总损失
        total_loss = (self.bone_length_weight * losses['bone_length'] +
                     self.joint_angle_weight * losses['joint_angle'] +
                     self.symmetry_weight * losses['symmetry'] +
                     self.temporal_weight * losses['temporal'])
        
        if 'supervised' in losses:
            total_loss += losses['supervised']
        
        losses['total'] = total_loss
        
        return losses
    
    def _bone_length_loss(self, joints: torch.Tensor) -> torch.Tensor:
        """骨骼长度约束损失"""
        batch_size, seq_len, num_joints, _ = joints.shape
        total_loss = torch.tensor(0.0, device=joints.device)
        
        for (start_idx, end_idx), (min_len, max_len) in self.bone_length_ranges.items():
            if start_idx < num_joints and end_idx < num_joints:
                # 计算骨骼长度
                bone_vectors = joints[:, :, end_idx] - joints[:, :, start_idx]
                bone_lengths = torch.norm(bone_vectors, dim=-1)  # [batch, seq_len]
                
                # 长度约束：超出范围的部分
                length_penalty = torch.clamp(bone_lengths - max_len, min=0) + \
                               torch.clamp(min_len - bone_lengths, min=0)
                
                total_loss += length_penalty.mean()
        
        return total_loss / len(self.bone_length_ranges)
    
    def _joint_angle_loss(self, joints: torch.Tensor) -> torch.Tensor:
        """关节角度约束损失"""
        batch_size, seq_len, num_joints, _ = joints.shape
        total_loss = torch.tensor(0.0, device=joints.device)
        count = 0
        
        # 检查关键关节的角度约束
        critical_joints = [
            (1, 2, 3),  # 左肩-左肘角度
            (1, 5, 6),  # 右肩-右肘角度
            (2, 3, 4),  # 左肘角度
            (5, 6, 7),  # 右肘角度
        ]
        
        for joint_a, joint_b, joint_c in critical_joints:
            if joint_a < num_joints and joint_b < num_joints and joint_c < num_joints:
                # 计算三个关节的角度
                vec1 = joints[:, :, joint_a] - joints[:, :, joint_b]  # [batch, seq_len, 3]
                vec2 = joints[:, :, joint_c] - joints[:, :, joint_b]
                
                # 归一化向量
                vec1_norm = F.normalize(vec1, dim=-1)
                vec2_norm = F.normalize(vec2, dim=-1)
                
                # 计算夹角余弦值
                cos_angle = torch.sum(vec1_norm * vec2_norm, dim=-1)
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                
                # 角度约束：肘部应该在合理范围内(30-150度)
                angle_radians = torch.acos(cos_angle)
                angle_degrees = angle_radians * 180.0 / np.pi
                
                # 惩罚过度弯曲或过度伸展
                angle_penalty = torch.clamp(30 - angle_degrees, min=0) + \
                              torch.clamp(angle_degrees - 150, min=0)
                
                total_loss += angle_penalty.mean()
                count += 1
        
        return total_loss / max(count, 1)
    
    def _symmetry_loss(self, joints: torch.Tensor) -> torch.Tensor:
        """对称性约束损失"""
        batch_size, seq_len, num_joints, _ = joints.shape
        total_loss = torch.tensor(0.0, device=joints.device)
        
        for left_idx, right_idx in self.symmetry_pairs:
            if left_idx < num_joints and right_idx < num_joints:
                # 左右关键点位置（x坐标应该对称，y,z坐标应该相似）
                left_joint = joints[:, :, left_idx]   # [batch, seq_len, 3]
                right_joint = joints[:, :, right_idx]
                
                # x坐标对称性（假设中心为0）
                x_symmetry_loss = torch.abs(left_joint[:, :, 0] + right_joint[:, :, 0])
                
                # y,z坐标相似性
                yz_similarity_loss = torch.abs(left_joint[:, :, 1:] - right_joint[:, :, 1:]).mean(dim=-1)
                
                total_loss += (x_symmetry_loss + yz_similarity_loss).mean()
        
        return total_loss / len(self.symmetry_pairs)
    
    def _temporal_consistency_loss(self, joints: torch.Tensor) -> torch.Tensor:
        """时间一致性约束损失"""
        batch_size, seq_len, num_joints, _ = joints.shape
        
        # 相邻帧之间的变化应该平滑
        joint_diff = joints[:, 1:] - joints[:, :-1]  # [batch, seq_len-1, num_joints, 3]
        
        # 计算加速度（二阶导数）
        if seq_len > 2:
            acceleration = joint_diff[:, 1:] - joint_diff[:, :-1]  # [batch, seq_len-2, num_joints, 3]
            # 惩罚过大的加速度
            accel_penalty = torch.norm(acceleration, dim=-1).mean()
        else:
            accel_penalty = torch.tensor(0.0, device=joints.device)
        
        # 惩罚过大的速度变化
        velocity_penalty = torch.norm(joint_diff, dim=-1).mean()
        
        return accel_penalty + 0.5 * velocity_penalty
    
    def _supervised_consistency_loss(self, pred_joints: torch.Tensor, 
                                   target_joints: torch.Tensor) -> torch.Tensor:
        """基于目标姿态的监督一致性损失"""
        # 计算目标姿态的骨骼长度作为参考
        target_bone_lengths = {}
        pred_bone_lengths = {}
        
        total_loss = torch.tensor(0.0, device=pred_joints.device)
        count = 0
        
        for start_idx, end_idx in REAL_CONNECTIONS:
            if start_idx < 50 and end_idx < 50:
                # 目标骨骼长度
                target_bone = torch.norm(target_joints[:, :, end_idx] - target_joints[:, :, start_idx], dim=-1)
                # 预测骨骼长度
                pred_bone = torch.norm(pred_joints[:, :, end_idx] - pred_joints[:, :, start_idx], dim=-1)
                
                # 骨骼长度应该相似
                bone_length_diff = torch.abs(pred_bone - target_bone)
                total_loss += bone_length_diff.mean()
                count += 1
        
        return total_loss / max(count, 1)


# 使用示例
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建损失函数
    consistency_loss = PoseConsistencyLoss(
        bone_length_weight=1.0,
        joint_angle_weight=0.5,
        symmetry_weight=0.3,
        temporal_weight=0.2,
        device=device
    )
    
    # 测试数据
    batch_size, seq_len, pose_dim = 2, 10, 150
    pred_poses = torch.randn(batch_size, seq_len, pose_dim, device=device)
    target_poses = torch.randn(batch_size, seq_len, pose_dim, device=device)
    
    # 计算损失
    losses = consistency_loss(pred_poses, target_poses)
    
    print("Pose Consistency Loss 测试:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}") 