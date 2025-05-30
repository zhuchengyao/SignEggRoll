#!/usr/bin/env python3
"""
改进的 SignLLM 评估系统
解决余弦相似度不适合姿态任务的问题
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


class ImprovedSignLLMEvaluator:
    """改进的SignLLM评估器 - 使用更适合姿态的评估指标"""
    
    def __init__(self):
        pass
    
    def evaluate_poses(self, predictions: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, float]:
        """评估姿态生成质量 - 使用多种互补的指标"""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length")
        
        metrics = {}
        
        # 1. DTW距离（时间序列相似性）
        dtw_scores = []
        for pred, target in zip(predictions, targets):
            dtw_distance, _ = fastdtw(pred, target, dist=euclidean)
            dtw_scores.append(dtw_distance)
        
        metrics["dtw_distance"] = np.mean(dtw_scores)
        metrics["dtw_score"] = 1.0 / (1.0 + metrics["dtw_distance"])
        
        # 2. 基础距离指标
        all_pred = np.concatenate([p.flatten() for p in predictions])
        all_target = np.concatenate([t.flatten() for t in targets])
        
        metrics["mse"] = mean_squared_error(all_target, all_pred)
        metrics["mae"] = mean_absolute_error(all_target, all_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        
        # 3. 改进的姿态相似度指标
        pose_similarities = []
        euclidean_similarities = []
        weighted_similarities = []
        
        for pred, target in zip(predictions, targets):
            # 多种相似度计算
            cosine_sim = self._calculate_cosine_similarity(pred, target)
            euclidean_sim = self._calculate_euclidean_similarity(pred, target)
            weighted_sim = self._calculate_weighted_pose_similarity(pred, target)
            
            pose_similarities.append(cosine_sim)
            euclidean_similarities.append(euclidean_sim)
            weighted_similarities.append(weighted_sim)
        
        metrics["cosine_similarity"] = np.mean(pose_similarities)  # 原来的指标
        metrics["euclidean_similarity"] = np.mean(euclidean_similarities)  # 基于距离
        metrics["weighted_similarity"] = np.mean(weighted_similarities)  # 加权指标
        metrics["pose_similarity"] = np.mean(weighted_similarities)  # 主要指标
        
        # 4. 运动平滑度
        motion_smoothness = []
        for pred in predictions:
            smoothness = self._calculate_motion_smoothness(pred)
            motion_smoothness.append(smoothness)
        
        metrics["motion_smoothness"] = np.mean(motion_smoothness)
        
        # 5. 关键点准确性（分部位评估）
        keypoint_metrics = self._evaluate_keypoint_accuracy(predictions, targets)
        metrics.update(keypoint_metrics)
        
        return metrics
    
    def _calculate_cosine_similarity(self, pred: np.ndarray, target: np.ndarray) -> float:
        """原来的余弦相似度（保留用于对比）"""
        min_len = min(len(pred), len(target))
        pred = pred[:min_len]
        target = target[:min_len]
        
        frame_similarities = []
        for p_frame, t_frame in zip(pred, target):
            p_norm = np.linalg.norm(p_frame)
            t_norm = np.linalg.norm(t_frame)
            
            if p_norm == 0 or t_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(p_frame, t_frame) / (p_norm * t_norm)
            
            frame_similarities.append(max(0.0, similarity))
        
        return np.mean(frame_similarities)
    
    def _calculate_euclidean_similarity(self, pred: np.ndarray, target: np.ndarray) -> float:
        """基于欧氏距离的相似度（更适合姿态任务）"""
        min_len = min(len(pred), len(target))
        pred = pred[:min_len]
        target = target[:min_len]
        
        frame_similarities = []
        for p_frame, t_frame in zip(pred, target):
            # 计算欧氏距离，转换为相似度
            distance = np.linalg.norm(p_frame - t_frame)
            similarity = 1.0 / (1.0 + distance)  # 距离越小，相似度越高
            frame_similarities.append(similarity)
        
        return np.mean(frame_similarities)
    
    def _calculate_weighted_pose_similarity(self, pred: np.ndarray, target: np.ndarray) -> float:
        """加权姿态相似度 - 重要部位权重更高"""
        min_len = min(len(pred), len(target))
        pred = pred[:min_len]
        target = target[:min_len]
        
        # 重塑为关键点格式 [seq_len, 50, 3]
        pred_joints = pred.reshape(min_len, 50, 3)
        target_joints = target.reshape(min_len, 50, 3)
        
        # 定义关键点权重
        weights = np.ones(50)
        weights[:8] = 2.0      # 上身关键点权重更高
        weights[8:29] = 1.5    # 左手
        weights[29:50] = 1.5   # 右手
        weights[[0, 1, 4, 7]] = 3.0  # 头部、颈部、手腕等关键点
        
        frame_similarities = []
        for p_frame, t_frame in zip(pred_joints, target_joints):
            # 计算每个关键点的距离
            joint_distances = np.linalg.norm(p_frame - t_frame, axis=1)
            
            # 应用权重
            weighted_distances = joint_distances * weights
            
            # 转换为相似度
            similarities = 1.0 / (1.0 + weighted_distances)
            
            # 加权平均
            frame_similarity = np.average(similarities, weights=weights)
            frame_similarities.append(frame_similarity)
        
        return np.mean(frame_similarities)
    
    def _evaluate_keypoint_accuracy(self, predictions: List[np.ndarray], 
                                   targets: List[np.ndarray]) -> Dict[str, float]:
        """分部位评估关键点准确性"""
        metrics = {}
        
        # 定义部位索引
        body_parts = {
            'upper_body': list(range(0, 8)),      # 上身
            'left_hand': list(range(8, 29)),      # 左手
            'right_hand': list(range(29, 50)),    # 右手
            'key_joints': [0, 1, 4, 7]            # 关键关节
        }
        
        for part_name, indices in body_parts.items():
            part_errors = []
            
            for pred, target in zip(predictions, targets):
                min_len = min(len(pred), len(target))
                pred_part = pred[:min_len].reshape(min_len, 50, 3)[:, indices, :]
                target_part = target[:min_len].reshape(min_len, 50, 3)[:, indices, :]
                
                # 计算该部位的平均误差
                error = np.mean(np.linalg.norm(pred_part - target_part, axis=2))
                part_errors.append(error)
            
            metrics[f"{part_name}_error"] = np.mean(part_errors)
            metrics[f"{part_name}_similarity"] = 1.0 / (1.0 + metrics[f"{part_name}_error"])
        
        return metrics
    
    def _calculate_motion_smoothness(self, poses: np.ndarray) -> float:
        """计算运动平滑度"""
        if len(poses) < 2:
            return 1.0
        
        # 计算相邻帧之间的差异
        diffs = []
        for i in range(1, len(poses)):
            diff = np.linalg.norm(poses[i] - poses[i-1])
            diffs.append(diff)
        
        # 平滑度 = 1 / (1 + 平均差异)
        avg_diff = np.mean(diffs)
        smoothness = 1.0 / (1.0 + avg_diff)
        
        return smoothness


def compare_evaluation_methods(predictions: List[np.ndarray], targets: List[np.ndarray]):
    """比较不同评估方法的结果"""
    from evaluation import SignLLMEvaluator
    
    # 原始评估器
    original_evaluator = SignLLMEvaluator()
    original_metrics = original_evaluator.evaluate_poses(predictions, targets)
    
    # 改进评估器
    improved_evaluator = ImprovedSignLLMEvaluator()
    improved_metrics = improved_evaluator.evaluate_poses(predictions, targets)
    
    print("📊 评估方法对比:")
    print("=" * 60)
    
    print("\n🔸 原始评估器:")
    for key, value in original_metrics.items():
        if 'similarity' in key:
            print(f"  {key:25s}: {value:.4f}")
    
    print("\n🔹 改进评估器:")
    for key, value in improved_metrics.items():
        if 'similarity' in key or 'error' in key:
            print(f"  {key:25s}: {value:.4f}")
    
    print("\n💡 建议:")
    print("  - 使用 weighted_similarity 作为主要指标")
    print("  - euclidean_similarity 更适合姿态任务")
    print("  - 分部位误差帮助诊断问题")
    
    return original_metrics, improved_metrics


if __name__ == "__main__":
    # 测试评估器
    evaluator = ImprovedSignLLMEvaluator()
    
    # 生成测试数据
    predictions = [np.random.randn(50, 150) for _ in range(5)]
    targets = [np.random.randn(50, 150) for _ in range(5)]
    
    # 测试改进的评估
    metrics = evaluator.evaluate_poses(predictions, targets)
    print("改进的姿态评估指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*50)
    compare_evaluation_methods(predictions, targets) 