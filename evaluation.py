"""
SignLLM评估模块
包含DTW、BLEU、姿态相似度等评估指标
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

# 确保NLTK数据已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class SignLLMEvaluator:
    """SignLLM评估器"""
    
    def __init__(self):
        self.smoothing_function = SmoothingFunction().method1
    
    def evaluate_poses(self, predictions: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, float]:
        """评估姿态生成质量"""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length")
        
        metrics = {}
        
        # DTW距离
        dtw_scores = []
        for pred, target in zip(predictions, targets):
            dtw_distance, _ = fastdtw(pred, target, dist=euclidean)
            dtw_scores.append(dtw_distance)
        
        metrics["dtw_distance"] = np.mean(dtw_scores)
        metrics["dtw_score"] = 1.0 / (1.0 + metrics["dtw_distance"])  # 转换为分数（越高越好）
        
        # MSE和MAE
        all_pred = np.concatenate([p.flatten() for p in predictions])
        all_target = np.concatenate([t.flatten() for t in targets])
        
        metrics["mse"] = mean_squared_error(all_target, all_pred)
        metrics["mae"] = mean_absolute_error(all_target, all_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        
        # 姿态相似度（基于关键点距离）
        pose_similarities = []
        for pred, target in zip(predictions, targets):
            similarity = self._calculate_pose_similarity(pred, target)
            pose_similarities.append(similarity)
        
        metrics["pose_similarity"] = np.mean(pose_similarities)
        
        # 运动平滑度
        motion_smoothness = []
        for pred in predictions:
            smoothness = self._calculate_motion_smoothness(pred)
            motion_smoothness.append(smoothness)
        
        metrics["motion_smoothness"] = np.mean(motion_smoothness)
        
        return metrics
    
    def evaluate_text_to_pose(self, texts: List[str], predictions: List[np.ndarray], 
                            targets: List[np.ndarray]) -> Dict[str, float]:
        """评估文本到姿态的生成质量"""
        # 基础姿态评估
        metrics = self.evaluate_poses(predictions, targets)
        
        # 添加文本相关的评估
        # 这里可以添加语义一致性评估等
        
        return metrics
    
    def evaluate_gloss_generation(self, predicted_gloss: List[List[str]], 
                                target_gloss: List[List[str]]) -> Dict[str, float]:
        """评估gloss生成质量"""
        if len(predicted_gloss) != len(target_gloss):
            raise ValueError("Predicted and target gloss must have the same length")
        
        metrics = {}
        
        # BLEU分数
        bleu_scores = []
        for pred, target in zip(predicted_gloss, target_gloss):
            if len(pred) == 0 or len(target) == 0:
                bleu_scores.append(0.0)
                continue
            
            # 计算BLEU-1到BLEU-4
            bleu_1 = sentence_bleu([target], pred, weights=(1, 0, 0, 0), 
                                 smoothing_function=self.smoothing_function)
            bleu_2 = sentence_bleu([target], pred, weights=(0.5, 0.5, 0, 0), 
                                 smoothing_function=self.smoothing_function)
            bleu_3 = sentence_bleu([target], pred, weights=(0.33, 0.33, 0.33, 0), 
                                 smoothing_function=self.smoothing_function)
            bleu_4 = sentence_bleu([target], pred, weights=(0.25, 0.25, 0.25, 0.25), 
                                 smoothing_function=self.smoothing_function)
            
            bleu_scores.append({
                "bleu_1": bleu_1,
                "bleu_2": bleu_2,
                "bleu_3": bleu_3,
                "bleu_4": bleu_4
            })
        
        # 平均BLEU分数
        for key in ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]:
            metrics[key] = np.mean([score[key] for score in bleu_scores])
        
        # 精确度和召回率
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for pred, target in zip(predicted_gloss, target_gloss):
            pred_set = set(pred)
            target_set = set(target)
            
            if len(pred_set) == 0:
                precision = 0.0
            else:
                precision = len(pred_set & target_set) / len(pred_set)
            
            if len(target_set) == 0:
                recall = 0.0
            else:
                recall = len(pred_set & target_set) / len(target_set)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        metrics["precision"] = np.mean(precision_scores)
        metrics["recall"] = np.mean(recall_scores)
        metrics["f1_score"] = np.mean(f1_scores)
        
        return metrics
    
    def _calculate_pose_similarity(self, pred: np.ndarray, target: np.ndarray) -> float:
        """计算姿态相似度"""
        # 确保维度一致
        min_len = min(len(pred), len(target))
        pred = pred[:min_len]
        target = target[:min_len]
        
        # 计算每帧的相似度
        frame_similarities = []
        for p_frame, t_frame in zip(pred, target):
            # 使用余弦相似度
            p_norm = np.linalg.norm(p_frame)
            t_norm = np.linalg.norm(t_frame)
            
            if p_norm == 0 or t_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(p_frame, t_frame) / (p_norm * t_norm)
            
            frame_similarities.append(max(0.0, similarity))  # 确保非负
        
        return np.mean(frame_similarities)
    
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
    
    def evaluate_multilingual(self, results_by_language: Dict[str, Dict]) -> Dict[str, float]:
        """评估多语言性能"""
        all_metrics = {}
        
        # 计算每种语言的平均性能
        for lang, metrics in results_by_language.items():
            for metric_name, value in metrics.items():
                key = f"{lang}_{metric_name}"
                all_metrics[key] = value
        
        # 计算总体平均性能
        metric_names = set()
        for metrics in results_by_language.values():
            metric_names.update(metrics.keys())
        
        for metric_name in metric_names:
            values = []
            for lang_metrics in results_by_language.values():
                if metric_name in lang_metrics:
                    values.append(lang_metrics[metric_name])
            
            if values:
                all_metrics[f"avg_{metric_name}"] = np.mean(values)
                all_metrics[f"std_{metric_name}"] = np.std(values)
        
        return all_metrics
    
    def back_translation_evaluation(self, original_texts: List[str], 
                                  generated_poses: List[np.ndarray],
                                  pose_to_text_model) -> Dict[str, float]:
        """反向翻译评估"""
        # 将生成的姿态转换回文本
        reconstructed_texts = []
        for poses in generated_poses:
            # 这里需要一个姿态到文本的模型
            # 暂时返回占位符
            reconstructed_texts.append("placeholder")
        
        # 计算文本相似度
        similarities = []
        for orig, recon in zip(original_texts, reconstructed_texts):
            # 这里可以使用BERT相似度或其他文本相似度指标
            # 暂时使用简单的字符串匹配
            similarity = len(set(orig.split()) & set(recon.split())) / max(len(orig.split()), 1)
            similarities.append(similarity)
        
        return {
            "back_translation_similarity": np.mean(similarities),
            "back_translation_std": np.std(similarities)
        }


class PoseQualityAssessment:
    """姿态质量评估"""
    
    @staticmethod
    def assess_anatomical_validity(poses: np.ndarray) -> float:
        """评估解剖学有效性"""
        # 检查关节角度是否在合理范围内
        # 这里需要根据具体的姿态表示来实现
        # 暂时返回占位符
        return 1.0
    
    @staticmethod
    def assess_temporal_consistency(poses: np.ndarray) -> float:
        """评估时间一致性"""
        if len(poses) < 2:
            return 1.0
        
        # 计算相邻帧之间的变化
        changes = []
        for i in range(1, len(poses)):
            change = np.linalg.norm(poses[i] - poses[i-1])
            changes.append(change)
        
        # 检查是否有突然的大幅变化
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        
        # 如果标准差相对于均值很大，说明一致性较差
        if mean_change == 0:
            return 1.0
        
        consistency = 1.0 / (1.0 + std_change / mean_change)
        return consistency
    
    @staticmethod
    def assess_naturalness(poses: np.ndarray) -> float:
        """评估自然度"""
        # 这里可以使用预训练的自然度评估模型
        # 或者基于统计的方法
        # 暂时返回占位符
        return 1.0


def compute_dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """计算两个序列之间的DTW距离"""
    distance, _ = fastdtw(seq1, seq2, dist=euclidean)
    return distance


def compute_frechet_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """计算Fréchet距离"""
    # 简化版的Fréchet距离计算
    # 实际实现可能需要更复杂的算法
    min_len = min(len(seq1), len(seq2))
    seq1 = seq1[:min_len]
    seq2 = seq2[:min_len]
    
    distances = [euclidean(s1, s2) for s1, s2 in zip(seq1, seq2)]
    return np.mean(distances)


if __name__ == "__main__":
    # 测试评估器
    evaluator = SignLLMEvaluator()
    
    # 生成测试数据
    predictions = [np.random.randn(50, 150) for _ in range(10)]
    targets = [np.random.randn(50, 150) for _ in range(10)]
    
    # 测试姿态评估
    pose_metrics = evaluator.evaluate_poses(predictions, targets)
    print("Pose evaluation metrics:")
    for key, value in pose_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 测试gloss评估
    pred_gloss = [["hello", "world"], ["how", "are", "you"]]
    target_gloss = [["hello", "world"], ["how", "are", "you", "today"]]
    
    gloss_metrics = evaluator.evaluate_gloss_generation(pred_gloss, target_gloss)
    print("\nGloss evaluation metrics:")
    for key, value in gloss_metrics.items():
        print(f"  {key}: {value:.4f}") 