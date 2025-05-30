#!/usr/bin/env python3
"""
数据质量检查工具 - 检测和修复手语数据集中的问题
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, pose_dim: int = 150):
        self.pose_dim = pose_dim
        self.quality_issues = []
        
    def check_pose_sequence(self, poses: np.ndarray, sequence_id: str = "") -> Dict[str, bool]:
        """检查单个姿态序列的质量"""
        issues = {}
        
        # 1. 检查NaN和无穷值
        issues['has_nan'] = np.isnan(poses).any()
        issues['has_inf'] = np.isinf(poses).any()
        
        # 2. 检查异常值（基于Z-score）
        z_scores = np.abs((poses - np.mean(poses, axis=0)) / (np.std(poses, axis=0) + 1e-8))
        issues['has_outliers'] = (z_scores > 4).any()
        
        # 3. 检查运动平滑度
        if len(poses) > 1:
            velocity = np.diff(poses, axis=0)
            acceleration = np.diff(velocity, axis=0)
            issues['abrupt_motion'] = (np.abs(acceleration) > 0.5).any()
        
        # 4. 检查序列长度
        issues['too_short'] = len(poses) < 10
        issues['too_long'] = len(poses) > 500
        
        # 5. 检查关键点缺失（假设某些维度为关键关节）
        key_joints_indices = list(range(0, min(21, self.pose_dim)))  # 前21个维度作为关键关节
        if key_joints_indices:
            key_joints = poses[:, key_joints_indices]
            issues['missing_key_joints'] = (np.abs(key_joints) < 1e-6).all(axis=0).any()
        
        # 记录有问题的序列
        if any(issues.values()):
            self.quality_issues.append({
                'sequence_id': sequence_id,
                'issues': {k: v for k, v in issues.items() if v}
            })
        
        return issues
    
    def fix_pose_sequence(self, poses: np.ndarray) -> np.ndarray:
        """修复姿态序列中的问题"""
        fixed_poses = poses.copy()
        
        # 1. 修复NaN和无穷值
        fixed_poses = np.nan_to_num(fixed_poses, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 2. 异常值处理（使用中位数滤波）
        from scipy.signal import medfilt
        for i in range(fixed_poses.shape[1]):
            if fixed_poses.shape[0] > 3:  # 确保有足够的点进行滤波
                fixed_poses[:, i] = medfilt(fixed_poses[:, i], kernel_size=3)
        
        # 3. 运动平滑化
        if len(fixed_poses) > 2:
            # 高斯滤波平滑化
            from scipy.ndimage import gaussian_filter1d
            for i in range(fixed_poses.shape[1]):
                fixed_poses[:, i] = gaussian_filter1d(fixed_poses[:, i], sigma=1.0)
        
        return fixed_poses
    
    def analyze_dataset_quality(self, dataset, sample_ratio: float = 1.0) -> Dict:
        """分析整个数据集的质量"""
        total_samples = len(dataset)
        sample_count = int(total_samples * sample_ratio)
        
        quality_stats = {
            'total_samples': total_samples,
            'checked_samples': sample_count,
            'issue_counts': {
                'has_nan': 0,
                'has_inf': 0,
                'has_outliers': 0,
                'abrupt_motion': 0,
                'too_short': 0,
                'too_long': 0,
                'missing_key_joints': 0
            },
            'problematic_samples': [],
            'quality_score': 0.0
        }
        
        logger.info(f"检查数据集质量 ({sample_count}/{total_samples} 样本)...")
        
        for i in range(min(sample_count, total_samples)):
            try:
                sample = dataset[i]
                poses = sample['pose_sequence'].numpy() if torch.is_tensor(sample['pose_sequence']) else sample['pose_sequence']
                
                issues = self.check_pose_sequence(poses, f"sample_{i}")
                
                # 统计问题
                for issue_type, has_issue in issues.items():
                    if has_issue:
                        quality_stats['issue_counts'][issue_type] += 1
                
                # 记录有问题的样本
                if any(issues.values()):
                    quality_stats['problematic_samples'].append({
                        'index': i,
                        'issues': [k for k, v in issues.items() if v],
                        'text': sample.get('text', 'Unknown')
                    })
                    
            except Exception as e:
                logger.warning(f"检查样本 {i} 时出错: {e}")
                quality_stats['issue_counts']['corrupted'] = quality_stats['issue_counts'].get('corrupted', 0) + 1
        
        # 计算质量分数
        total_issues = sum(quality_stats['issue_counts'].values())
        quality_stats['quality_score'] = max(0.0, 1.0 - (total_issues / sample_count))
        
        return quality_stats
    
    def generate_quality_report(self, quality_stats: Dict, output_dir: str = "quality_reports"):
        """生成质量报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 文本报告
        report_text = f"""
数据质量报告
============

总体统计:
- 总样本数: {quality_stats['total_samples']}
- 检查样本数: {quality_stats['checked_samples']}
- 质量分数: {quality_stats['quality_score']:.3f}

问题统计:
"""
        for issue_type, count in quality_stats['issue_counts'].items():
            percentage = (count / quality_stats['checked_samples']) * 100
            report_text += f"- {issue_type}: {count} ({percentage:.1f}%)\n"
        
        report_text += f"\n有问题的样本总数: {len(quality_stats['problematic_samples'])}\n"
        
        # 保存文本报告
        with open(output_path / "quality_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        
        # 2. 可视化报告
        self._create_quality_visualizations(quality_stats, output_path)
        
        logger.info(f"质量报告已保存到: {output_path}")
        return report_text
    
    def _create_quality_visualizations(self, quality_stats: Dict, output_path: Path):
        """创建质量可视化图表"""
        # 问题分布饼图
        issue_counts = quality_stats['issue_counts']
        valid_issues = {k: v for k, v in issue_counts.items() if v > 0}
        
        if valid_issues:
            plt.figure(figsize=(10, 6))
            plt.pie(valid_issues.values(), labels=valid_issues.keys(), autopct='%1.1f%%')
            plt.title('数据质量问题分布')
            plt.savefig(output_path / "issue_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 质量分数条形图
        plt.figure(figsize=(8, 6))
        categories = ['整体质量分数']
        scores = [quality_stats['quality_score']]
        
        bars = plt.bar(categories, scores, color=['green' if s > 0.8 else 'orange' if s > 0.6 else 'red' for s in scores])
        plt.ylim(0, 1.0)
        plt.ylabel('质量分数')
        plt.title('数据集质量评估')
        
        # 添加分数标签
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.savefig(output_path / "quality_score.png", dpi=300, bbox_inches='tight')
        plt.close()


class DataAugmentor:
    """数据增强器 - 提高数据质量和多样性"""
    
    def __init__(self, pose_dim: int = 150):
        self.pose_dim = pose_dim
    
    def temporal_augmentation(self, poses: np.ndarray, methods: List[str] = ['speed', 'reverse']) -> List[np.ndarray]:
        """时间增强"""
        augmented = [poses]  # 原始序列
        
        if 'speed' in methods:
            # 速度变化（0.8x, 1.2x）
            for factor in [0.8, 1.2]:
                new_length = int(len(poses) * factor)
                if new_length > 5:  # 确保最小长度
                    indices = np.linspace(0, len(poses)-1, new_length)
                    augmented_poses = np.array([np.interp(indices, range(len(poses)), poses[:, i]) 
                                              for i in range(poses.shape[1])]).T
                    augmented.append(augmented_poses)
        
        if 'reverse' in methods:
            # 时间反转
            augmented.append(poses[::-1])
        
        return augmented
    
    def spatial_augmentation(self, poses: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """空间增强"""
        # 添加高斯噪声
        noise = np.random.normal(0, noise_std, poses.shape)
        return poses + noise
    
    def pose_normalization(self, poses: np.ndarray, method: str = 'standard') -> np.ndarray:
        """姿态标准化"""
        if method == 'standard':
            # 标准化到 [-1, 1]
            pose_min, pose_max = poses.min(), poses.max()
            if pose_max > pose_min:
                return 2 * (poses - pose_min) / (pose_max - pose_min) - 1
            return poses
        elif method == 'zero_mean':
            # 零均值标准化
            return (poses - poses.mean(axis=0)) / (poses.std(axis=0) + 1e-8)
        
        return poses


# 使用示例
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    
    from data_processor import MultilingualSignDataset
    
    # 创建数据质量检查器
    checker = DataQualityChecker(pose_dim=150)
    
    # 加载数据集
    try:
        dataset = MultilingualSignDataset(
            data_dirs={"ASL": "datasets/signllm_data_complete"},
            languages=["ASL"],
            split="dev",
            max_sequence_length=256,
            pose_dim=150,
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 分析数据质量
        quality_stats = checker.analyze_dataset_quality(dataset, sample_ratio=0.1)  # 检查10%的样本
        
        # 生成报告
        report = checker.generate_quality_report(quality_stats)
        print(report)
        
    except Exception as e:
        print(f"数据集加载失败: {e}")
        print("请确保数据集路径正确") 