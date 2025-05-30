#!/usr/bin/env python3
"""
数据质量检查运行脚本 - 支持命令行参数
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data_quality_checker import DataQualityChecker
from data_processor import MultilingualSignDataset


def main():
    parser = argparse.ArgumentParser(description="运行数据质量检查")
    parser.add_argument("--data-dir", type=str, 
                       default="datasets/signllm_data_complete",
                       help="数据目录路径")
    parser.add_argument("--language", type=str, default="ASL",
                       help="检查的语言")
    parser.add_argument("--split", type=str, default="dev",
                       choices=["dev", "test"],
                       help="数据集分割 (dev=训练集, test=验证集)")
    parser.add_argument("--sample-ratio", type=float, default=0.1,
                       help="检查的样本比例 (0.0-1.0)")
    parser.add_argument("--output-dir", type=str, default="quality_reports",
                       help="输出报告目录")
    
    args = parser.parse_args()
    
    print("🔍 开始数据质量检查...")
    print(f"  数据目录: {args.data_dir}")
    print(f"  语言: {args.language}")
    print(f"  数据分割: {args.split}")
    print(f"  检查比例: {args.sample_ratio * 100:.1f}%")
    
    # 创建数据质量检查器
    checker = DataQualityChecker(pose_dim=150)
    
    try:
        # 加载数据集
        dataset = MultilingualSignDataset(
            data_dirs={args.language: args.data_dir},
            languages=[args.language],
            split=args.split,
            max_sequence_length=256,
            pose_dim=150,
        )
        
        print(f"📊 数据集大小: {len(dataset)} 样本")
        
        # 分析数据质量
        quality_stats = checker.analyze_dataset_quality(
            dataset, sample_ratio=args.sample_ratio
        )
        
        # 生成报告
        report = checker.generate_quality_report(quality_stats, args.output_dir)
        print("\n" + report)
        
        # 质量建议
        quality_score = quality_stats['quality_score']
        if quality_score > 0.8:
            print("✅ 数据质量良好")
        elif quality_score > 0.6:
            print("⚠️  数据质量一般，建议进行数据清理")
        else:
            print("❌ 数据质量较差，强烈建议数据预处理")
        
    except Exception as e:
        print(f"❌ 数据质量检查失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 