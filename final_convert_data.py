#!/usr/bin/env python3
"""
最终版数据转换脚本
正确解析.skels文件格式：每帧150维数据 + 时间戳
"""

import os
import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_skels_line_final(line):
    """
    最终版解析.skels文件中的一行数据
    格式：每帧150维数据后跟一个时间戳
    """
    parts = line.strip().split()
    if not parts:
        return []
    
    frames = []
    i = 0
    
    while i < len(parts):
        # 收集150个数值作为一帧
        frame_data = []
        
        # 收集150维姿态数据
        for j in range(150):
            if i + j < len(parts):
                try:
                    value = float(parts[i + j])
                    frame_data.append(value)
                except ValueError:
                    frame_data.append(0.0)
            else:
                frame_data.append(0.0)
        
        # 如果收集到了150维数据，保存这一帧
        if len(frame_data) == 150:
            frames.append(frame_data)
        
        # 移动到下一帧（150维数据 + 1个时间戳）
        i += 151
    
    return frames


def convert_pose_to_signllm_format(pose_data, target_dim=150):
    """将姿态数据转换为SignLLM期望的格式"""
    poses = []
    
    for frame_data in pose_data:
        # 数据已经是150维，直接分解
        # 根据骨架模型：50个关键点 * 3坐标 = 150维
        # 分配：上身8个点(24维) + 左手21个点(63维) + 右手21个点(63维)
        
        # 选择重要的上身关键点（头部、肩膀、手腕等）
        # 关键点索引：0(头), 1(颈), 2(左肩), 3(左肘), 4(左腕), 5(右肩), 6(右肘), 7(右腕)
        upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        pose_keypoints = []
        for idx in upper_body_indices:
            start = idx * 3
            pose_keypoints.extend(frame_data[start:start+3])
        
        # 左手关键点：8-28 (21个点)
        hand_left_keypoints = []
        for idx in range(8, 29):  # 8-28包含21个点
            start = idx * 3
            hand_left_keypoints.extend(frame_data[start:start+3])
        
        # 右手关键点：29-49 (21个点)
        hand_right_keypoints = []
        for idx in range(29, 50):  # 29-49包含21个点
            start = idx * 3
            hand_right_keypoints.extend(frame_data[start:start+3])
        
        pose_dict = {
            "pose_keypoints_2d": pose_keypoints,
            "hand_left_keypoints_2d": hand_left_keypoints,
            "hand_right_keypoints_2d": hand_right_keypoints,
            "face_keypoints_2d": [0.0] * 210  # 占位符
        }
        
        poses.append(pose_dict)
    
    return poses


def create_signllm_dataset(data_dir, output_dir, split="dev", language="ASL", max_samples=None):
    """创建SignLLM格式的数据集"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # 读取文件
    text_file = data_dir / f"{split}.text"
    skels_file = data_dir / f"{split}.skels"
    files_file = data_dir / f"{split}.files"
    
    if not all([text_file.exists(), skels_file.exists()]):
        logger.error(f"Required files not found in {data_dir}")
        return False
    
    # 读取文件列表
    file_names = []
    if files_file.exists():
        with open(files_file, 'r') as f:
            file_names = [line.strip() for line in f.readlines()]
    
    # 读取文本数据
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    
    # 读取骨架数据
    with open(skels_file, 'r') as f:
        skels_lines = [line.strip() for line in f.readlines()]
    
    if len(texts) != len(skels_lines):
        logger.error(f"Text and skels data count mismatch: {len(texts)} vs {len(skels_lines)}")
        return False
    
    # 限制处理样本数（用于测试）
    if max_samples:
        texts = texts[:max_samples]
        skels_lines = skels_lines[:max_samples]
        file_names = file_names[:max_samples] if file_names else []
    
    # 创建输出目录
    lang_output_dir = output_dir / language / split
    lang_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理样本
    processed_count = 0
    failed_count = 0
    total_frames = 0
    
    for i, (text, skels_line) in enumerate(tqdm(zip(texts, skels_lines), total=len(texts), desc=f"Processing {split}")):
        try:
            # 解析骨架数据
            pose_frames = parse_skels_line_final(skels_line)
            
            if not pose_frames:
                logger.warning(f"No pose data found for sample {i}")
                failed_count += 1
                continue
            
            # 转换为SignLLM格式
            poses = convert_pose_to_signllm_format(pose_frames)
            
            # 创建样本目录
            if i < len(file_names) and file_names[i]:
                sample_name = file_names[i].replace('/', '_').replace('\\', '_')
                if sample_name.startswith(f"{split}/"):
                    sample_name = sample_name[len(f"{split}/"):]
            else:
                sample_name = f"sample_{i:06d}"
            
            sample_dir = lang_output_dir / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存文本
            with open(sample_dir / "text.txt", 'w', encoding='utf-8') as f:
                f.write(text)
            
            # 保存姿态数据
            pose_data = {
                "poses": poses,
                "num_frames": len(poses),
                "original_index": i
            }
            
            with open(sample_dir / "pose.json", 'w') as f:
                json.dump(pose_data, f, indent=2)
            
            processed_count += 1
            total_frames += len(poses)
            
            # 每处理100个样本输出一次统计
            if processed_count % 100 == 0:
                avg_frames = total_frames / processed_count
                logger.info(f"Processed {processed_count} samples, average {avg_frames:.1f} frames per sample")
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            failed_count += 1
            continue
    
    # 创建数据集索引
    create_dataset_index(lang_output_dir, language, split)
    
    avg_frames = total_frames / processed_count if processed_count > 0 else 0
    logger.info(f"Conversion completed for {split}:")
    logger.info(f"  Processed: {processed_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"  Success rate: {processed_count/(processed_count+failed_count)*100:.1f}%")
    logger.info(f"  Average frames per sample: {avg_frames:.1f}")
    logger.info(f"  Total frames: {total_frames}")
    
    return processed_count > 0


def create_dataset_index(dataset_dir, language, split):
    """创建数据集索引文件"""
    index_data = []
    
    for sample_dir in dataset_dir.iterdir():
        if sample_dir.is_dir():
            text_file = sample_dir / "text.txt"
            pose_file = sample_dir / "pose.json"
            
            if text_file.exists() and pose_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                # 修复路径分隔符问题
                relative_pose_path = pose_file.relative_to(dataset_dir.parent.parent)
                pose_file_path = str(relative_pose_path).replace('\\', '/')
                
                index_data.append({
                    "id": sample_dir.name,
                    "text": text,
                    "pose_file": pose_file_path,
                    "language": language
                })
    
    # 保存索引
    index_file = dataset_dir.parent.parent / f"{language}_{split}_index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created index file: {index_file} with {len(index_data)} samples")


def main():
    parser = argparse.ArgumentParser(description="Final data conversion to SignLLM format")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing .text, .skels, and .files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for SignLLM format data")
    parser.add_argument("--splits", nargs='+', default=["dev"],
                       help="Data splits to process")
    parser.add_argument("--language", type=str, default="ASL",
                       help="Language code for the dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    logger.info(f"Converting data from {data_dir} to {output_dir}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Splits: {args.splits}")
    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples}")
    
    success_count = 0
    for split in args.splits:
        logger.info(f"\n=== Processing {split} split ===")
        if create_signllm_dataset(data_dir, output_dir, split, args.language, args.max_samples):
            success_count += 1
    
    if success_count > 0:
        # 创建数据目录配置
        config = {
            "data_dirs": {
                args.language: str(output_dir)
            },
            "languages": [args.language],
            "splits": args.splits,
            "pose_dim": 150,
            "description": "Converted from processed skeletal data (final version)"
        }
        
        config_file = output_dir / "dataset_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"\n=== Conversion completed successfully ===")
        logger.info(f"Dataset configuration saved to: {config_file}")
        logger.info(f"Processed {success_count}/{len(args.splits)} splits")
    else:
        logger.error("All conversions failed!")


if __name__ == "__main__":
    main() 