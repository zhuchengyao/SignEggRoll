#!/usr/bin/env python3
"""
ASL数据格式转换脚本
将.skels和.text格式的数据转换为dev_*目录格式
"""

import os
import json
import numpy as np
from typing import List, Dict
import argparse
from tqdm import tqdm
import shutil

def read_skels_file(filepath: str) -> List[np.ndarray]:
    """读取.skels文件"""
    sequences = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # 解析每行的数据
                try:
                    values = list(map(float, line.split()))
                    # 重塑为 (num_frames, 67, 3) 格式
                    if len(values) % 201 == 0:  # 67*3=201
                        num_frames = len(values) // 201
                        sequence = np.array(values).reshape(num_frames, 67, 3)
                        sequences.append(sequence)
                except Exception as e:
                    print(f"解析行数据失败: {e}")
                    continue
    
    return sequences

def read_text_file(filepath: str) -> List[str]:
    """读取.text文件"""
    texts = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                texts.append(text)
    
    return texts

def convert_skeleton_to_openpose(skeleton: np.ndarray) -> List[Dict]:
    """
    将skeleton格式转换为OpenPose格式
    skeleton: (num_frames, 67, 3)
    """
    poses = []
    
    for frame_idx in range(skeleton.shape[0]):
        frame_data = skeleton[frame_idx]  # (67, 3)
        
        # 提取不同部分的关键点
        # 假设前18个点是身体关键点，接下来的21个是左手，再21个是右手，最后7个是面部/其他
        pose_keypoints = frame_data[:18].flatten()  # 身体关键点
        left_hand_keypoints = frame_data[18:39].flatten()  # 左手关键点
        right_hand_keypoints = frame_data[39:60].flatten()  # 右手关键点
        face_keypoints = frame_data[60:67].flatten()  # 面部关键点（简化）
        
        # 补齐面部关键点到70个点（210维）
        face_keypoints_full = np.zeros(210)
        face_keypoints_full[:len(face_keypoints)] = face_keypoints
        
        pose_data = {
            'pose_keypoints_2d': pose_keypoints.tolist(),
            'hand_left_keypoints_2d': left_hand_keypoints.tolist(),
            'hand_right_keypoints_2d': right_hand_keypoints.tolist(),
            'face_keypoints_2d': face_keypoints_full.tolist()
        }
        
        poses.append(pose_data)
    
    return poses

def convert_dataset(input_dir: str, output_dir: str, subset: str = "dev", max_samples: int = None):
    """
    转换数据集
    
    Args:
        input_dir: 输入目录路径（包含.skels和.text文件）
        output_dir: 输出目录路径
        subset: 数据子集名称 (dev, test, val)
        max_samples: 最大转换样本数（用于测试）
    """
    
    print(f"开始转换 {subset} 数据集...")
    
    # 文件路径
    skels_file = os.path.join(input_dir, f"{subset}.skels")
    text_file = os.path.join(input_dir, f"{subset}.text")
    
    if not os.path.exists(skels_file) or not os.path.exists(text_file):
        print(f"找不到文件: {skels_file} 或 {text_file}")
        return
    
    # 读取数据
    print("读取skeleton数据...")
    sequences = read_skels_file(skels_file)
    
    print("读取文本数据...")
    texts = read_text_file(text_file)
    
    print(f"找到 {len(sequences)} 个skeleton序列和 {len(texts)} 个文本")
    
    # 确保数据数量匹配
    min_count = min(len(sequences), len(texts))
    if max_samples:
        min_count = min(min_count, max_samples)
    
    print(f"将转换 {min_count} 个样本")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换每个样本
    for i in tqdm(range(min_count), desc="转换数据"):
        # 创建样本目录
        sample_dir = os.path.join(output_dir, f"dev_{i:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 转换skeleton到OpenPose格式
        poses = convert_skeleton_to_openpose(sequences[i])
        
        # 保存pose.json
        pose_data = {'poses': poses}
        pose_file = os.path.join(sample_dir, 'pose.json')
        with open(pose_file, 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        # 保存text.txt
        text_file_path = os.path.join(sample_dir, 'text.txt')
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(texts[i])
    
    print(f"转换完成！输出目录: {output_dir}")
    print(f"成功转换 {min_count} 个样本")

def main():
    parser = argparse.ArgumentParser(description='转换ASL数据格式')
    parser.add_argument('--input_dir', type=str, default='datasets/final_data', 
                       help='输入目录路径')
    parser.add_argument('--output_dir', type=str, default='datasets/asl_converted', 
                       help='输出目录路径')
    parser.add_argument('--subset', type=str, choices=['dev', 'test', 'val'], 
                       default='dev', help='要转换的数据子集')
    parser.add_argument('--max_samples', type=int, default=100, 
                       help='最大转换样本数（测试用）')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"输入目录不存在: {args.input_dir}")
        return
    
    # 转换数据
    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        subset=args.subset,
        max_samples=args.max_samples
    )
    
    print("\n数据转换完成！")
    print(f"现在可以使用以下命令测试数据加载:")
    print(f"python test_text2video.py data")

if __name__ == "__main__":
    main() 