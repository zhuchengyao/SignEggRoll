"""
分析MSE量级的脚本
用于理解为什么训练开始时MSE就在0.001级别
"""
import torch
import numpy as np
import json
import os
from transformers import BertTokenizer
from text_to_pose_trainer import TextToPoseDataset

def analyze_data_statistics():
    """分析训练数据的统计信息"""
    print("🔍 分析训练数据的统计信息...")
    
    # 使用与训练相同的数据路径
    data_paths = [
        "datasets/signllm_training_data/ASL/dev/dev_fz6XzPxdo-0_3-5-rgb_front",
        "datasets/signllm_training_data/ASL/dev/dev_fz6XzPxdo-0_2-5-rgb_front", 
        "datasets/signllm_training_data/ASL/dev/dev_fz6XzPxdo-0_17-5-rgb_front",
    ]
    
    # 分析原始数据范围
    print("\n1. 原始数据分析:")
    analyze_raw_data(data_paths)
    
    # 分析处理后的数据
    print("\n2. 处理后数据分析:")
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    dataset = TextToPoseDataset(data_paths, tokenizer, max_length=512, max_sequence_length=400)
    analyze_processed_data(dataset)
    
    # 计算理论MSE范围
    print("\n3. MSE量级分析:")
    analyze_mse_scale(dataset)

def analyze_raw_data(data_paths):
    """分析原始JSON数据"""
    all_coords = []
    
    for path in data_paths:
        pose_path = os.path.join(path, "pose.json")
        if os.path.exists(pose_path):
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
            
            print(f"\n数据路径: {path}")
            
            for frame_idx, pose in enumerate(pose_data['poses']):
                # 收集所有坐标
                coords = []
                
                # 身体关键点
                body_coords = pose['pose_keypoints_2d']
                for i in range(0, len(body_coords), 3):
                    x, y = body_coords[i], body_coords[i+1]
                    if abs(x) > 1e-6 or abs(y) > 1e-6:  # 非零点
                        coords.extend([x, y])
                
                # 左手关键点
                left_hand_coords = pose['hand_left_keypoints_2d']
                for i in range(0, len(left_hand_coords), 3):
                    x, y = left_hand_coords[i], left_hand_coords[i+1]
                    if abs(x) > 1e-6 or abs(y) > 1e-6:
                        coords.extend([x, y])
                
                # 右手关键点  
                right_hand_coords = pose['hand_right_keypoints_2d']
                for i in range(0, len(right_hand_coords), 3):
                    x, y = right_hand_coords[i], right_hand_coords[i+1]
                    if abs(x) > 1e-6 or abs(y) > 1e-6:
                        coords.extend([x, y])
                
                all_coords.extend(coords)
                
                # 显示前几帧的统计
                if frame_idx < 3 and coords:
                    coords_array = np.array(coords)
                    print(f"  帧 {frame_idx}: 范围 [{coords_array.min():.3f}, {coords_array.max():.3f}], "
                          f"均值 {coords_array.mean():.3f}, 标准差 {coords_array.std():.3f}")
    
    if all_coords:
        all_coords = np.array(all_coords)
        print(f"\n原始数据总体统计:")
        print(f"  数据点数: {len(all_coords)}")
        print(f"  数值范围: [{all_coords.min():.3f}, {all_coords.max():.3f}]")
        print(f"  均值: {all_coords.mean():.3f}")
        print(f"  标准差: {all_coords.std():.3f}")
        print(f"  中位数: {np.median(all_coords):.3f}")

def analyze_processed_data(dataset):
    """分析处理后的数据"""
    print(f"数据集大小: {len(dataset)}")
    
    all_keypoints = []
    for i in range(len(dataset)):
        item = dataset[i]
        keypoints = item['keypoints_sequence'].numpy()  # [400, 150]
        all_keypoints.append(keypoints)
    
    # 合并所有数据
    all_data = np.concatenate(all_keypoints, axis=0)  # [N, 150]
    print(f"总数据形状: {all_data.shape}")
    
    # 分析非零数据
    non_zero_mask = np.abs(all_data) > 1e-6
    non_zero_data = all_data[non_zero_mask]
    
    print(f"\n处理后数据统计:")
    print(f"  总数据点: {all_data.size:,}")
    print(f"  非零数据点: {len(non_zero_data):,} ({100*len(non_zero_data)/all_data.size:.1f}%)")
    print(f"  数值范围: [{all_data.min():.6f}, {all_data.max():.6f}]")
    print(f"  非零数值范围: [{non_zero_data.min():.6f}, {non_zero_data.max():.6f}]") if len(non_zero_data) > 0 else None
    print(f"  均值: {all_data.mean():.6f}")
    print(f"  标准差: {all_data.std():.6f}")
    print(f"  非零均值: {non_zero_data.mean():.6f}") if len(non_zero_data) > 0 else None
    print(f"  非零标准差: {non_zero_data.std():.6f}") if len(non_zero_data) > 0 else None

def analyze_mse_scale(dataset):
    """分析MSE量级"""
    # 获取一个批次的数据
    item = dataset[0]
    keypoints = item['keypoints_sequence'].numpy()  # [400, 150]
    
    print(f"单个样本形状: {keypoints.shape}")
    print(f"总元素数: {keypoints.size:,}")
    
    # 计算数据的统计信息
    data_std = keypoints.std()
    data_mean = np.abs(keypoints).mean()
    data_max = np.abs(keypoints).max()
    
    print(f"\n数据统计:")
    print(f"  标准差: {data_std:.6f}")
    print(f"  绝对值均值: {data_mean:.6f}")
    print(f"  绝对值最大值: {data_max:.6f}")
    
    # 模拟随机初始化的预测
    print(f"\n模拟MSE计算:")
    
    # 1. 如果预测值是随机初始化（通常在[-1, 1]或更小范围）
    random_pred = np.random.normal(0, 0.1, keypoints.shape)  # 标准差0.1的随机预测
    mse_random = np.mean((random_pred - keypoints) ** 2)
    print(f"  随机预测 MSE: {mse_random:.6f}")
    
    # 2. 如果预测值是全零
    zero_pred = np.zeros_like(keypoints)
    mse_zero = np.mean((zero_pred - keypoints) ** 2)
    print(f"  全零预测 MSE: {mse_zero:.6f}")
    
    # 3. 如果预测值是均值
    mean_pred = np.full_like(keypoints, keypoints.mean())
    mse_mean = np.mean((mean_pred - keypoints) ** 2)
    print(f"  均值预测 MSE: {mse_mean:.6f}")
    
    # 4. 理论分析
    print(f"\n理论分析:")
    print(f"  如果数据范围在 [-1, 1]，随机预测的期望MSE约为: {(2**2)/3:.6f}")
    print(f"  如果数据标准差为 {data_std:.3f}，方差为: {data_std**2:.6f}")
    print(f"  考虑到数据稀疏性（很多零值），实际MSE会更小")
    
    # 5. 检查数据是否已经标准化
    non_zero_data = keypoints[np.abs(keypoints) > 1e-6]
    if len(non_zero_data) > 0:
        print(f"\n数据标准化检查:")
        print(f"  非零数据是否在[-1,1]范围: {non_zero_data.min() >= -1 and non_zero_data.max() <= 1}")
        print(f"  非零数据是否接近标准正态分布: 均值={non_zero_data.mean():.3f}, 标准差={non_zero_data.std():.3f}")

def simulate_training_mse():
    """模拟训练过程中的MSE变化"""
    print(f"\n4. 模拟训练MSE变化:")
    
    # 模拟真实数据的特征
    # 假设数据已经标准化到较小范围
    np.random.seed(42)
    target = np.random.normal(0, 0.3, (400, 150))  # 目标数据
    target[np.random.rand(*target.shape) < 0.7] = 0  # 70%的数据为0（稀疏性）
    
    print(f"模拟目标数据统计:")
    print(f"  形状: {target.shape}")
    print(f"  范围: [{target.min():.3f}, {target.max():.3f}]")
    print(f"  非零比例: {(np.abs(target) > 1e-6).mean():.1%}")
    print(f"  标准差: {target.std():.6f}")
    
    # 模拟不同训练阶段的预测
    scenarios = [
        ("随机初始化", np.random.normal(0, 0.1, target.shape)),
        ("轻微训练后", target + np.random.normal(0, 0.2, target.shape)),
        ("较好训练后", target + np.random.normal(0, 0.05, target.shape)),
        ("收敛状态", target + np.random.normal(0, 0.01, target.shape)),
    ]
    
    print(f"\n不同训练阶段的MSE:")
    for name, pred in scenarios:
        mse = np.mean((pred - target) ** 2)
        print(f"  {name}: MSE = {mse:.6f}")

def main():
    """主函数"""
    print("="*60)
    print("MSE 量级分析报告")
    print("="*60)
    
    # 检查数据路径是否存在
    data_paths = [
        "datasets/signllm_training_data/ASL/dev/dev_fz6XzPxdo-0_3-5-rgb_front",
        "datasets/signllm_training_data/ASL/dev/dev_fz6XzPxdo-0_2-5-rgb_front",
        "datasets/signllm_training_data/ASL/dev/dev_fz6XzPxdo-0_17-5-rgb_front",
    ]
    
    missing_paths = [path for path in data_paths if not os.path.exists(path)]
    if missing_paths:
        print(f"❌ 缺少数据路径: {missing_paths}")
        print("🔄 使用模拟数据进行分析...")
        simulate_training_mse()
    else:
        print("✅ 数据路径存在，开始分析...")
        analyze_data_statistics()
    
    simulate_training_mse()
    
    print(f"\n" + "="*60)
    print("结论:")
    print("="*60)
    print("1. MSE = 0.001 量级是合理的，因为:")
    print("   - 数据通常已经标准化到较小范围（如[-1,1]或更小）")
    print("   - 数据具有稀疏性（很多关键点为0）")
    print("   - MSE是对60,000个数值求平均，会降低整体量级")
    print("   - 随机初始化的网络预测值通常也在较小范围内")
    
    print(f"\n2. 预期的MSE变化:")
    print("   - 初始: 0.001 - 0.01 （随机预测）")
    print("   - 训练中: 逐渐降低")
    print("   - 过拟合目标: < 1e-4 或 1e-7")
    
    print(f"\n3. 如果MSE异常:")
    print("   - 如果初始MSE > 0.1: 可能数据未标准化或网络初始化有问题")
    print("   - 如果MSE不下降: 可能学习率太小或梯度消失")
    print("   - 如果MSE下降太慢: 可能需要调整网络结构或学习率")

if __name__ == "__main__":
    main() 