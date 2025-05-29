#!/usr/bin/env python3
"""
骨架数据可视化脚本
可视化处理后的SignLLM格式骨架数据
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import cv2

# 骨架连接定义（基于50个关键点的骨架模型）
SKELETON_CONNECTIONS = [
    # 上身连接
    (0, 1),   # 头-颈
    (1, 2),   # 颈-左肩
    (1, 5),   # 颈-右肩
    (2, 3),   # 左肩-左肘
    (3, 4),   # 左肘-左腕
    (5, 6),   # 右肩-右肘
    (6, 7),   # 右肘-右腕
    
    # 左手连接（简化版）
    (8, 9), (9, 10), (10, 11),     # 拇指
    (8, 12), (12, 13), (13, 14),   # 食指
    (8, 15), (15, 16), (16, 17),   # 中指
    (8, 18), (18, 19), (19, 20),   # 无名指
    (8, 21), (21, 22), (22, 23),   # 小指
    
    # 右手连接（简化版）
    (29, 30), (30, 31), (31, 32),  # 拇指
    (29, 33), (33, 34), (34, 35),  # 食指
    (29, 36), (36, 37), (37, 38),  # 中指
    (29, 39), (39, 40), (40, 41),  # 无名指
    (29, 42), (42, 43), (43, 44),  # 小指
]

# 关键点颜色定义
KEYPOINT_COLORS = {
    'head': 'red',
    'upper_body': 'blue',
    'left_hand': 'green',
    'right_hand': 'orange'
}


def load_skeleton_data(data_dir, sample_name, language="ASL", split="dev"):
    """加载骨架数据"""
    sample_path = Path(data_dir) / language / split / sample_name
    pose_file = sample_path / "pose.json"
    text_file = sample_path / "text.txt"
    
    if not pose_file.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
    
    # 加载姿态数据
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    
    # 加载文本
    text = ""
    if text_file.exists():
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    
    return pose_data, text


def extract_3d_keypoints(pose_data):
    """从姿态数据中提取3D关键点"""
    frames = []
    
    for frame in pose_data['poses']:
        # 重构150维数据为50个3D点
        keypoints_3d = []
        
        # 上身关键点 (8个点 * 3坐标 = 24维)
        pose_kpts = frame['pose_keypoints_2d']
        for i in range(0, len(pose_kpts), 3):
            if i + 2 < len(pose_kpts):
                keypoints_3d.append([pose_kpts[i], pose_kpts[i+1], pose_kpts[i+2]])
        
        # 左手关键点 (21个点 * 3坐标 = 63维)
        left_hand = frame['hand_left_keypoints_2d']
        for i in range(0, len(left_hand), 3):
            if i + 2 < len(left_hand):
                keypoints_3d.append([left_hand[i], left_hand[i+1], left_hand[i+2]])
        
        # 右手关键点 (21个点 * 3坐标 = 63维)
        right_hand = frame['hand_right_keypoints_2d']
        for i in range(0, len(right_hand), 3):
            if i + 2 < len(right_hand):
                keypoints_3d.append([right_hand[i], right_hand[i+1], right_hand[i+2]])
        
        # 确保有50个关键点
        while len(keypoints_3d) < 50:
            keypoints_3d.append([0.0, 0.0, 0.0])
        
        frames.append(np.array(keypoints_3d[:50]))
    
    return np.array(frames)


def plot_skeleton_frame(ax, keypoints, frame_idx, text=""):
    """绘制单帧骨架"""
    ax.clear()
    
    # 设置坐标轴
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 绘制关键点
    x, y, z = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
    
    # 不同部位使用不同颜色
    # 头部和上身 (0-7)
    ax.scatter(x[:8], y[:8], z[:8], c='red', s=50, alpha=0.8, label='Upper Body')
    
    # 左手 (8-28)
    ax.scatter(x[8:29], y[8:29], z[8:29], c='green', s=30, alpha=0.8, label='Left Hand')
    
    # 右手 (29-49)
    ax.scatter(x[29:50], y[29:50], z[29:50], c='orange', s=30, alpha=0.8, label='Right Hand')
    
    # 绘制骨架连接
    for connection in SKELETON_CONNECTIONS:
        if connection[0] < len(keypoints) and connection[1] < len(keypoints):
            point1, point2 = keypoints[connection[0]], keypoints[connection[1]]
            # 只绘制有效连接（非零点）
            if not (np.allclose(point1, 0) or np.allclose(point2, 0)):
                ax.plot([point1[0], point2[0]], 
                       [point1[1], point2[1]], 
                       [point1[2], point2[2]], 'b-', alpha=0.6, linewidth=1)
    
    ax.set_title(f'Frame {frame_idx}\nText: {text[:50]}...', fontsize=10)
    ax.legend()


def create_skeleton_animation(keypoints_sequence, text, output_file=None):
    """创建骨架动画"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(frame_idx):
        if frame_idx < len(keypoints_sequence):
            plot_skeleton_frame(ax, keypoints_sequence[frame_idx], frame_idx, text)
        return ax,
    
    # 创建动画
    anim = animation.FuncAnimation(
        fig, animate, frames=len(keypoints_sequence), 
        interval=100, blit=False, repeat=True
    )
    
    if output_file:
        print(f"保存动画到: {output_file}")
        anim.save(output_file, writer='pillow', fps=10)
    
    plt.tight_layout()
    plt.show()
    
    return anim


def plot_data_statistics(keypoints_sequence, text):
    """绘制数据统计图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 关键点轨迹
    ax1 = axes[0, 0]
    for i in range(min(8, keypoints_sequence.shape[1])):  # 只显示前8个关键点
        trajectory = keypoints_sequence[:, i, :]
        ax1.plot(trajectory[:, 0], label=f'Point {i} X')
    ax1.set_title('关键点X坐标轨迹')
    ax1.set_xlabel('帧数')
    ax1.set_ylabel('X坐标')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 数据有效性分析
    ax2 = axes[0, 1]
    valid_points_per_frame = []
    for frame in keypoints_sequence:
        valid_count = 0
        for point in frame:
            if not np.allclose(point, 0):
                valid_count += 1
        valid_points_per_frame.append(valid_count)
    
    ax2.plot(valid_points_per_frame, 'g-', linewidth=2)
    ax2.set_title('每帧有效关键点数量')
    ax2.set_xlabel('帧数')
    ax2.set_ylabel('有效点数')
    ax2.grid(True)
    
    # 3. 运动幅度分析
    ax3 = axes[1, 0]
    motion_magnitude = []
    for i in range(1, len(keypoints_sequence)):
        diff = keypoints_sequence[i] - keypoints_sequence[i-1]
        magnitude = np.sqrt(np.sum(diff**2, axis=1))
        motion_magnitude.append(np.mean(magnitude))
    
    ax3.plot(motion_magnitude, 'r-', linewidth=2)
    ax3.set_title('帧间运动幅度')
    ax3.set_xlabel('帧数')
    ax3.set_ylabel('平均运动幅度')
    ax3.grid(True)
    
    # 4. 数据分布热图
    ax4 = axes[1, 1]
    # 计算每个关键点的活跃度
    activity_map = np.zeros((50, 3))
    for i in range(50):
        for j in range(3):  # x, y, z
            coords = keypoints_sequence[:, i, j]
            activity_map[i, j] = np.std(coords) if not np.all(coords == 0) else 0
    
    im = ax4.imshow(activity_map.T, cmap='viridis', aspect='auto')
    ax4.set_title('关键点活跃度热图')
    ax4.set_xlabel('关键点索引')
    ax4.set_ylabel('坐标轴 (X, Y, Z)')
    plt.colorbar(im, ax=ax4)
    
    plt.suptitle(f'骨架数据分析\n文本: {text[:100]}...', fontsize=14)
    plt.tight_layout()
    plt.show()


def compare_multiple_samples(data_dir, sample_names, language="ASL", split="dev"):
    """比较多个样本的骨架数据"""
    fig = plt.figure(figsize=(5*len(sample_names), 10))
    
    for i, sample_name in enumerate(sample_names):
        try:
            pose_data, text = load_skeleton_data(data_dir, sample_name, language, split)
            keypoints = extract_3d_keypoints(pose_data)
            
            # 绘制第一帧 (3D)
            ax1 = fig.add_subplot(2, len(sample_names), i+1, projection='3d')
            if keypoints.shape[0] > 0:
                plot_skeleton_frame(ax1, keypoints[0], 0, text)
            ax1.set_title(f'样本 {i+1} - 第1帧\n{text[:30]}...')
            
            # 绘制运动轨迹 (2D)
            ax2 = fig.add_subplot(2, len(sample_names), len(sample_names) + i + 1)
            if keypoints.shape[0] > 1:
                # 绘制手腕轨迹
                left_wrist = keypoints[:, 4, :]  # 左手腕
                right_wrist = keypoints[:, 7, :]  # 右手腕
                
                ax2.plot(left_wrist[:, 0], left_wrist[:, 1], 'g-', label='左手腕', linewidth=2)
                ax2.plot(right_wrist[:, 0], right_wrist[:, 1], 'r-', label='右手腕', linewidth=2)
                ax2.set_xlabel('X坐标')
                ax2.set_ylabel('Y坐标')
                ax2.set_title(f'手腕运动轨迹')
                ax2.legend()
                ax2.grid(True)
            
        except Exception as e:
            print(f"处理样本 {sample_name} 时出错: {e}")
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="可视化骨架数据")
    parser.add_argument("--data_dir", type=str, default="./datasets/signllm_data_final",
                       help="SignLLM格式数据目录")
    parser.add_argument("--sample", type=str, default=None,
                       help="要可视化的样本名称")
    parser.add_argument("--language", type=str, default="ASL",
                       help="语言代码")
    parser.add_argument("--split", type=str, default="dev",
                       help="数据集分割")
    parser.add_argument("--mode", type=str, default="animation",
                       choices=["animation", "statistics", "compare"],
                       help="可视化模式")
    parser.add_argument("--save_gif", type=str, default=None,
                       help="保存动画为GIF文件")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # 如果没有指定样本，自动选择第一个
    if args.sample is None:
        sample_dir = data_dir / args.language / args.split
        if sample_dir.exists():
            samples = [d.name for d in sample_dir.iterdir() if d.is_dir()]
            if samples:
                args.sample = samples[0]
                print(f"自动选择样本: {args.sample}")
            else:
                print("❌ 没有找到样本数据")
                return
        else:
            print(f"❌ 数据目录不存在: {sample_dir}")
            return
    
    try:
        if args.mode == "animation":
            # 动画模式
            pose_data, text = load_skeleton_data(args.data_dir, args.sample, args.language, args.split)
            keypoints_sequence = extract_3d_keypoints(pose_data)
            
            print(f"📊 数据信息:")
            print(f"样本: {args.sample}")
            print(f"文本: {text}")
            print(f"帧数: {len(keypoints_sequence)}")
            print(f"关键点数: {keypoints_sequence.shape[1]}")
            
            # 创建动画
            anim = create_skeleton_animation(keypoints_sequence, text, args.save_gif)
            
        elif args.mode == "statistics":
            # 统计模式
            pose_data, text = load_skeleton_data(args.data_dir, args.sample, args.language, args.split)
            keypoints_sequence = extract_3d_keypoints(pose_data)
            plot_data_statistics(keypoints_sequence, text)
            
        elif args.mode == "compare":
            # 比较模式
            sample_dir = data_dir / args.language / args.split
            samples = [d.name for d in sample_dir.iterdir() if d.is_dir()][:4]  # 最多比较4个样本
            compare_multiple_samples(args.data_dir, samples, args.language, args.split)
    
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 