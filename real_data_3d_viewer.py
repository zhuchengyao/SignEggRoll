#!/usr/bin/env python3
"""
真实数据交互式3D骨架查看器
可视化来自datasets/signllm_data_complete/ASL/dev/的真实训练数据
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from pathlib import Path
import sys
import json
import glob

# 设置交互式后端
matplotlib.use('TkAgg')  # 支持交互式操作
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# 真实的50关节点骨架连接
REAL_SKELETON_STRUCTURE = [
    # head
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


def convert_real_data_to_150d(pose_frame):
    """将真实数据转换为150维格式 (50关节点 × 3坐标)"""
    # 根据真实的数据结构分解
    pose_keypoints = pose_frame['pose_keypoints_2d']  # 8点 × 3 = 24维
    hand_left = pose_frame['hand_left_keypoints_2d']   # 21点 × 3 = 63维
    hand_right = pose_frame['hand_right_keypoints_2d'] # 21点 × 3 = 63维
    
    # 重构为50个3D点
    all_points = []
    
    # 上身关键点 (8个点)
    for i in range(0, len(pose_keypoints), 3):
        if i + 2 < len(pose_keypoints):
            all_points.append([pose_keypoints[i], pose_keypoints[i+1], pose_keypoints[i+2]])
    
    # 左手关键点 (21个点)
    for i in range(0, len(hand_left), 3):
        if i + 2 < len(hand_left):
            all_points.append([hand_left[i], hand_left[i+1], hand_left[i+2]])
    
    # 右手关键点 (21个点)
    for i in range(0, len(hand_right), 3):
        if i + 2 < len(hand_right):
            all_points.append([hand_right[i], hand_right[i+1], hand_right[i+2]])
    
    # 确保正好50个点
    while len(all_points) < 50:
        all_points.append([0.0, 0.0, 0.0])
    
    return np.array(all_points[:50])


def load_real_training_samples(max_samples=10):
    """加载多个真实训练样本"""
    data_dir = Path("datasets/signllm_data_complete/ASL/dev")
    
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return []
    
    # 获取所有样本目录
    sample_dirs = list(data_dir.glob("dev_*"))
    
    if not sample_dirs:
        print(f"❌ 没有找到训练样本")
        return []
    
    print(f"📂 找到 {len(sample_dirs)} 个训练样本，加载前 {max_samples} 个...")
    
    all_data = []
    
    for i, sample_dir in enumerate(sample_dirs[:max_samples]):
        try:
            # 读取文本
            text_file = sample_dir / "text.txt"
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # 读取姿态数据
            pose_file = sample_dir / "pose.json"
            with open(pose_file, 'r', encoding='utf-8') as f:
                pose_data = json.load(f)
            
            # 转换为50×3格式
            poses_3d = []
            for frame in pose_data['poses']:
                joints_3d = convert_real_data_to_150d(frame)
                poses_3d.append(joints_3d)
            
            poses_array = np.array(poses_3d)  # [frames, 50, 3]
            
            all_data.append({
                'text': text,
                'poses': poses_array,
                'sample_name': sample_dir.name,
                'frames': poses_array.shape[0]
            })
            
            print(f"✅ {i+1}: '{text}' -> {poses_array.shape[0]} 帧")
            
        except Exception as e:
            print(f"❌ 加载 {sample_dir.name} 失败: {e}")
            continue
    
    return all_data


class RealDataViewer:
    """真实数据交互式3D查看器"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.current_frame = 0
        self.data_list = []
        self.current_data_idx = 0
        
    def load_real_data(self):
        """加载真实训练数据"""
        print("🚀 真实数据交互式3D骨架查看器")
        print("=" * 50)
        
        # 加载真实训练样本
        self.data_list = load_real_training_samples(max_samples=15)
        
        if not self.data_list:
            print("❌ 没有可用的训练数据")
            return False
        
        print(f"\n✅ 成功加载 {len(self.data_list)} 个真实训练样本")
        return True
    
    def create_interactive_viewer(self):
        """创建交互式3D查看器"""
        if not self.data_list:
            print("❌ 没有数据可显示")
            return
        
        self.current_data_idx = 0
        self.current_frame = 0
        
        # 创建图形窗口
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("真实训练数据交互式3D骨架查看器\n使用鼠标拖拽旋转，滚轮缩放，键盘切换帧/样本", fontsize=14)
        
        # 创建3D子图
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 初始绘制
        self.update_plot()
        
        # 添加控制说明
        self.add_control_instructions()
        
        print("\n🎮 控制说明:")
        print("  ← → : 切换帧")
        print("  ↑ ↓ : 切换样本")
        print("  鼠标拖拽: 旋转视角")
        print("  鼠标滚轮: 缩放")
        print("  R: 重置视角")
        print("  S: 保存当前视图")
        print("  I: 显示样本信息")
        print("  Q: 退出")
        
        plt.show()
    
    def update_plot(self):
        """更新3D图像"""
        self.ax.clear()
        
        # 获取当前数据
        current_data = self.data_list[self.current_data_idx]
        poses = current_data['poses']
        text = current_data['text']
        sample_name = current_data['sample_name']
        
        # 确保帧索引有效
        if self.current_frame >= poses.shape[0]:
            self.current_frame = 0
        
        # 获取当前帧的关节数据
        joints = poses[self.current_frame]  # [50, 3]
        
        x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
        
        # 绘制关节点 - 不同部位用不同颜色和大小
        # 上身 (0-7)
        self.ax.scatter(x[:8], y[:8], z[:8], c='red', s=80, alpha=0.9, 
                       label='Upper Body (上身)', edgecolors='darkred', linewidth=1)
        
        # 左手 (8-28)
        self.ax.scatter(x[8:29], y[8:29], z[8:29], c='blue', s=40, alpha=0.8, 
                       label='Left Hand (左手)', edgecolors='darkblue', linewidth=0.5)
        
        # 右手 (29-49)
        self.ax.scatter(x[29:50], y[29:50], z[29:50], c='green', s=40, alpha=0.8, 
                       label='Right Hand (右手)', edgecolors='darkgreen', linewidth=0.5)
        
        # 绘制骨架连接
        for start, end in REAL_CONNECTIONS:
            if start < len(joints) and end < len(joints):
                # 跳过零点连接（可能是无效数据）
                if not (np.allclose(joints[start], 0) or np.allclose(joints[end], 0)):
                    # 根据连接类型使用不同颜色
                    if start < 8 and end < 8:  # 上身连接
                        color = 'red'
                        linewidth = 3
                        alpha = 0.8
                    elif 8 <= start < 29 and 8 <= end < 29:  # 左手连接
                        color = 'blue'
                        linewidth = 1.5
                        alpha = 0.7
                    elif 29 <= start < 50 and 29 <= end < 50:  # 右手连接
                        color = 'green'
                        linewidth = 1.5
                        alpha = 0.7
                    else:  # 跨部位连接
                        color = 'black'
                        linewidth = 3
                        alpha = 0.9
                    
                    self.ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 
                               color=color, alpha=alpha, linewidth=linewidth)
        
        # 标注重要关节点
        important_joints = [0, 1, 2, 5, 8, 29]  # 头、颈、肩、手腕
        joint_labels = ['头部', '颈部', '左肩', '右肩', '左腕', '右腕']
        
        for i, label in zip(important_joints, joint_labels):
            if i < len(joints) and not np.allclose(joints[i], 0):
                self.ax.text(x[i], y[i], z[i], f'{i}\n{label}', fontsize=9, 
                           color='black', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
        
        # 设置坐标轴
        self.ax.set_xlabel('X 坐标', fontsize=12)
        self.ax.set_ylabel('Y 坐标', fontsize=12)
        self.ax.set_zlabel('Z 坐标', fontsize=12)
        
        # 设置标题
        title = f"真实训练数据: '{text}'\n样本: {sample_name}\n帧 {self.current_frame+1}/{poses.shape[0]} | 样本 {self.current_data_idx+1}/{len(self.data_list)}"
        self.ax.set_title(title, fontsize=11, pad=20)
        
        # 设置相等的坐标轴比例
        all_points = joints[~np.all(joints == 0, axis=1)]  # 排除零点
        if len(all_points) > 0:
            ranges = [all_points[:, 0].max()-all_points[:, 0].min(), 
                     all_points[:, 1].max()-all_points[:, 1].min(), 
                     all_points[:, 2].max()-all_points[:, 2].min()]
            max_range = max(ranges) / 2.0 if max(ranges) > 0 else 0.1
            center = [all_points[:, 0].mean(), all_points[:, 1].mean(), all_points[:, 2].mean()]
            
            self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
            self.ax.set_ylim(center[1] - max_range, center[1] + max_range)
            self.ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        # 添加图例
        self.ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # 设置网格
        self.ax.grid(True, alpha=0.3)
        
        # 刷新显示
        self.fig.canvas.draw()
    
    def on_key_press(self, event):
        """键盘事件处理"""
        current_data = self.data_list[self.current_data_idx]
        max_frames = current_data['poses'].shape[0]
        
        if event.key == 'left':  # 上一帧
            self.current_frame = (self.current_frame - 1) % max_frames
            self.update_plot()
        elif event.key == 'right':  # 下一帧
            self.current_frame = (self.current_frame + 1) % max_frames
            self.update_plot()
        elif event.key == 'up':  # 上一个样本
            self.current_data_idx = (self.current_data_idx - 1) % len(self.data_list)
            self.current_frame = 0  # 重置帧
            self.update_plot()
        elif event.key == 'down':  # 下一个样本
            self.current_data_idx = (self.current_data_idx + 1) % len(self.data_list)
            self.current_frame = 0  # 重置帧
            self.update_plot()
        elif event.key == 'r':  # 重置视角
            self.ax.view_init(elev=20, azim=45)
            self.fig.canvas.draw()
        elif event.key == 's':  # 保存当前视图
            save_dir = Path("real_data_3d_views")
            save_dir.mkdir(exist_ok=True)
            filename = f"real_view_{self.current_data_idx}_{self.current_frame}.png"
            save_path = save_dir / filename
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 保存视图: {save_path}")
        elif event.key == 'i':  # 显示信息
            current_data = self.data_list[self.current_data_idx]
            print(f"\n📊 当前样本信息:")
            print(f"   文本: '{current_data['text']}'")
            print(f"   样本名: {current_data['sample_name']}")
            print(f"   总帧数: {current_data['frames']}")
            print(f"   当前帧: {self.current_frame + 1}")
            print(f"   数据形状: {current_data['poses'].shape}")
        elif event.key == 'q':  # 退出
            plt.close(self.fig)
    
    def add_control_instructions(self):
        """添加控制说明文本"""
        instruction_text = """
真实数据控制说明:
← → 切换帧    ↑ ↓ 切换样本
R 重置视角    S 保存视图    
I 样本信息    Q 退出
鼠标拖拽旋转, 滚轮缩放
        """
        self.fig.text(0.02, 0.02, instruction_text, fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))


def main():
    """主函数"""
    try:
        viewer = RealDataViewer()
        
        # 加载真实训练数据
        if not viewer.load_real_data():
            return
        
        # 启动交互式查看器
        print(f"\n🎮 启动真实数据交互式3D查看器...")
        viewer.create_interactive_viewer()
        
    except KeyboardInterrupt:
        print("\n👋 用户退出")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 