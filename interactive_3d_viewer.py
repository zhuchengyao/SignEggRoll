#!/usr/bin/env python3
"""
交互式3D骨架查看器 - 支持旋转、缩放、多角度观察
类似Matlab的3D查看器功能，现在支持真实数据正面视角的平面图和动画生成
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib
from pathlib import Path
import sys
import json
from PIL import Image
import imageio

# 设置交互式后端
matplotlib.use('TkAgg')  # 支持交互式操作
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']  # 优先使用SimHei支持中文，后备DejaVu Sans
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from signllm_model import SignLLM, ModelConfig, CONFIG


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


class Interactive3DViewer:
    """交互式3D骨架查看器"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.current_frame = 0
        self.pose_sequence = None
        self.text = ""
        self.is_2d_mode = False  # 新增：是否为2D平面模式
        
    def load_model_and_generate(self):
        """加载真实数据集进行可视化"""
        print("🚀 交互式3D骨架查看器")
        print("=" * 50)
        
        # 直接读取原始数据，获得真实的帧数
        print("📚 直接加载原始数据...")
        
        data_dir = Path("datasets/signllm_data_complete/ASL/dev")
        if not data_dir.exists():
            print("❌ 数据目录不存在，请检查路径")
            raise RuntimeError("数据目录不存在")
        
        # 获取所有样本目录
        sample_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if len(sample_dirs) == 0:
            print("❌ 未找到样本数据")
            raise RuntimeError("未找到样本数据")
        
        print(f"📊 找到 {len(sample_dirs)} 个样本")
        
        # 选择前几个样本进行可视化
        num_samples = min(5, len(sample_dirs))
        all_poses = []
        
        for i, sample_dir in enumerate(sample_dirs[:num_samples]):
            try:
                # 读取文本
                text_file = sample_dir / "text.txt"
                pose_file = sample_dir / "pose.json"
                
                if not (text_file.exists() and pose_file.exists()):
                    print(f"⚠️  样本 {sample_dir.name} 缺少必要文件，跳过")
                    continue
                
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                # 读取原始姿态数据
                with open(pose_file, 'r') as f:
                    pose_data = json.load(f)
                
                poses = pose_data.get("poses", [])
                actual_frames = pose_data.get("num_frames", len(poses))
                
                print(f"   样本 {i+1}: {actual_frames} 帧")
                
                # 转换姿态数据为3D坐标
                pose_3d_list = []
                for pose in poses[:actual_frames]:  # 只取实际帧数
                    # 提取关键点坐标
                    pose_kpts = pose.get("pose_keypoints_2d", [])
                    left_hand_kpts = pose.get("hand_left_keypoints_2d", [])
                    right_hand_kpts = pose.get("hand_right_keypoints_2d", [])
                    
                    # 构建50个关键点的3D坐标
                    joints_3d = np.zeros((50, 3))
                    
                    # 上身关键点 (0-7): 选择重要的8个上身点
                    upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 7]
                    for j, idx in enumerate(upper_body_indices):
                        if idx * 3 + 2 < len(pose_kpts):
                            joints_3d[j] = [pose_kpts[idx*3], pose_kpts[idx*3+1], pose_kpts[idx*3+2]]
                    
                    # 左手关键点 (8-28): 21个点
                    for j in range(21):
                        if j * 3 + 2 < len(left_hand_kpts):
                            joints_3d[8 + j] = [left_hand_kpts[j*3], left_hand_kpts[j*3+1], left_hand_kpts[j*3+2]]
                    
                    # 右手关键点 (29-49): 21个点  
                    for j in range(21):
                        if j * 3 + 2 < len(right_hand_kpts):
                            joints_3d[29 + j] = [right_hand_kpts[j*3], right_hand_kpts[j*3+1], right_hand_kpts[j*3+2]]
                    
                    pose_3d_list.append(joints_3d)
                
                if len(pose_3d_list) == 0:
                    print(f"⚠️  样本 {sample_dir.name} 无有效姿态数据，跳过")
                    continue
                
                # 转换为numpy数组 [actual_frames, 50, 3]
                pose_3d = np.array(pose_3d_list)
                
                all_poses.append({
                    'text': text,
                    'poses': pose_3d,
                    'quality': 1.0,  # 真实数据没有质量分数
                    'sample_idx': i,
                    'sample_name': sample_dir.name,
                    'actual_frames': actual_frames
                })
                
                print(f"✅ 样本 {i+1}: '{text[:50]}...' -> 形状: {pose_3d.shape}")
                
            except Exception as e:
                print(f"❌ 处理样本 {sample_dir.name} 失败: {e}")
                continue
        
        if len(all_poses) == 0:
            print("❌ 未能加载任何有效样本")
            raise RuntimeError("未能加载任何有效样本")
        
        return all_poses
    
    def create_interactive_viewer(self, pose_data_list):
        """创建交互式3D查看器"""
        self.pose_data_list = pose_data_list
        self.current_data_idx = 0
        self.current_frame = 0
        
        # 创建图形窗口
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle("交互式3D骨架查看器\n使用鼠标拖拽旋转，滚轮缩放，键盘切换帧/文本", fontsize=14)
        
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
        print("  ↑ ↓ : 切换文本")
        print("  P   : 切换2D平面视图")
        print("  A   : 生成2D动画 (当前文本)")
        print("  3   : 生成俯视3D动画 (elev=90°, azim=0°)")
        print("  R: 重置视角")
        print("  S: 保存当前视图")
        print("  Q: 退出")
        print("  鼠标拖拽: 旋转视角")
        print("  鼠标滚轮: 缩放")
        
        plt.show()
    
    def update_plot(self):
        """更新图像"""
        if self.is_2d_mode:
            self.update_2d_plot()
        else:
            self.update_3d_plot()
    
    def update_3d_plot(self):
        """更新3D图像"""
        self.ax.clear()
        
        # 如果当前是2D轴，需要重新创建3D轴
        if not hasattr(self.ax, 'zaxis'):
            self.ax.remove()
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 获取当前数据
        current_data = self.pose_data_list[self.current_data_idx]
        poses = current_data['poses']
        text = current_data['text']
        
        # 确保帧索引有效
        if self.current_frame >= poses.shape[0]:
            self.current_frame = 0
        
        # 获取当前帧的关节数据
        joints = poses[self.current_frame]  # [50, 3]
        
        x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
        
        # 绘制关节点 - 不同部位用不同颜色和大小
        # 上身 (0-7)
        self.ax.scatter(x[:8], y[:8], z[:8], c='red', s=60, alpha=0.9, 
                       label='Upper Body', edgecolors='darkred', linewidth=1)
        
        # 左手 (8-28)
        self.ax.scatter(x[8:29], y[8:29], z[8:29], c='blue', s=30, alpha=0.8, 
                       label='Left Hand', edgecolors='darkblue', linewidth=0.5)
        
        # 右手 (29-49)
        self.ax.scatter(x[29:50], y[29:50], z[29:50], c='green', s=30, alpha=0.8, 
                       label='Right Hand', edgecolors='darkgreen', linewidth=0.5)
        
        # 绘制骨架连接
        for start, end in REAL_CONNECTIONS:
            if start < len(joints) and end < len(joints):
                if not (np.allclose(joints[start], 0) or np.allclose(joints[end], 0)):
                    # 根据连接类型使用不同颜色
                    if start < 8 and end < 8:  # 上身连接
                        color = 'red'
                        linewidth = 2
                    elif 8 <= start < 29 and 8 <= end < 29:  # 左手连接
                        color = 'blue'
                        linewidth = 1
                    elif 29 <= start < 50 and 29 <= end < 50:  # 右手连接
                        color = 'green'
                        linewidth = 1
                    else:  # 跨部位连接
                        color = 'black'
                        linewidth = 2
                    
                    self.ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 
                               color=color, alpha=0.7, linewidth=linewidth)
        
        # 标注重要关节点
        important_joints = [0, 1, 2, 5, 8, 29]  # 头、颈、肩、手腕
        for i in important_joints:
            if i < len(joints):
                self.ax.text(x[i], y[i], z[i], f'{i}', fontsize=8, color='black', 
                           fontweight='bold')
        
        # 设置坐标轴
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.set_zlabel('Z', fontsize=12)
        
        # 设置标题
        sample_idx = current_data.get('sample_idx', self.current_data_idx)
        title = f"3D视图 (真实数据): '{text[:40]}...'\n帧 {self.current_frame+1}/{poses.shape[0]} | 样本 {sample_idx+1}/{len(self.pose_data_list)}"
        self.ax.set_title(title, fontsize=12, pad=20)
        
        # 设置相等的坐标轴比例
        ranges = [x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]
        max_range = max(ranges) / 2.0
        center = [x.mean(), y.mean(), z.mean()]
        
        self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
        self.ax.set_ylim(center[1] - max_range, center[1] + max_range)
        self.ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        # 添加图例
        self.ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # 设置网格
        self.ax.grid(True, alpha=0.3)
        
        # 刷新显示
        self.fig.canvas.draw()
    
    def update_2d_plot(self):
        """更新2D平面图像（真实数据正面视角：上面是X轴，左边是Y轴）"""
        self.ax.clear()
        
        # 如果当前是3D轴，需要重新创建2D轴
        if hasattr(self.ax, 'zaxis'):
            self.ax.remove()
            self.ax = self.fig.add_subplot(111)
        
        # 获取当前数据
        current_data = self.pose_data_list[self.current_data_idx]
        poses = current_data['poses']
        text = current_data['text']
        
        # 确保帧索引有效
        if self.current_frame >= poses.shape[0]:
            self.current_frame = 0
        
        # 获取当前帧的关节数据
        joints = poses[self.current_frame]  # [50, 3]
        
        # 正面视角坐标
        plot_x = joints[:, 0]   # X轴：左右方向
        plot_y = joints[:, 1]   # Y轴：上下方向
        plot_z = joints[:, 2]   # Z轴用于颜色编码
        
        # 绘制关节点 - 不同部位用不同颜色和大小
        # 上身 (0-7)
        self.ax.scatter(plot_x[:8], plot_y[:8], c='red', s=80, alpha=0.9,
                       label='Upper Body (上身)', edgecolors='darkred', linewidth=1)
        
        # 左手 (8-28)
        self.ax.scatter(plot_x[8:29], plot_y[8:29], c='blue', s=40, alpha=0.8,
                       label='Left Hand (左手)', edgecolors='darkblue', linewidth=0.5)
        
        # 右手 (29-49)  
        self.ax.scatter(plot_x[29:50], plot_y[29:50], c='green', s=40, alpha=0.8,
                       label='Right Hand (右手)', edgecolors='darkgreen', linewidth=0.5)
        
        # 绘制骨架连接
        for start, end in REAL_CONNECTIONS:
            if start < len(joints) and end < len(joints):
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
                    
                    self.ax.plot([plot_x[start], plot_x[end]], [plot_y[start], plot_y[end]], 
                               color=color, alpha=alpha, linewidth=linewidth)
        
        # 标注重要关节点
        important_joints = [0, 1, 2, 5, 8, 29]  # 头、颈、肩、手腕
        joint_labels = ['头部', '颈部', '左肩', '右肩', '左腕', '右腕']
        
        for i, label in zip(important_joints, joint_labels):
            if i < len(joints) and not np.allclose(joints[i], 0):
                self.ax.annotate(f'{i}\n{label}', (plot_x[i], plot_y[i]), 
                               fontsize=9, color='black', fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
                               ha='center', va='center')
        
        # 设置坐标轴（旋转180度后）
        self.ax.set_xlabel('X轴 (左←→右)', fontsize=12)
        self.ax.set_ylabel('Y轴 (下↓↑上)', fontsize=12)
        
        # 设置标题
        sample_idx = current_data.get('sample_idx', self.current_data_idx)
        title = f"2D正面视图 (真实数据): '{text[:40]}...'\n帧 {self.current_frame+1}/{poses.shape[0]} | 样本 {sample_idx+1}/{len(self.pose_data_list)}"
        self.ax.set_title(title, fontsize=12, pad=20)
        
        # 设置相等的坐标轴比例
        all_points = joints[~np.all(joints == 0, axis=1)]  # 排除零点
        if len(all_points) > 0:
            plot_x_valid = all_points[:, 0]  # X轴坐标
            plot_y_valid = all_points[:, 1]  # Y轴坐标
            
            x_range = plot_x_valid.max() - plot_x_valid.min()
            y_range = plot_y_valid.max() - plot_y_valid.min()
            max_range = max(x_range, y_range) / 2.0 if max(x_range, y_range) > 0 else 0.1
            
            center_x = plot_x_valid.mean()
            center_y = plot_y_valid.mean()
            
            self.ax.set_xlim(center_x - max_range, center_x + max_range)
            self.ax.set_ylim(center_y - max_range, center_y + max_range)
        
        # 设置等比例
        self.ax.set_aspect('equal', adjustable='box')
        
        # 添加图例
        self.ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # 设置网格
        self.ax.grid(True, alpha=0.3)
        
        # 添加坐标轴说明
        self.ax.text(0.98, 0.02, '坐标系：正面视角\n右→X轴  上↑Y轴', 
                    transform=self.ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                    ha='right', va='bottom')
        
        # 刷新显示
        self.fig.canvas.draw()
    
    def generate_2d_animation(self):
        """生成当前文本的2D动画"""
        current_data = self.pose_data_list[self.current_data_idx]
        poses = current_data['poses']
        text = current_data['text']
        
        print(f"\n🎬 开始生成2D动画...")
        print(f"   文本: '{text}'")
        print(f"   总帧数: {poses.shape[0]}")
        
        # 创建输出目录
        anim_dir = Path("sign_animations")
        anim_dir.mkdir(exist_ok=True)
        
        # 安全的文件名
        safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_text = safe_text.replace(' ', '_')
        
        # 临时图片目录
        temp_dir = anim_dir / f"temp_{safe_text}"
        temp_dir.mkdir(exist_ok=True)
        
        # 计算全局坐标范围（用于保持动画中的一致性）
        all_joints = poses.reshape(-1, 3)  # [frames*50, 3]
        all_points = all_joints[~np.all(all_joints == 0, axis=1)]  # 排除零点
        
        if len(all_points) > 0:
            global_plot_x = all_points[:, 0]   # X轴：左右方向
            global_plot_y = all_points[:, 1]   # Y轴：上下方向
            
            global_x_range = global_plot_x.max() - global_plot_x.min()
            global_y_range = global_plot_y.max() - global_plot_y.min()
            global_max_range = max(global_x_range, global_y_range) / 2.0 * 1.1  # 稍微放大一点
            
            global_center_x = global_plot_x.mean()
            global_center_y = global_plot_y.mean()
            
            global_xlim = (global_center_x - global_max_range, global_center_x + global_max_range)
            global_ylim = (global_center_y - global_max_range, global_center_y + global_max_range)
        else:
            global_xlim = (-1, 1)
            global_ylim = (-1, 1)
        
        # 生成每一帧图片
        frame_files = []
        
        for frame_idx in range(poses.shape[0]):
            print(f"   生成帧 {frame_idx+1}/{poses.shape[0]}", end='\r')
            
            # 创建新的图形
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor('white')
            
            # 获取当前帧数据
            joints = poses[frame_idx]  # [50, 3]
            
            # 正面视角坐标
            plot_x = joints[:, 0]   # X轴：左右方向
            plot_y = joints[:, 1]   # Y轴：上下方向
            plot_z = joints[:, 2]   # Z轴：前后方向
            
            # 绘制关节点
            ax.scatter(plot_x[:8], plot_y[:8], plot_z[:8], c='red', s=100, alpha=0.9,
                      label='Upper Body', edgecolors='darkred', linewidth=2)
            ax.scatter(plot_x[8:29], plot_y[8:29], plot_z[8:29], c='blue', s=60, alpha=0.8,
                      label='Left Hand', edgecolors='darkblue', linewidth=1)
            ax.scatter(plot_x[29:50], plot_y[29:50], plot_z[29:50], c='green', s=60, alpha=0.8,
                      label='Right Hand', edgecolors='darkgreen', linewidth=1)
            
            # 绘制骨架连接
            for start, end in REAL_CONNECTIONS:
                if start < len(joints) and end < len(joints):
                    if not (np.allclose(joints[start], 0) or np.allclose(joints[end], 0)):
                        if start < 8 and end < 8:  # 上身连接
                            color, linewidth, alpha = 'red', 4, 0.9
                        elif 8 <= start < 29 and 8 <= end < 29:  # 左手连接
                            color, linewidth, alpha = 'blue', 2, 0.8
                        elif 29 <= start < 50 and 29 <= end < 50:  # 右手连接
                            color, linewidth, alpha = 'green', 2, 0.8
                        else:  # 跨部位连接
                            color, linewidth, alpha = 'black', 4, 0.9
                        
                        ax.plot([plot_x[start], plot_x[end]], [plot_y[start], plot_y[end]], 
                               color=color, alpha=alpha, linewidth=linewidth)
            
            # 设置固定的坐标范围
            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)
            ax.set_aspect('equal', adjustable='box')
            
            # 设置标题和标签
            ax.set_xlabel('X轴 (左←→右)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Y轴 (下↓↑上)', fontsize=14, fontweight='bold')
            ax.set_title(f"手语动画 (真实数据): '{text[:30]}...'\n帧 {frame_idx+1}/{poses.shape[0]}", 
                        fontsize=16, fontweight='bold', pad=20)
            
            # 设置网格和图例
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=12)
            
            # 添加帧编号水印
            ax.text(0.98, 0.98, f"Frame {frame_idx+1}", 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   ha='right', va='top')
            
            # 保存当前帧
            frame_file = temp_dir / f"frame_{frame_idx:03d}.png"
            plt.savefig(frame_file, dpi=120, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            frame_files.append(str(frame_file))
        
        print(f"\n   ✅ 完成帧生成")
        
        # 生成GIF动画
        print(f"   🎞️  合成GIF动画...")
        
        # 读取所有帧
        images = []
        for frame_file in frame_files:
            img = Image.open(frame_file)
            images.append(img)
        
        # 保存GIF
        gif_path = anim_dir / f"{safe_text}_animation_front_view.gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=200,  # 每帧200ms
            loop=0  # 无限循环
        )
        
        print(f"   ✅ GIF动画保存: {gif_path}")
        
        # 生成MP4动画（如果有imageio）
        try:
            print(f"   🎥 合成MP4动画...")
            mp4_path = anim_dir / f"{safe_text}_animation_front_view.mp4"
            
            with imageio.get_writer(mp4_path, fps=5) as writer:
                for frame_file in frame_files:
                    image = imageio.imread(frame_file)
                    writer.append_data(image)
            
            print(f"   ✅ MP4动画保存: {mp4_path}")
            
        except Exception as e:
            print(f"   ⚠️  MP4生成失败: {e}")
        
        # 清理临时文件
        print(f"   🧹 清理临时文件...")
        for frame_file in frame_files:
            Path(frame_file).unlink()
        temp_dir.rmdir()
        
        print(f"\n🎉 动画生成完成！")
        print(f"📁 输出目录: {anim_dir}")
        print(f"📽️  动画文件: {gif_path.name}")
    
    def generate_3d_angle_animation(self, elev=20, azim=45):
        """生成固定3D角度的动画"""
        current_data = self.pose_data_list[self.current_data_idx]
        poses = current_data['poses']
        text = current_data['text']
        
        print(f"\n🎬 开始生成3D固定角度动画...")
        print(f"   文本: '{text}'")
        print(f"   总帧数: {poses.shape[0]}")
        print(f"   视角: 仰角={elev}°, 方位角={azim}°")
        
        # 创建输出目录
        anim_dir = Path("sign_animations")
        anim_dir.mkdir(exist_ok=True)
        
        # 安全的文件名
        safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_text = safe_text.replace(' ', '_')
        
        # 临时图片目录
        temp_dir = anim_dir / f"temp_{safe_text}_3d_{elev}_{azim}"
        temp_dir.mkdir(exist_ok=True)
        
        # 计算全局坐标范围（用于保持动画中的一致性）
        all_joints = poses.reshape(-1, 3)  # [frames*50, 3]
        all_points = all_joints[~np.all(all_joints == 0, axis=1)]  # 排除零点
        
        if len(all_points) > 0:
            global_x_range = all_points[:, 0].max() - all_points[:, 0].min()
            global_y_range = all_points[:, 1].max() - all_points[:, 1].min()
            global_z_range = all_points[:, 2].max() - all_points[:, 2].min()
            global_max_range = max(global_x_range, global_y_range, global_z_range) / 2.0 * 1.1
            
            global_center = [all_points[:, 0].mean(), all_points[:, 1].mean(), all_points[:, 2].mean()]
            
            global_xlim = (global_center[0] - global_max_range, global_center[0] + global_max_range)
            global_ylim = (global_center[1] - global_max_range, global_center[1] + global_max_range)
            global_zlim = (global_center[2] - global_max_range, global_center[2] + global_max_range)
        else:
            global_xlim = (-1, 1)
            global_ylim = (-1, 1)
            global_zlim = (-1, 1)
        
        # 生成每一帧图片
        frame_files = []
        
        for frame_idx in range(poses.shape[0]):
            print(f"   生成帧 {frame_idx+1}/{poses.shape[0]}", end='\r')
            
            # 创建新的图形
            fig = plt.figure(figsize=(12, 9))
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111, projection='3d')
            
            # 获取当前帧数据
            joints = poses[frame_idx]  # [50, 3]
            
            # 正面视角坐标
            plot_x = joints[:, 0]   # X轴：左右方向
            plot_y = joints[:, 1]   # Y轴：上下方向
            plot_z = joints[:, 2]   # Z轴：前后方向
            
            # 绘制关节点
            ax.scatter(plot_x[:8], plot_y[:8], plot_z[:8], c='red', s=80, alpha=0.9,
                      label='Upper Body', edgecolors='darkred', linewidth=2)
            ax.scatter(plot_x[8:29], plot_y[8:29], plot_z[8:29], c='blue', s=50, alpha=0.8,
                      label='Left Hand', edgecolors='darkblue', linewidth=1)
            ax.scatter(plot_x[29:50], plot_y[29:50], plot_z[29:50], c='green', s=50, alpha=0.8,
                      label='Right Hand', edgecolors='darkgreen', linewidth=1)
            
            # 绘制骨架连接
            for start, end in REAL_CONNECTIONS:
                if start < len(joints) and end < len(joints):
                    if not (np.allclose(joints[start], 0) or np.allclose(joints[end], 0)):
                        if start < 8 and end < 8:  # 上身连接
                            color, linewidth, alpha = 'red', 3, 0.9
                        elif 8 <= start < 29 and 8 <= end < 29:  # 左手连接
                            color, linewidth, alpha = 'blue', 2, 0.8
                        elif 29 <= start < 50 and 29 <= end < 50:  # 右手连接
                            color, linewidth, alpha = 'green', 2, 0.8
                        else:  # 跨部位连接
                            color, linewidth, alpha = 'black', 3, 0.9
                        
                        ax.plot([plot_x[start], plot_x[end]], [plot_y[start], plot_y[end]], [plot_z[start], plot_z[end]], 
                               color=color, alpha=alpha, linewidth=linewidth)
            
            # 设置固定的坐标范围和视角
            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)
            ax.set_zlim(global_zlim)
            ax.view_init(elev=elev, azim=azim)  # 设置固定视角
            
            # 设置标题和标签
            ax.set_xlabel('X轴', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y轴', fontsize=12, fontweight='bold')
            ax.set_zlabel('Z轴', fontsize=12, fontweight='bold')
            ax.set_title(f"3D手语动画 (真实数据): '{text[:30]}...'\n帧 {frame_idx+1}/{poses.shape[0]} | 视角: {elev}°,{azim}°", 
                        fontsize=14, fontweight='bold', pad=20)
            
            # 设置网格和图例
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=10)
            
            # 添加帧编号水印
            ax.text2D(0.98, 0.98, f"Frame {frame_idx+1}", 
                     transform=ax.transAxes, fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                     ha='right', va='top')
            
            # 保存当前帧
            frame_file = temp_dir / f"frame_{frame_idx:03d}.png"
            plt.savefig(frame_file, dpi=120, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            frame_files.append(str(frame_file))
        
        print(f"\n   ✅ 完成帧生成")
        
        # 生成GIF动画
        print(f"   🎞️  合成GIF动画...")
        
        # 读取所有帧
        images = []
        for frame_file in frame_files:
            img = Image.open(frame_file)
            images.append(img)
        
        # 保存GIF
        gif_path = anim_dir / f"{safe_text}_3d_elev{elev}_azim{azim}.gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=200,  # 每帧200ms
            loop=0  # 无限循环
        )
        
        print(f"   ✅ GIF动画保存: {gif_path}")
        
        # 生成MP4动画
        try:
            print(f"   🎥 合成MP4动画...")
            mp4_path = anim_dir / f"{safe_text}_3d_elev{elev}_azim{azim}.mp4"
            
            with imageio.get_writer(mp4_path, fps=5) as writer:
                for frame_file in frame_files:
                    image = imageio.imread(frame_file)
                    writer.append_data(image)
            
            print(f"   ✅ MP4动画保存: {mp4_path}")
            
        except Exception as e:
            print(f"   ⚠️  MP4生成失败: {e}")
        
        # 清理临时文件
        print(f"   🧹 清理临时文件...")
        for frame_file in frame_files:
            Path(frame_file).unlink()
        temp_dir.rmdir()
        
        print(f"\n🎉 3D角度动画生成完成！")
        print(f"📁 输出目录: {anim_dir}")
        print(f"📽️  动画文件: {gif_path.name}")
    
    def on_key_press(self, event):
        """键盘事件处理"""
        current_data = self.pose_data_list[self.current_data_idx]
        max_frames = current_data['poses'].shape[0]
        
        if event.key == 'left':  # 上一帧
            self.current_frame = (self.current_frame - 1) % max_frames
            self.update_plot()
        elif event.key == 'right':  # 下一帧
            self.current_frame = (self.current_frame + 1) % max_frames
            self.update_plot()
        elif event.key == 'up':  # 上一个文本
            self.current_data_idx = (self.current_data_idx - 1) % len(self.pose_data_list)
            self.current_frame = 0  # 重置帧
            self.update_plot()
        elif event.key == 'down':  # 下一个文本
            self.current_data_idx = (self.current_data_idx + 1) % len(self.pose_data_list)
            self.current_frame = 0  # 重置帧
            self.update_plot()
        elif event.key == 'p':  # 切换2D/3D模式
            self.is_2d_mode = not self.is_2d_mode
            mode_str = "2D平面视图" if self.is_2d_mode else "3D立体视图"
            print(f"🔄 切换到 {mode_str}")
            self.update_plot()
        elif event.key == 'a':  # 生成2D动画
            print(f"�� 开始生成当前文本的2D动画...")
            try:
                self.generate_2d_animation()
            except Exception as e:
                print(f"❌ 动画生成失败: {e}")
                import traceback
                traceback.print_exc()
        elif event.key == '3':  # 生成俯视角3D动画
            print(f"🎬 生成俯视角3D动画 (elev=90°, azim=0°)...")
            try:
                self.generate_3d_angle_animation(elev=90, azim=0)
            except Exception as e:
                print(f"❌ 动画生成失败: {e}")
        elif event.key == 'r':  # 重置视角
            if not self.is_2d_mode:
                self.ax.view_init(elev=20, azim=45)
            self.fig.canvas.draw()
        elif event.key == 's':  # 保存当前视图
            save_dir = Path("interactive_3d_views")
            save_dir.mkdir(exist_ok=True)
            mode_suffix = "2d" if self.is_2d_mode else "3d"
            filename = f"view_{mode_suffix}_{self.current_data_idx}_{self.current_frame}.png"
            save_path = save_dir / filename
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 保存视图: {save_path}")
        elif event.key == 'q':  # 退出
            plt.close(self.fig)
    
    def add_control_instructions(self):
        """添加控制说明文本"""
        instruction_text = """
控制说明:
← → 切换帧    ↑ ↓ 切换文本
P 切换2D/3D   A 生成2D动画
3 俯视3D       R 重置视角
S 保存视图    Q 退出
鼠标拖拽旋转, 滚轮缩放
        """
        self.fig.text(0.02, 0.02, instruction_text, fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))


def main():
    """主函数"""
    try:
        viewer = Interactive3DViewer()
        
        # 加载真实数据
        pose_data_list = viewer.load_model_and_generate()
        
        # 启动交互式查看器
        print(f"\n🎮 启动交互式3D查看器 (真实数据)...")
        print(f"   加载了 {len(pose_data_list)} 个真实样本的姿态数据")
        
        viewer.create_interactive_viewer(pose_data_list)
        
    except KeyboardInterrupt:
        print("\n👋 用户退出")
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请确保已安装 tkinter: conda install tk 或 apt-get install python3-tk")


if __name__ == "__main__":
    main() 