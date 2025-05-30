#!/usr/bin/env python3
"""
SignLLM推理结果3D可视化器 - 支持输入文本生成手语姿态并3D可视化
基于interactive_3d_viewer.py的骨架显示结构
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
import argparse

# 设置交互式后端
matplotlib.use('TkAgg')
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from signllm_model import SignLLM, ModelConfig, CONFIG

# 真实的50关节点骨架连接（与interactive_3d_viewer.py相同）
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


class SignLLMInference3DViewer:
    """SignLLM推理结果3D可视化器"""
    
    def __init__(self, model_path: str = None, model_size: str = CONFIG.model_size):
        self.fig = None
        self.ax = None
        self.current_frame = 0
        self.is_2d_mode = False
        self.inference_results = []
        self.current_result_idx = 0
        
        # 加载模型
        self.model = self.load_model(model_path, model_size)
        
    def load_model(self, model_path: str, model_size: str):
        """加载训练好的SignLLM模型"""
        print("🚀 SignLLM推理结果3D可视化器")
        print("=" * 50)
        
        # 设置模型配置
        global CONFIG
        CONFIG = ModelConfig(model_size)
        
        print(f"📦 加载模型 ({model_size})...")
        model = SignLLM(languages=["ASL"])
        
        if model_path and Path(model_path).exists():
            try:
                # 先运行一次前向传播来创建动态层
                dummy_text = ["hello"]
                with torch.no_grad():
                    model(dummy_text, "ASL", max_length=16)
                
                checkpoint = torch.load(model_path, map_location='cpu')
                state_dict = checkpoint['model_state_dict']
                model.load_state_dict(state_dict)
                
                epoch = checkpoint.get('epoch', 'Unknown')
                loss = checkpoint.get('loss', 'Unknown')
                print(f"✅ 成功加载训练模型: Epoch {epoch}, Loss {loss}")
                
            except Exception as e:
                print(f"⚠️  加载模型失败，使用随机初始化: {e}")
        else:
            print(f"⚠️  未找到模型文件，使用随机初始化模型")
        
        model.eval()
        return model
    
    def generate_poses(self, texts: list, language: str = "ASL", mode: str = "mlsf", max_length: int = None):
        """生成手语姿态"""
        print(f"\n🎯 开始推理生成...")
        
        if max_length is None:
            max_length = CONFIG.default_max_frames
        
        results = []
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                print(f"   正在处理: '{text[:50]}...'")
                
                try:
                    # 模型推理
                    pred_poses, quality_scores = self.model(
                        texts=[text],
                        language=language,
                        mode=mode,
                        max_length=max_length
                    )
                    
                    # 转换为numpy
                    pose_data = pred_poses[0].detach().cpu().numpy()  # [seq_len, 150]
                    pose_3d = pose_data.reshape(-1, 50, 3)  # [seq_len, 50, 3]
                    
                    # 计算质量分数
                    avg_quality = quality_scores.mean().item()
                    
                    results.append({
                        'text': text,
                        'poses': pose_3d,
                        'quality': avg_quality,
                        'mode': mode,
                        'language': language,
                        'actual_frames': pose_3d.shape[0]
                    })
                    
                    print(f"   ✅ 生成成功: {pose_3d.shape[0]} 帧, 质量: {avg_quality:.4f}")
                    
                except Exception as e:
                    print(f"   ❌ 生成失败: {e}")
                    continue
        
        return results
    
    def create_interactive_viewer(self, inference_results):
        """创建交互式3D查看器"""
        self.inference_results = inference_results
        self.current_result_idx = 0
        self.current_frame = 0
        
        if len(inference_results) == 0:
            print("❌ 没有推理结果可以显示")
            return
        
        # 创建图形窗口
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("SignLLM推理结果3D可视化器\n使用鼠标拖拽旋转，滚轮缩放，键盘切换帧/文本", fontsize=14)
        
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
        print("  ↑ ↓ : 切换推理结果")
        print("  P   : 切换2D平面视图")
        print("  A   : 生成2D动画")
        print("  3   : 生成3D俯视动画")
        print("  I   : 添加新的推理文本")
        print("  R   : 重置视角")
        print("  S   : 保存当前视图")
        print("  Q   : 退出")
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
        
        # 获取当前推理结果
        current_result = self.inference_results[self.current_result_idx]
        poses = current_result['poses']
        text = current_result['text']
        quality = current_result['quality']
        mode = current_result['mode']
        
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
        title = f"3D推理结果 ({mode.upper()}): '{text[:40]}...'\n帧 {self.current_frame+1}/{poses.shape[0]} | 结果 {self.current_result_idx+1}/{len(self.inference_results)} | 质量: {quality:.4f}"
        self.ax.set_title(title, fontsize=12, pad=20)
        
        # 设置相等的坐标轴比例
        ranges = [x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]
        max_range = max(ranges) / 2.0 if max(ranges) > 0 else 0.1
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
        """更新2D平面图像（正面视角）"""
        self.ax.clear()
        
        # 如果当前是3D轴，需要重新创建2D轴
        if hasattr(self.ax, 'zaxis'):
            self.ax.remove()
            self.ax = self.fig.add_subplot(111)
        
        # 获取当前推理结果
        current_result = self.inference_results[self.current_result_idx]
        poses = current_result['poses']
        text = current_result['text']
        quality = current_result['quality']
        mode = current_result['mode']
        
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
        
        # 设置坐标轴
        self.ax.set_xlabel('X轴 (左←→右)', fontsize=12)
        self.ax.set_ylabel('Y轴 (下↓↑上)', fontsize=12)
        
        # 设置标题
        title = f"2D推理结果 ({mode.upper()}): '{text[:40]}...'\n帧 {self.current_frame+1}/{poses.shape[0]} | 结果 {self.current_result_idx+1}/{len(self.inference_results)} | 质量: {quality:.4f}"
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
        """生成当前推理结果的2D动画"""
        current_result = self.inference_results[self.current_result_idx]
        poses = current_result['poses']
        text = current_result['text']
        mode = current_result['mode']
        quality = current_result['quality']
        
        print(f"\n🎬 开始生成推理结果2D动画...")
        print(f"   文本: '{text}'")
        print(f"   模式: {mode.upper()}")
        print(f"   质量: {quality:.4f}")
        print(f"   总帧数: {poses.shape[0]}")
        
        # 创建输出目录
        anim_dir = Path("inference_animations")
        anim_dir.mkdir(exist_ok=True)
        
        # 安全的文件名
        safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_text = safe_text.replace(' ', '_')[:30]  # 限制长度
        
        # 临时图片目录
        temp_dir = anim_dir / f"temp_{safe_text}_{mode}"
        temp_dir.mkdir(exist_ok=True)
        
        # 计算全局坐标范围
        all_joints = poses.reshape(-1, 3)
        all_points = all_joints[~np.all(all_joints == 0, axis=1)]
        
        if len(all_points) > 0:
            global_plot_x = all_points[:, 0]
            global_plot_y = all_points[:, 1]
            
            global_x_range = global_plot_x.max() - global_plot_x.min()
            global_y_range = global_plot_y.max() - global_plot_y.min()
            global_max_range = max(global_x_range, global_y_range) / 2.0 * 1.1
            
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
            joints = poses[frame_idx]
            
            # 正面视角坐标
            plot_x = joints[:, 0]
            plot_y = joints[:, 1]
            
            # 绘制关节点
            ax.scatter(plot_x[:8], plot_y[:8], c='red', s=100, alpha=0.9,
                      label='Upper Body', edgecolors='darkred', linewidth=2)
            ax.scatter(plot_x[8:29], plot_y[8:29], c='blue', s=60, alpha=0.8,
                      label='Left Hand', edgecolors='darkblue', linewidth=1)
            ax.scatter(plot_x[29:50], plot_y[29:50], c='green', s=60, alpha=0.8,
                      label='Right Hand', edgecolors='darkgreen', linewidth=1)
            
            # 绘制骨架连接
            for start, end in REAL_CONNECTIONS:
                if start < len(joints) and end < len(joints):
                    if not (np.allclose(joints[start], 0) or np.allclose(joints[end], 0)):
                        if start < 8 and end < 8:
                            color, linewidth, alpha = 'red', 4, 0.9
                        elif 8 <= start < 29 and 8 <= end < 29:
                            color, linewidth, alpha = 'blue', 2, 0.8
                        elif 29 <= start < 50 and 29 <= end < 50:
                            color, linewidth, alpha = 'green', 2, 0.8
                        else:
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
            ax.set_title(f"SignLLM推理结果 ({mode.upper()}): '{text[:30]}...'\n帧 {frame_idx+1}/{poses.shape[0]} | 质量: {quality:.4f}", 
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
        
        images = []
        for frame_file in frame_files:
            img = Image.open(frame_file)
            images.append(img)
        
        # 保存GIF
        gif_path = anim_dir / f"{safe_text}_{mode}_inference.gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=200,
            loop=0
        )
        
        print(f"   ✅ GIF动画保存: {gif_path}")
        
        # 生成MP4动画
        try:
            print(f"   🎥 合成MP4动画...")
            mp4_path = anim_dir / f"{safe_text}_{mode}_inference.mp4"
            
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
        
        print(f"\n🎉 推理结果动画生成完成！")
        print(f"📁 输出目录: {anim_dir}")
        print(f"📽️  动画文件: {gif_path.name}")
    
    def add_new_inference(self):
        """添加新的推理文本"""
        print("\n📝 输入新的文本进行推理:")
        try:
            new_text = input("请输入文本 (按Enter确认): ").strip()
            if new_text:
                print(f"开始推理: '{new_text}'")
                new_results = self.generate_poses([new_text])
                if new_results:
                    self.inference_results.extend(new_results)
                    self.current_result_idx = len(self.inference_results) - 1
                    self.current_frame = 0
                    self.update_plot()
                    print(f"✅ 添加成功，当前显示新结果")
                else:
                    print("❌ 推理失败")
            else:
                print("❌ 输入为空")
        except Exception as e:
            print(f"❌ 输入失败: {e}")
    
    def on_key_press(self, event):
        """键盘事件处理"""
        if len(self.inference_results) == 0:
            return
            
        current_result = self.inference_results[self.current_result_idx]
        max_frames = current_result['poses'].shape[0]
        
        if event.key == 'left':  # 上一帧
            self.current_frame = (self.current_frame - 1) % max_frames
            self.update_plot()
        elif event.key == 'right':  # 下一帧
            self.current_frame = (self.current_frame + 1) % max_frames
            self.update_plot()
        elif event.key == 'up':  # 上一个推理结果
            self.current_result_idx = (self.current_result_idx - 1) % len(self.inference_results)
            self.current_frame = 0
            self.update_plot()
        elif event.key == 'down':  # 下一个推理结果
            self.current_result_idx = (self.current_result_idx + 1) % len(self.inference_results)
            self.current_frame = 0
            self.update_plot()
        elif event.key == 'p':  # 切换2D/3D模式
            self.is_2d_mode = not self.is_2d_mode
            mode_str = "2D平面视图" if self.is_2d_mode else "3D立体视图"
            print(f"🔄 切换到 {mode_str}")
            self.update_plot()
        elif event.key == 'a':  # 生成2D动画
            print(f"🎬 开始生成当前推理结果的2D动画...")
            try:
                self.generate_2d_animation()
            except Exception as e:
                print(f"❌ 动画生成失败: {e}")
                import traceback
                traceback.print_exc()
        elif event.key == 'i':  # 添加新推理
            self.add_new_inference()
        elif event.key == 'r':  # 重置视角
            if not self.is_2d_mode:
                self.ax.view_init(elev=20, azim=45)
            self.fig.canvas.draw()
        elif event.key == 's':  # 保存当前视图
            save_dir = Path("inference_views")
            save_dir.mkdir(exist_ok=True)
            mode_suffix = "2d" if self.is_2d_mode else "3d"
            filename = f"inference_{mode_suffix}_{self.current_result_idx}_{self.current_frame}.png"
            save_path = save_dir / filename
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 保存视图: {save_path}")
        elif event.key == 'q':  # 退出
            plt.close(self.fig)
    
    def add_control_instructions(self):
        """添加控制说明文本"""
        instruction_text = """
控制说明:
← → 切换帧      ↑ ↓ 切换结果
P 切换2D/3D     A 生成2D动画
I 添加新推理    R 重置视角
S 保存视图      Q 退出
鼠标拖拽旋转, 滚轮缩放
        """
        self.fig.text(0.02, 0.02, instruction_text, fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SignLLM推理结果3D可视化器")
    parser.add_argument("--model_path", type=str, default="checkpoints/eggroll_train/epoch_10.pth",
                       help="训练模型路径")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small", "medium", "large"],
                       help="模型大小")
    parser.add_argument("--texts", nargs="+", 
                       default=["Hello, how are you?", "Nice to meet you", "Thank you very much"],
                       help="推理文本列表")
    parser.add_argument("--language", type=str, default="ASL", help="目标语言")
    parser.add_argument("--mode", type=str, default="mlsf", choices=["mlsf", "prompt2langgloss"],
                       help="推理模式")
    parser.add_argument("--max_length", type=int, default=None, help="最大生成长度")
    
    args = parser.parse_args()
    
    try:
        # 创建可视化器
        viewer = SignLLMInference3DViewer(args.model_path, args.model_size)
        
        # 生成推理结果
        print(f"📝 推理文本:")
        for i, text in enumerate(args.texts, 1):
            print(f"   {i}. {text}")
        
        inference_results = viewer.generate_poses(
            args.texts, 
            args.language, 
            args.mode, 
            args.max_length
        )
        
        if len(inference_results) == 0:
            print("❌ 没有成功的推理结果")
            return
        
        # 启动交互式查看器
        print(f"\n🎮 启动推理结果3D可视化器...")
        print(f"   推理模式: {args.mode.upper()}")
        print(f"   生成了 {len(inference_results)} 个推理结果")
        
        viewer.create_interactive_viewer(inference_results)
        
    except KeyboardInterrupt:
        print("\n👋 用户退出")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 