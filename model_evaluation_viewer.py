#!/usr/bin/env python3
"""
SignLLM模型评估可视化器 - 对比真实数据与推理结果
支持计算评估指标并3D可视化对比
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
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置交互式后端
matplotlib.use('TkAgg')
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from signllm_model import SignLLM, ModelConfig, CONFIG
from data_processor import MultilingualSignDataset

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


class ModelEvaluationViewer:
    """SignLLM模型评估可视化器"""
    
    def __init__(self, model_path: str = None, model_size: str = "tiny"):
        self.fig = None
        self.ax_left = None
        self.ax_right = None
        self.current_frame = 0
        self.current_sample_idx = 0
        self.is_2d_mode = False
        self.evaluation_results = []
        
        # 加载模型
        self.model = self.load_model(model_path, model_size)
        
    def load_model(self, model_path: str, model_size: str):
        """加载训练好的SignLLM模型"""
        print("🚀 SignLLM模型评估可视化器")
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
    
    def evaluate_model(self, data_dir: str, split: str = "dev", max_samples: int = 20, mode: str = "mlsf"):
        """在数据集上评估模型"""
        print(f"\n🎯 开始模型评估...")
        print(f"   数据集: {split}")
        print(f"   最大样本数: {max_samples}")
        print(f"   推理模式: {mode.upper()}")
        
        # 加载数据集
        dataset = MultilingualSignDataset(
            data_dirs={"ASL": data_dir},
            languages=["ASL"],
            split=split,
            max_sequence_length=512,
            pose_dim=150
        )
        
        if len(dataset) == 0:
            raise RuntimeError("数据集为空")
        
        # 限制样本数量
        num_samples = min(max_samples, len(dataset))
        print(f"   实际评估样本数: {num_samples}")
        
        evaluation_results = []
        total_mse = 0
        total_dtw = 0
        successful_samples = 0
        
        with torch.no_grad():
            for i in range(num_samples):
                try:
                    print(f"   处理样本 {i+1}/{num_samples}", end='\r')
                    
                    # 获取真实数据
                    sample = dataset[i]
                    text = sample['text']
                    true_poses_tensor = sample['pose_sequence']  # [seq_len, 150]
                    
                    # 转换真实数据为3D格式
                    if isinstance(true_poses_tensor, torch.Tensor):
                        true_poses_data = true_poses_tensor.detach().cpu().numpy()
                    else:
                        true_poses_data = true_poses_tensor
                    
                    # 截取有效长度（去除填充）
                    sample_length = sample.get('length', true_poses_data.shape[0])
                    true_poses_data = true_poses_data[:sample_length]
                    true_poses_3d = true_poses_data.reshape(-1, 50, 3)
                    
                    # 模型推理
                    pred_poses, quality_scores = self.model(
                        texts=[text],
                        language="ASL",
                        mode=mode,
                        max_length=true_poses_3d.shape[0]  # 使用真实长度
                    )
                    
                    # 转换预测结果
                    pred_poses_data = pred_poses[0].detach().cpu().numpy()
                    pred_poses_3d = pred_poses_data.reshape(-1, 50, 3)
                    
                    # 计算评估指标
                    metrics = self.calculate_metrics(true_poses_3d, pred_poses_3d)
                    
                    evaluation_results.append({
                        'sample_idx': i,
                        'text': text,
                        'true_poses': true_poses_3d,
                        'pred_poses': pred_poses_3d,
                        'quality_score': quality_scores.mean().item(),
                        'metrics': metrics,
                        'mode': mode
                    })
                    
                    total_mse += metrics['mse']
                    total_dtw += metrics['dtw_distance']
                    successful_samples += 1
                    
                except Exception as e:
                    print(f"\n   ❌ 样本 {i+1} 处理失败: {e}")
                    continue
        
        # 计算整体统计
        if successful_samples > 0:
            avg_mse = total_mse / successful_samples
            avg_dtw = total_dtw / successful_samples
            
            print(f"\n📊 评估完成:")
            print(f"   成功样本: {successful_samples}/{num_samples}")
            print(f"   平均MSE: {avg_mse:.6f}")
            print(f"   平均DTW: {avg_dtw:.6f}")
            print(f"   平均DTW分数: {1.0/(1.0+avg_dtw):.4f}")
        
        return evaluation_results
    
    def calculate_metrics(self, true_poses, pred_poses):
        """计算评估指标"""
        # 确保两个序列长度一致
        min_len = min(true_poses.shape[0], pred_poses.shape[0])
        true_poses_trim = true_poses[:min_len]
        pred_poses_trim = pred_poses[:min_len]
        
        # 展平为2D用于计算
        true_flat = true_poses_trim.reshape(min_len, -1)
        pred_flat = pred_poses_trim.reshape(min_len, -1)
        
        # MSE和MAE
        mse = mean_squared_error(true_flat, pred_flat)
        mae = mean_absolute_error(true_flat, pred_flat)
        rmse = np.sqrt(mse)
        
        # DTW距离
        try:
            dtw_distance, _ = fastdtw(true_flat, pred_flat, dist=euclidean)
            dtw_score = 1.0 / (1.0 + dtw_distance)
        except:
            dtw_distance = float('inf')
            dtw_score = 0.0
        
        # 姿态相似度（基于关键点距离）
        pose_similarities = []
        for i in range(min_len):
            similarity = 1.0 / (1.0 + np.linalg.norm(true_poses_trim[i] - pred_poses_trim[i]))
            pose_similarities.append(similarity)
        avg_pose_similarity = np.mean(pose_similarities)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'dtw_distance': dtw_distance,
            'dtw_score': dtw_score,
            'pose_similarity': avg_pose_similarity
        }
    
    def create_comparison_viewer(self, evaluation_results):
        """创建对比可视化界面"""
        self.evaluation_results = evaluation_results
        self.current_sample_idx = 0
        self.current_frame = 0
        
        if len(evaluation_results) == 0:
            print("❌ 没有评估结果可以显示")
            return
        
        # 创建图形窗口（左右分屏）
        self.fig = plt.figure(figsize=(20, 10))
        self.fig.suptitle("SignLLM模型评估对比 | 左侧：真实数据 | 右侧：模型预测\n键盘控制：← → 切换帧，↑ ↓ 切换样本", fontsize=16)
        
        # 创建左右子图
        if self.is_2d_mode:
            self.ax_left = self.fig.add_subplot(121)
            self.ax_right = self.fig.add_subplot(122)
        else:
            self.ax_left = self.fig.add_subplot(121, projection='3d')
            self.ax_right = self.fig.add_subplot(122, projection='3d')
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 初始绘制
        self.update_comparison_plot()
        
        # 添加控制说明
        self.add_control_instructions()
        
        print("\n🎮 控制说明:")
        print("  ← → : 切换帧")
        print("  ↑ ↓ : 切换样本")
        print("  P   : 切换2D/3D视图")
        print("  A   : 生成对比动画")
        print("  M   : 显示详细指标")
        print("  R   : 重置视角")
        print("  S   : 保存当前对比视图")
        print("  Q   : 退出")
        
        plt.show()
    
    def update_comparison_plot(self):
        """更新对比图像"""
        if self.is_2d_mode:
            self.update_2d_comparison()
        else:
            self.update_3d_comparison()
    
    def update_3d_comparison(self):
        """更新3D对比图像"""
        # 清除现有图像
        self.ax_left.clear()
        self.ax_right.clear()
        
        # 确保是3D轴
        if not hasattr(self.ax_left, 'zaxis'):
            self.ax_left.remove()
            self.ax_right.remove()
            self.ax_left = self.fig.add_subplot(121, projection='3d')
            self.ax_right = self.fig.add_subplot(122, projection='3d')
        
        # 获取当前样本数据
        current_result = self.evaluation_results[self.current_sample_idx]
        true_poses = current_result['true_poses']
        pred_poses = current_result['pred_poses']
        text = current_result['text']
        metrics = current_result['metrics']
        mode = current_result['mode']
        
        # 确保帧索引有效
        max_frames = min(true_poses.shape[0], pred_poses.shape[0])
        if self.current_frame >= max_frames:
            self.current_frame = 0
        
        # 绘制真实数据（左侧）
        self.draw_3d_skeleton(self.ax_left, true_poses[self.current_frame], "真实数据", 'blue')
        
        # 绘制预测数据（右侧）
        self.draw_3d_skeleton(self.ax_right, pred_poses[self.current_frame], "模型预测", 'red')
        
        # 设置标题
        left_title = f"真实数据\n'{text[:30]}...'\n帧 {self.current_frame+1}/{max_frames}"
        right_title = f"模型预测 ({mode.upper()})\nMSE: {metrics['mse']:.6f}\nDTW: {metrics['dtw_score']:.4f}"
        
        self.ax_left.set_title(left_title, fontsize=12, pad=20)
        self.ax_right.set_title(right_title, fontsize=12, pad=20)
        
        # 设置相同的坐标范围
        self.sync_3d_axes_limits(self.ax_left, self.ax_right, true_poses[self.current_frame], pred_poses[self.current_frame])
        
        # 刷新显示
        self.fig.canvas.draw()
    
    def update_2d_comparison(self):
        """更新2D对比图像"""
        # 清除现有图像
        self.ax_left.clear()
        self.ax_right.clear()
        
        # 确保是2D轴
        if hasattr(self.ax_left, 'zaxis'):
            self.ax_left.remove()
            self.ax_right.remove()
            self.ax_left = self.fig.add_subplot(121)
            self.ax_right = self.fig.add_subplot(122)
        
        # 获取当前样本数据
        current_result = self.evaluation_results[self.current_sample_idx]
        true_poses = current_result['true_poses']
        pred_poses = current_result['pred_poses']
        text = current_result['text']
        metrics = current_result['metrics']
        mode = current_result['mode']
        
        # 确保帧索引有效
        max_frames = min(true_poses.shape[0], pred_poses.shape[0])
        if self.current_frame >= max_frames:
            self.current_frame = 0
        
        # 绘制真实数据（左侧）
        self.draw_2d_skeleton(self.ax_left, true_poses[self.current_frame], "真实数据", 'blue')
        
        # 绘制预测数据（右侧）
        self.draw_2d_skeleton(self.ax_right, pred_poses[self.current_frame], "模型预测", 'red')
        
        # 设置标题
        left_title = f"真实数据\n'{text[:30]}...'\n帧 {self.current_frame+1}/{max_frames}"
        right_title = f"模型预测 ({mode.upper()})\nMSE: {metrics['mse']:.6f} | DTW: {metrics['dtw_score']:.4f}"
        
        self.ax_left.set_title(left_title, fontsize=12, pad=20)
        self.ax_right.set_title(right_title, fontsize=12, pad=20)
        
        # 设置相同的坐标范围
        self.sync_2d_axes_limits(self.ax_left, self.ax_right, true_poses[self.current_frame], pred_poses[self.current_frame])
        
        # 刷新显示
        self.fig.canvas.draw()
    
    def draw_3d_skeleton(self, ax, joints, title, main_color):
        """绘制3D骨架"""
        x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
        
        # 绘制关节点
        ax.scatter(x[:8], y[:8], z[:8], c=main_color, s=60, alpha=0.9, 
                  label='Upper Body', edgecolors='darkred', linewidth=1)
        ax.scatter(x[8:29], y[8:29], z[8:29], c='green', s=30, alpha=0.8, 
                  label='Left Hand', edgecolors='darkgreen', linewidth=0.5)
        ax.scatter(x[29:50], y[29:50], z[29:50], c='orange', s=30, alpha=0.8, 
                  label='Right Hand', edgecolors='darkorange', linewidth=0.5)
        
        # 绘制骨架连接
        for start, end in REAL_CONNECTIONS:
            if start < len(joints) and end < len(joints):
                if not (np.allclose(joints[start], 0) or np.allclose(joints[end], 0)):
                    if start < 8 and end < 8:  # 上身连接
                        color, linewidth = main_color, 2
                    elif 8 <= start < 29 and 8 <= end < 29:  # 左手连接
                        color, linewidth = 'green', 1
                    elif 29 <= start < 50 and 29 <= end < 50:  # 右手连接
                        color, linewidth = 'orange', 1
                    else:  # 跨部位连接
                        color, linewidth = 'black', 2
                    
                    ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 
                           color=color, alpha=0.7, linewidth=linewidth)
        
        # 设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True, alpha=0.3)
    
    def draw_2d_skeleton(self, ax, joints, title, main_color):
        """绘制2D骨架"""
        plot_x = joints[:, 0]  # X轴：左右方向
        plot_y = joints[:, 1]  # Y轴：上下方向
        
        # 绘制关节点
        ax.scatter(plot_x[:8], plot_y[:8], c=main_color, s=80, alpha=0.9,
                  label='Upper Body', edgecolors='darkred', linewidth=1)
        ax.scatter(plot_x[8:29], plot_y[8:29], c='green', s=40, alpha=0.8,
                  label='Left Hand', edgecolors='darkgreen', linewidth=0.5)
        ax.scatter(plot_x[29:50], plot_y[29:50], c='orange', s=40, alpha=0.8,
                  label='Right Hand', edgecolors='darkorange', linewidth=0.5)
        
        # 绘制骨架连接
        for start, end in REAL_CONNECTIONS:
            if start < len(joints) and end < len(joints):
                if not (np.allclose(joints[start], 0) or np.allclose(joints[end], 0)):
                    if start < 8 and end < 8:  # 上身连接
                        color, linewidth = main_color, 3
                    elif 8 <= start < 29 and 8 <= end < 29:  # 左手连接
                        color, linewidth = 'green', 1.5
                    elif 29 <= start < 50 and 29 <= end < 50:  # 右手连接
                        color, linewidth = 'orange', 1.5
                    else:  # 跨部位连接
                        color, linewidth = 'black', 3
                    
                    ax.plot([plot_x[start], plot_x[end]], [plot_y[start], plot_y[end]], 
                           color=color, alpha=0.7, linewidth=linewidth)
        
        # 设置坐标轴
        ax.set_xlabel('X轴 (左←→右)')
        ax.set_ylabel('Y轴 (下↓↑上)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
    
    def sync_3d_axes_limits(self, ax1, ax2, joints1, joints2):
        """同步3D坐标轴范围"""
        # 合并两个数据计算全局范围
        all_joints = np.vstack([joints1, joints2])
        x, y, z = all_joints[:, 0], all_joints[:, 1], all_joints[:, 2]
        
        ranges = [x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]
        max_range = max(ranges) / 2.0 if max(ranges) > 0 else 0.1
        center = [x.mean(), y.mean(), z.mean()]
        
        for ax in [ax1, ax2]:
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    def sync_2d_axes_limits(self, ax1, ax2, joints1, joints2):
        """同步2D坐标轴范围"""
        # 合并两个数据计算全局范围
        all_joints = np.vstack([joints1, joints2])
        all_points = all_joints[~np.all(all_joints == 0, axis=1)]
        
        if len(all_points) > 0:
            plot_x_valid = all_points[:, 0]
            plot_y_valid = all_points[:, 1]
            
            x_range = plot_x_valid.max() - plot_x_valid.min()
            y_range = plot_y_valid.max() - plot_y_valid.min()
            max_range = max(x_range, y_range) / 2.0 * 1.1
            
            center_x = plot_x_valid.mean()
            center_y = plot_y_valid.mean()
            
            xlim = (center_x - max_range, center_x + max_range)
            ylim = (center_y - max_range, center_y + max_range)
            
            for ax in [ax1, ax2]:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
    
    def show_detailed_metrics(self):
        """显示详细评估指标"""
        current_result = self.evaluation_results[self.current_sample_idx]
        metrics = current_result['metrics']
        text = current_result['text']
        mode = current_result['mode']
        
        print(f"\n📊 详细评估指标 - 样本 {self.current_sample_idx+1}")
        print(f"   文本: '{text[:50]}...'")
        print(f"   模式: {mode.upper()}")
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   MAE: {metrics['mae']:.6f}")
        print(f"   RMSE: {metrics['rmse']:.6f}")
        print(f"   DTW距离: {metrics['dtw_distance']:.6f}")
        print(f"   DTW分数: {metrics['dtw_score']:.4f}")
        print(f"   姿态相似度: {metrics['pose_similarity']:.4f}")
        
        # 计算全局统计
        all_mse = [r['metrics']['mse'] for r in self.evaluation_results]
        all_dtw = [r['metrics']['dtw_score'] for r in self.evaluation_results]
        
        print(f"\n🌐 全局统计:")
        print(f"   平均MSE: {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
        print(f"   平均DTW分数: {np.mean(all_dtw):.4f} ± {np.std(all_dtw):.4f}")
        print(f"   最佳MSE: {np.min(all_mse):.6f}")
        print(f"   最佳DTW分数: {np.max(all_dtw):.4f}")
    
    def generate_comparison_animation(self):
        """生成对比动画"""
        current_result = self.evaluation_results[self.current_sample_idx]
        true_poses = current_result['true_poses']
        pred_poses = current_result['pred_poses']
        text = current_result['text']
        metrics = current_result['metrics']
        mode = current_result['mode']
        
        print(f"\n🎬 开始生成对比动画...")
        print(f"   文本: '{text}'")
        print(f"   模式: {mode.upper()}")
        print(f"   MSE: {metrics['mse']:.6f}")
        
        # 创建输出目录
        anim_dir = Path("evaluation_animations")
        anim_dir.mkdir(exist_ok=True)
        
        # 安全的文件名
        safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_text = safe_text.replace(' ', '_')[:20]
        
        # 临时图片目录
        temp_dir = anim_dir / f"temp_comparison_{safe_text}_{mode}"
        temp_dir.mkdir(exist_ok=True)
        
        # 生成每一帧对比图片
        max_frames = min(true_poses.shape[0], pred_poses.shape[0])
        frame_files = []
        
        for frame_idx in range(max_frames):
            print(f"   生成帧 {frame_idx+1}/{max_frames}", end='\r')
            
            # 创建对比图形
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))
            fig.patch.set_facecolor('white')
            
            # 绘制真实数据和预测数据
            self.draw_2d_skeleton(ax_left, true_poses[frame_idx], "真实数据", 'blue')
            self.draw_2d_skeleton(ax_right, pred_poses[frame_idx], "模型预测", 'red')
            
            # 设置标题
            ax_left.set_title(f"真实数据\n帧 {frame_idx+1}/{max_frames}", fontsize=14, fontweight='bold')
            ax_right.set_title(f"模型预测 ({mode.upper()})\nMSE: {metrics['mse']:.6f}", fontsize=14, fontweight='bold')
            
            # 同步坐标范围
            self.sync_2d_axes_limits(ax_left, ax_right, true_poses[frame_idx], pred_poses[frame_idx])
            
            # 添加总标题
            fig.suptitle(f"模型评估对比: '{text[:40]}...'", fontsize=16, fontweight='bold')
            
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
        gif_path = anim_dir / f"{safe_text}_{mode}_comparison.gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=300,  # 稍慢一点便于观察
            loop=0
        )
        
        print(f"   ✅ GIF动画保存: {gif_path}")
        
        # 清理临时文件
        print(f"   🧹 清理临时文件...")
        for frame_file in frame_files:
            Path(frame_file).unlink()
        temp_dir.rmdir()
        
        print(f"\n🎉 对比动画生成完成！")
    
    def on_key_press(self, event):
        """键盘事件处理"""
        if len(self.evaluation_results) == 0:
            return
        
        current_result = self.evaluation_results[self.current_sample_idx]
        max_frames = min(current_result['true_poses'].shape[0], current_result['pred_poses'].shape[0])
        
        if event.key == 'left':  # 上一帧
            self.current_frame = (self.current_frame - 1) % max_frames
            self.update_comparison_plot()
        elif event.key == 'right':  # 下一帧
            self.current_frame = (self.current_frame + 1) % max_frames
            self.update_comparison_plot()
        elif event.key == 'up':  # 上一个样本
            self.current_sample_idx = (self.current_sample_idx - 1) % len(self.evaluation_results)
            self.current_frame = 0
            self.update_comparison_plot()
        elif event.key == 'down':  # 下一个样本
            self.current_sample_idx = (self.current_sample_idx + 1) % len(self.evaluation_results)
            self.current_frame = 0
            self.update_comparison_plot()
        elif event.key == 'p':  # 切换2D/3D模式
            self.is_2d_mode = not self.is_2d_mode
            
            # 重新创建轴
            self.ax_left.remove()
            self.ax_right.remove()
            
            if self.is_2d_mode:
                self.ax_left = self.fig.add_subplot(121)
                self.ax_right = self.fig.add_subplot(122)
            else:
                self.ax_left = self.fig.add_subplot(121, projection='3d')
                self.ax_right = self.fig.add_subplot(122, projection='3d')
            
            mode_str = "2D平面视图" if self.is_2d_mode else "3D立体视图"
            print(f"🔄 切换到 {mode_str}")
            self.update_comparison_plot()
        elif event.key == 'a':  # 生成对比动画
            print(f"🎬 开始生成当前样本的对比动画...")
            try:
                self.generate_comparison_animation()
            except Exception as e:
                print(f"❌ 动画生成失败: {e}")
        elif event.key == 'm':  # 显示详细指标
            self.show_detailed_metrics()
        elif event.key == 'r':  # 重置视角
            if not self.is_2d_mode:
                self.ax_left.view_init(elev=20, azim=45)
                self.ax_right.view_init(elev=20, azim=45)
            self.fig.canvas.draw()
        elif event.key == 's':  # 保存当前视图
            save_dir = Path("evaluation_views")
            save_dir.mkdir(exist_ok=True)
            mode_suffix = "2d" if self.is_2d_mode else "3d"
            filename = f"comparison_{mode_suffix}_{self.current_sample_idx}_{self.current_frame}.png"
            save_path = save_dir / filename
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 保存对比视图: {save_path}")
        elif event.key == 'q':  # 退出
            plt.close(self.fig)
    
    def add_control_instructions(self):
        """添加控制说明文本"""
        instruction_text = """
控制说明:
← → 切换帧       ↑ ↓ 切换样本
P 切换2D/3D      A 生成对比动画
M 显示详细指标   R 重置视角
S 保存视图       Q 退出
        """
        self.fig.text(0.02, 0.02, instruction_text, fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SignLLM模型评估可视化器")
    parser.add_argument("--model_path", type=str, default="checkpoints/eggroll_train/epoch_10.pth",
                       help="训练模型路径")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small", "medium", "large"],
                       help="模型大小")
    parser.add_argument("--data_dir", type=str, default="datasets/signllm_data_complete",
                       help="数据集目录")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"],
                       help="数据集划分")
    parser.add_argument("--max_samples", type=int, default=20, help="最大评估样本数")
    parser.add_argument("--mode", type=str, default="mlsf", choices=["mlsf", "prompt2langgloss"],
                       help="推理模式")
    
    args = parser.parse_args()
    
    try:
        # 创建评估器
        evaluator = ModelEvaluationViewer(args.model_path, args.model_size)
        
        # 执行模型评估
        evaluation_results = evaluator.evaluate_model(
            args.data_dir, 
            args.split, 
            args.max_samples,
            args.mode
        )
        
        if len(evaluation_results) == 0:
            print("❌ 没有成功的评估结果")
            return
        
        # 启动对比可视化器
        print(f"\n🎮 启动模型评估对比可视化器...")
        print(f"   评估模式: {args.mode.upper()}")
        print(f"   数据集: {args.split}")
        print(f"   成功样本: {len(evaluation_results)}")
        
        evaluator.create_comparison_viewer(evaluation_results)
        
    except KeyboardInterrupt:
        print("\n👋 用户退出")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 