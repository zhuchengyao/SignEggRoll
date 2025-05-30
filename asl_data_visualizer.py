import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class ASLDataVisualizer:
    def __init__(self, data_path):
        """
        初始化 ASL 数据可视化器
        
        Args:
            data_path: 数据目录路径（包含 pose.json 和 text.txt）
        """
        self.data_path = data_path
        self.pose_data = None
        self.text_data = None
        self.frames = []
        
        # OpenPose 骨骼连接定义
        self.pose_connections = [
            (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),  # 上半身
            (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),  # 下半身
            (0, 1), (0, 14), (0, 15), (14, 16), (15, 17)  # 头部和眼睛
        ]
        
        # 手部关键点连接（21个点）
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
        ]
        
    def load_data(self):
        """加载姿态和文本数据"""
        # 加载文本数据
        text_path = os.path.join(self.data_path, 'text.txt')
        with open(text_path, 'r', encoding='utf-8') as f:
            self.text_data = f.read().strip()
        
        # 加载姿态数据
        pose_path = os.path.join(self.data_path, 'pose.json')
        with open(pose_path, 'r') as f:
            self.pose_data = json.load(f)
        
        print(f"文本内容: {self.text_data}")
        print(f"总帧数: {len(self.pose_data['poses'])}")
        
        return self.pose_data, self.text_data
    
    def analyze_data_structure(self):
        """分析数据结构"""
        if not self.pose_data:
            self.load_data()
        
        poses = self.pose_data['poses']
        first_frame = poses[0]
        
        print("\n=== 数据结构分析 ===")
        print(f"总帧数: {len(poses)}")
        print(f"每帧包含的数据:")
        
        for key, value in first_frame.items():
            if isinstance(value, list):
                points_count = len(value) // 3  # 每个关键点有 x, y, confidence
                print(f"  - {key}: {len(value)} 个值 ({points_count} 个关键点)")
                
                # 检查是否有有效数据（非零值）
                non_zero_count = sum(1 for x in value if abs(x) > 1e-6)
                print(f"    有效数据点: {non_zero_count}/{len(value)}")
        
        # 分析数据分布
        self._analyze_keypoints_distribution(poses)
        
    def _analyze_keypoints_distribution(self, poses):
        """分析关键点数据分布"""
        print("\n=== 关键点数据分布分析 ===")
        
        # 分析身体关键点
        pose_coords = []
        for pose in poses:
            coords = pose['pose_keypoints_2d']
            # 提取 x, y 坐标（忽略置信度）
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 3)]
            pose_coords.extend(points)
        
        pose_coords = np.array(pose_coords)
        valid_poses = pose_coords[np.any(np.abs(pose_coords) > 1e-6, axis=1)]
        
        print(f"身体关键点范围:")
        if len(valid_poses) > 0:
            print(f"  X 范围: [{valid_poses[:, 0].min():.3f}, {valid_poses[:, 0].max():.3f}]")
            print(f"  Y 范围: [{valid_poses[:, 1].min():.3f}, {valid_poses[:, 1].max():.3f}]")
        
        # 分析手部关键点
        self._analyze_hand_keypoints(poses, 'hand_left_keypoints_2d', '左手')
        self._analyze_hand_keypoints(poses, 'hand_right_keypoints_2d', '右手')
    
    def _analyze_hand_keypoints(self, poses, hand_key, hand_name):
        """分析手部关键点"""
        hand_coords = []
        valid_frames = 0
        
        for pose in poses:
            coords = pose[hand_key]
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 3)]
            
            # 检查是否有有效数据
            if any(abs(x) > 1e-6 or abs(y) > 1e-6 for x, y in points):
                valid_frames += 1
                hand_coords.extend(points)
        
        if hand_coords:
            hand_coords = np.array(hand_coords)
            valid_hands = hand_coords[np.any(np.abs(hand_coords) > 1e-6, axis=1)]
            
            print(f"{hand_name}关键点:")
            print(f"  有效帧数: {valid_frames}/{len(poses)}")
            if len(valid_hands) > 0:
                print(f"  X 范围: [{valid_hands[:, 0].min():.3f}, {valid_hands[:, 0].max():.3f}]")
                print(f"  Y 范围: [{valid_hands[:, 1].min():.3f}, {valid_hands[:, 1].max():.3f}]")
        else:
            print(f"{hand_name}关键点: 无有效数据")
    
    def extract_keypoints(self, frame_data):
        """从单帧数据中提取关键点坐标"""
        # 身体关键点 (18个点)
        pose_coords = frame_data['pose_keypoints_2d']
        pose_points = np.array([[pose_coords[i], pose_coords[i+1], pose_coords[i+2]] 
                               for i in range(0, len(pose_coords), 3)])
        
        # 左手关键点 (21个点)
        left_hand_coords = frame_data['hand_left_keypoints_2d']
        left_hand_points = np.array([[left_hand_coords[i], left_hand_coords[i+1], left_hand_coords[i+2]] 
                                    for i in range(0, len(left_hand_coords), 3)])
        
        # 右手关键点 (21个点)
        right_hand_coords = frame_data['hand_right_keypoints_2d']
        right_hand_points = np.array([[right_hand_coords[i], right_hand_coords[i+1], right_hand_coords[i+2]] 
                                     for i in range(0, len(right_hand_coords), 3)])
        
        return pose_points, left_hand_points, right_hand_points
    
    def create_static_plot(self, frame_idx=0, save_path=None):
        """创建静态可视化图"""
        if not self.pose_data:
            self.load_data()
        
        frame_data = self.pose_data['poses'][frame_idx]
        pose_points, left_hand_points, right_hand_points = self.extract_keypoints(frame_data)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 绘制身体姿态
        self._plot_pose(ax1, pose_points, "Body Pose")
        
        # 绘制左手
        self._plot_hand(ax2, left_hand_points, "Left Hand", color='blue')
        
        # 绘制右手
        self._plot_hand(ax3, right_hand_points, "Right Hand", color='red')
        
        plt.suptitle(f'Frame {frame_idx}: {self.text_data[:50]}...', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"静态图保存到: {save_path}")
        
        plt.show()
        
        return fig
    
    def _plot_pose(self, ax, points, title):
        """绘制身体姿态"""
        # 绘制关键点 - 只要坐标不为0就显示
        valid_points = points[(points[:, 0] != 0) | (points[:, 1] != 0)]
        if len(valid_points) > 0:
            ax.scatter(valid_points[:, 0], -valid_points[:, 1], c='red', s=50)
        
        # 绘制骨骼连接 - 只要两个点的坐标都不为0就连接
        for connection in self.pose_connections:
            if (connection[0] < len(points) and connection[1] < len(points) and
                (points[connection[0], 0] != 0 or points[connection[0], 1] != 0) and
                (points[connection[1], 0] != 0 or points[connection[1], 1] != 0)):
                x_coords = [points[connection[0], 0], points[connection[1], 0]]
                y_coords = [-points[connection[0], 1], -points[connection[1], 1]]
                ax.plot(x_coords, y_coords, 'b-', linewidth=2)
        
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_hand(self, ax, points, title, color='blue'):
        """绘制手部关键点"""
        # 绘制关键点 - 只要坐标不为0就显示
        valid_points = points[(points[:, 0] != 0) | (points[:, 1] != 0)]
        if len(valid_points) > 0:
            ax.scatter(valid_points[:, 0], -valid_points[:, 1], c=color, s=30)
        
        # 绘制手指连接 - 只要两个点的坐标都不为0就连接
        for connection in self.hand_connections:
            if (connection[0] < len(points) and connection[1] < len(points) and
                (points[connection[0], 0] != 0 or points[connection[0], 1] != 0) and
                (points[connection[1], 0] != 0 or points[connection[1], 1] != 0)):
                x_coords = [points[connection[0], 0], points[connection[1], 0]]
                y_coords = [-points[connection[0], 1], -points[connection[1], 1]]
                ax.plot(x_coords, y_coords, color=color, linewidth=1.5)
        
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def create_animation(self, output_path='asl_animation.gif', frame_interval=100, max_frames=None):
        """创建动画"""
        if not self.pose_data:
            self.load_data()
        
        poses = self.pose_data['poses']
        if max_frames:
            poses = poses[:max_frames]
        
        print(f"创建动画，总帧数: {len(poses)}")
        
        # 设置图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 计算所有帧的坐标范围
        all_coords = []
        for pose in poses:
            pose_points, left_hand_points, right_hand_points = self.extract_keypoints(pose)
            
            for points in [pose_points, left_hand_points, right_hand_points]:
                valid_points = points[points[:, 2] > 0.1]
                if len(valid_points) > 0:
                    all_coords.extend(valid_points[:, :2])
        
        if all_coords:
            all_coords = np.array(all_coords)
            x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
            y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
            
            # 添加边距
            margin = 0.1
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(-(y_max + margin), -(y_min - margin))
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'ASL Sign Language Animation: {self.text_data}', fontsize=14, pad=20)
        
        def animate(frame_idx):
            ax.clear()
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(-(y_max + margin), -(y_min - margin))
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            pose_points, left_hand_points, right_hand_points = self.extract_keypoints(poses[frame_idx])
            
            # 绘制身体
            self._draw_skeleton(ax, pose_points, self.pose_connections, 'red', 'blue', 50, 2)
            
            # 绘制左手
            self._draw_skeleton(ax, left_hand_points, self.hand_connections, 'green', 'green', 20, 1.5)
            
            # 绘制右手
            self._draw_skeleton(ax, right_hand_points, self.hand_connections, 'orange', 'orange', 20, 1.5)
            
            ax.set_title(f'ASL Sign Language Animation: {self.text_data}\nFrame {frame_idx+1}/{len(poses)}', 
                        fontsize=12, pad=20)
            
            return ax.patches + ax.lines
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(poses), 
                                     interval=frame_interval, blit=False, repeat=True)
        
        # 保存动画
        print(f"保存动画到: {output_path}")
        anim.save(output_path, writer='pillow', fps=10)
        plt.show()
        
        return anim
    
    def _draw_skeleton(self, ax, points, connections, point_color, line_color, point_size, line_width):
        """绘制骨骼结构"""
        # 绘制关键点 - 只要坐标不为0就显示
        valid_points = points[(points[:, 0] != 0) | (points[:, 1] != 0)]
        if len(valid_points) > 0:
            ax.scatter(valid_points[:, 0], -valid_points[:, 1], 
                      c=point_color, s=point_size, alpha=0.8)
        
        # 绘制连接线 - 只要两个点的坐标都不为0就连接
        for connection in connections:
            if (connection[0] < len(points) and connection[1] < len(points) and
                (points[connection[0], 0] != 0 or points[connection[0], 1] != 0) and
                (points[connection[1], 0] != 0 or points[connection[1], 1] != 0)):
                x_coords = [points[connection[0], 0], points[connection[1], 0]]
                y_coords = [-points[connection[0], 1], -points[connection[1], 1]]
                ax.plot(x_coords, y_coords, color=line_color, linewidth=line_width, alpha=0.8)

    def print_data_usage_analysis(self):
        """打印数据使用分析"""
        print("\n" + "="*60)
        print("ASL 数据集使用分析报告")
        print("="*60)
        
        print(f"\n1. 数据集概述:")
        print(f"   - 数据路径: {self.data_path}")
        print(f"   - 语义标签: {self.text_data}")
        print(f"   - 总帧数: {len(self.pose_data['poses'])}")
        
        print(f"\n2. 数据结构:")
        print(f"   - 身体关键点: 18个点 (包括头部、躯干、四肢)")
        print(f"   - 左手关键点: 21个点 (详细手指关节)")
        print(f"   - 右手关键点: 21个点 (详细手指关节)")
        print(f"   - 面部关键点: 70个点 (当前数据中为空)")
        print(f"   - 每个关键点包含: x坐标, y坐标, 置信度")
        
        print(f"\n3. 数据用途:")
        print(f"   - 手语识别: 将手语动作序列转换为文本")
        print(f"   - 手语生成: 根据文本生成对应的手语动作")
        print(f"   - 手语翻译: 多语言手语之间的转换")
        print(f"   - 动作分析: 研究手语的语法和表达模式")
        
        print(f"\n4. 数据格式特点:")
        print(f"   - OpenPose格式的2D关键点数据")
        print(f"   - 标准化坐标系 (通常在-1到1之间)")
        print(f"   - 时序数据，可用于RNN/LSTM/Transformer训练")
        print(f"   - 包含置信度信息，便于过滤低质量数据")
        
        print(f"\n5. 建议的使用方法:")
        print(f"   - 预处理: 标准化坐标，处理缺失值")
        print(f"   - 特征工程: 计算关节角度、运动速度等特征")
        print(f"   - 模型训练: 使用序列到序列模型进行训练")
        print(f"   - 数据增强: 旋转、缩放、噪声等增强方法")
        
        print("="*60)

# 示例使用
if __name__ == "__main__":
    # 指定数据路径
    data_path = "datasets/signllm_training_data/ASL/dev/dev__2u0MkRqpjA_5-5-rgb_front"
    
    # 创建可视化器
    visualizer = ASLDataVisualizer(data_path)
    
    # 加载并分析数据
    print("=== 加载数据 ===")
    visualizer.load_data()
    
    print("\n=== 数据结构分析 ===")
    visualizer.analyze_data_structure()
    
    # 打印详细分析报告
    visualizer.print_data_usage_analysis()
    
    # 创建静态图
    print("\n=== 创建静态图 ===")
    visualizer.create_static_plot(frame_idx=0, save_path='asl_static_frame.png')
    
    # 创建动画（限制帧数以加快生成速度）
    print("\n=== 创建动画 ===")
    visualizer.create_animation(output_path='asl_animation.gif', 
                               frame_interval=200, max_frames=500)