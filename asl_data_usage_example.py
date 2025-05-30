import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import glob
from typing import List, Tuple, Dict
import seaborn as sns

class ASLDataProcessor:
    """
    ASL 数据处理器 - 用于处理手语数据并为机器学习任务做准备
    """
    
    def __init__(self, dataset_path: str):
        """
        初始化数据处理器
        
        Args:
            dataset_path: 数据集根路径
        """
        self.dataset_path = dataset_path
        self.pose_sequences = []
        self.text_labels = []
        self.raw_data = []
        
        # 关键点索引定义（根据OpenPose标准）
        self.pose_keypoints_indices = {
            'nose': 0, 'neck': 1, 'right_shoulder': 2, 'right_elbow': 3,
            'right_wrist': 4, 'left_shoulder': 5, 'left_elbow': 6, 'left_wrist': 7
        }
        
        # 手部关键点：21个点，从拇指根部到小指尖
        self.hand_keypoints_names = [
            'wrist', 'thumb_1', 'thumb_2', 'thumb_3', 'thumb_4',
            'index_1', 'index_2', 'index_3', 'index_4',
            'middle_1', 'middle_2', 'middle_3', 'middle_4',
            'ring_1', 'ring_2', 'ring_3', 'ring_4',
            'pinky_1', 'pinky_2', 'pinky_3', 'pinky_4'
        ]
    
    def load_all_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        加载所有数据样本
        
        Returns:
            pose_sequences: 姿态序列列表
            text_labels: 对应的文本标签列表
        """
        print("开始加载数据...")
        
        # 获取所有数据目录
        data_dirs = glob.glob(os.path.join(self.dataset_path, "dev_*"))
        
        for i, data_dir in enumerate(data_dirs[:10]):  # 限制加载前10个样本用于演示
            print(f"加载样本 {i+1}/{min(10, len(data_dirs))}: {os.path.basename(data_dir)}")
            
            # 加载姿态数据
            pose_path = os.path.join(data_dir, 'pose.json')
            text_path = os.path.join(data_dir, 'text.txt')
            
            if os.path.exists(pose_path) and os.path.exists(text_path):
                # 读取姿态数据
                with open(pose_path, 'r') as f:
                    pose_data = json.load(f)
                
                # 读取文本标签
                with open(text_path, 'r', encoding='utf-8') as f:
                    text_label = f.read().strip()
                
                # 处理姿态序列
                sequence = self._process_pose_sequence(pose_data['poses'])
                
                if sequence is not None and len(sequence) > 0:
                    self.pose_sequences.append(sequence)
                    self.text_labels.append(text_label)
                    self.raw_data.append({
                        'path': data_dir,
                        'sequence': sequence,
                        'text': text_label,
                        'frames': len(sequence)
                    })
        
        print(f"成功加载 {len(self.pose_sequences)} 个数据样本")
        return self.pose_sequences, self.text_labels
    
    def _process_pose_sequence(self, poses: List[Dict]) -> np.ndarray:
        """
        处理单个姿态序列
        
        Args:
            poses: 姿态帧列表
            
        Returns:
            processed_sequence: 处理后的序列 [frames, features]
        """
        sequence_features = []
        
        for pose in poses:
            frame_features = self._extract_frame_features(pose)
            if frame_features is not None:
                sequence_features.append(frame_features)
        
        if len(sequence_features) == 0:
            return None
        
        return np.array(sequence_features)
    
    def _extract_frame_features(self, pose: Dict) -> np.ndarray:
        """
        从单帧中提取特征
        
        Args:
            pose: 单帧姿态数据
            
        Returns:
            features: 特征向量
        """
        # 固定特征维度
        POSE_DIM = 16  # 8个身体关键点 * 2坐标
        LEFT_HAND_DIM = 42  # 21个左手关键点 * 2坐标
        RIGHT_HAND_DIM = 42  # 21个右手关键点 * 2坐标
        
        features = []
        
        # 1. 身体关键点特征
        pose_coords = pose['pose_keypoints_2d']
        pose_points = np.array([[pose_coords[i], pose_coords[i+1], pose_coords[i+2]] 
                               for i in range(0, len(pose_coords), 3)])
        
        # 固定身体关键点特征维度
        pose_features = []
        for i in range(8):  # 确保只取前8个点
            if i < len(pose_points) and pose_points[i, 2] > 0.1:
                pose_features.extend([pose_points[i, 0], pose_points[i, 1]])
            else:
                pose_features.extend([0.0, 0.0])
        
        # 2. 左手关键点特征
        left_hand_coords = pose['hand_left_keypoints_2d']
        left_hand_points = np.array([[left_hand_coords[i], left_hand_coords[i+1], left_hand_coords[i+2]] 
                                    for i in range(0, len(left_hand_coords), 3)])
        
        left_hand_features = []
        for i in range(21):  # 确保21个手部关键点
            if i < len(left_hand_points) and left_hand_points[i, 2] > 0.1:
                left_hand_features.extend([left_hand_points[i, 0], left_hand_points[i, 1]])
            else:
                left_hand_features.extend([0.0, 0.0])
        
        # 3. 右手关键点特征
        right_hand_coords = pose['hand_right_keypoints_2d']
        right_hand_points = np.array([[right_hand_coords[i], right_hand_coords[i+1], right_hand_coords[i+2]] 
                                     for i in range(0, len(right_hand_coords), 3)])
        
        right_hand_features = []
        for i in range(21):  # 确保21个手部关键点
            if i < len(right_hand_points) and right_hand_points[i, 2] > 0.1:
                right_hand_features.extend([right_hand_points[i, 0], right_hand_points[i, 1]])
            else:
                right_hand_features.extend([0.0, 0.0])
        
        # 合并所有特征，确保维度一致
        all_features = pose_features + left_hand_features + right_hand_features
        
        # 确保特征维度为100 (16 + 42 + 42)
        assert len(all_features) == 100, f"Feature dimension mismatch: {len(all_features)}"
        
        return np.array(all_features)
    
    def extract_advanced_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        提取高级特征（运动学特征）
        
        Args:
            sequence: 原始姿态序列 [frames, features]
            
        Returns:
            advanced_features: 高级特征向量
        """
        if len(sequence) < 2:
            return np.array([])
        
        # 1. 速度特征（相邻帧的差值）
        velocity = np.diff(sequence, axis=0)
        velocity_stats = [
            np.mean(velocity, axis=0),
            np.std(velocity, axis=0),
            np.max(velocity, axis=0),
            np.min(velocity, axis=0)
        ]
        
        # 2. 加速度特征
        if len(sequence) >= 3:
            acceleration = np.diff(velocity, axis=0)
            acceleration_stats = [
                np.mean(acceleration, axis=0),
                np.std(acceleration, axis=0)
            ]
        else:
            acceleration_stats = [np.zeros_like(velocity_stats[0]), np.zeros_like(velocity_stats[0])]
        
        # 3. 序列统计特征
        sequence_stats = [
            np.mean(sequence, axis=0),
            np.std(sequence, axis=0),
            np.max(sequence, axis=0),
            np.min(sequence, axis=0)
        ]
        
        # 合并所有特征
        all_features = []
        for stats in velocity_stats + acceleration_stats + sequence_stats:
            all_features.extend(stats.flatten())
        
        return np.array(all_features)
    
    def prepare_ml_data(self, max_sequence_length: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        为机器学习准备数据
        
        Args:
            max_sequence_length: 最大序列长度
            
        Returns:
            X: 特征矩阵
            y: 标签向量（编码后的）
        """
        if not self.pose_sequences:
            self.load_all_data()
        
        # 1. 序列填充/截断
        padded_sequences = []
        for seq in self.pose_sequences:
            if len(seq) > max_sequence_length:
                # 截断
                padded_seq = seq[:max_sequence_length]
            else:
                # 填充
                padding = np.zeros((max_sequence_length - len(seq), seq.shape[1]))
                padded_seq = np.vstack([seq, padding])
            
            padded_sequences.append(padded_seq)
        
        X = np.array(padded_sequences)
        
        # 2. 标签编码（简单示例：使用文本长度作为标签）
        y = np.array([len(text.split()) for text in self.text_labels])
        
        return X, y
    
    def analyze_data_statistics(self):
        """
        分析数据统计信息
        """
        if not self.raw_data:
            self.load_all_data()
        
        print("\n" + "="*60)
        print("数据统计分析")
        print("="*60)
        
        # 序列长度统计
        sequence_lengths = [data['frames'] for data in self.raw_data]
        print(f"\n序列长度统计:")
        print(f"  平均长度: {np.mean(sequence_lengths):.2f} 帧")
        print(f"  最短序列: {np.min(sequence_lengths)} 帧")
        print(f"  最长序列: {np.max(sequence_lengths)} 帧")
        print(f"  标准差: {np.std(sequence_lengths):.2f}")
        
        # 文本长度统计
        text_lengths = [len(data['text'].split()) for data in self.raw_data]
        print(f"\n文本长度统计:")
        print(f"  平均词数: {np.mean(text_lengths):.2f} 词")
        print(f"  最短文本: {np.min(text_lengths)} 词")
        print(f"  最长文本: {np.max(text_lengths)} 词")
        
        # 特征维度统计
        if self.pose_sequences:
            feature_dim = self.pose_sequences[0].shape[1]
            print(f"\n特征维度: {feature_dim}")
        
        # 绘制统计图
        self._plot_statistics(sequence_lengths, text_lengths)
    
    def _plot_statistics(self, sequence_lengths: List[int], text_lengths: List[int]):
        """
        绘制统计图表
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 序列长度分布
        axes[0, 0].hist(sequence_lengths, bins=10, alpha=0.7, color='blue')
        axes[0, 0].set_title('Sequence Length Distribution')
        axes[0, 0].set_xlabel('Number of Frames')
        axes[0, 0].set_ylabel('Frequency')
        
        # 文本长度分布
        axes[0, 1].hist(text_lengths, bins=10, alpha=0.7, color='green')
        axes[0, 1].set_title('Text Length Distribution')
        axes[0, 1].set_xlabel('Number of Words')
        axes[0, 1].set_ylabel('Frequency')
        
        # 序列长度vs文本长度散点图
        axes[1, 0].scatter(sequence_lengths, text_lengths, alpha=0.6)
        axes[1, 0].set_title('Sequence Length vs Text Length')
        axes[1, 0].set_xlabel('Sequence Length (frames)')
        axes[1, 0].set_ylabel('Text Length (words)')
        
        # 特征示例（显示第一个序列的前几帧）
        if self.pose_sequences:
            sample_seq = self.pose_sequences[0][:min(10, len(self.pose_sequences[0]))]
            im = axes[1, 1].imshow(sample_seq.T, aspect='auto', cmap='viridis')
            axes[1, 1].set_title('Sample Sequence Features')
            axes[1, 1].set_xlabel('Frame')
            axes[1, 1].set_ylabel('Feature Index')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('asl_data_statistics.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_training_example(self):
        """
        创建训练示例
        """
        print("\n" + "="*60)
        print("机器学习训练示例")
        print("="*60)
        
        # 准备数据
        X, y = self.prepare_ml_data(max_sequence_length=50)
        
        print(f"\n数据形状:")
        print(f"  特征矩阵 X: {X.shape}")
        print(f"  标签向量 y: {y.shape}")
        
        # 数据归一化
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y, test_size=0.3, random_state=42
        )
        
        print(f"\n数据分割:")
        print(f"  训练集: {X_train.shape[0]} 样本")
        print(f"  测试集: {X_test.shape[0]} 样本")
        
        print(f"\n建议的模型架构:")
        print(f"  1. LSTM/GRU 模型用于序列建模")
        print(f"     - 输入形状: (batch_size, {X.shape[1]}, {X.shape[2]})")
        print(f"     - LSTM层: 128-256 个隐藏单元")
        print(f"     - 输出层: 回归或分类")
        
        print(f"\n  2. Transformer 模型用于更复杂的序列关系")
        print(f"     - 自注意力机制捕获长距离依赖")
        print(f"     - 位置编码处理时序信息")
        
        print(f"\n  3. CNN+RNN 混合模型")
        print(f"     - CNN提取局部特征")
        print(f"     - RNN建模时序关系")

def main():
    """
    主函数 - 演示完整的数据处理流程
    """
    print("ASL 数据处理和机器学习应用示例")
    print("="*50)
    
    # 初始化数据处理器
    dataset_path = "datasets/signllm_training_data/ASL/dev"
    processor = ASLDataProcessor(dataset_path)
    
    # 加载数据
    sequences, labels = processor.load_all_data()
    
    # 分析数据统计
    processor.analyze_data_statistics()
    
    # 创建训练示例
    processor.create_training_example()
    
    print(f"\n数据使用建议:")
    print(f"1. 数据预处理:")
    print(f"   - 标准化坐标到[-1, 1]范围")
    print(f"   - 处理缺失关键点（插值或掩码）")
    print(f"   - 序列长度归一化")
    
    print(f"\n2. 特征工程:")
    print(f"   - 关节角度计算")
    print(f"   - 运动轨迹特征")
    print(f"   - 手形特征")
    print(f"   - 时频域特征")
    
    print(f"\n3. 模型选择:")
    print(f"   - 序列到序列模型（Seq2Seq）")
    print(f"   - 注意力机制")
    print(f"   - 多模态融合（姿态+文本）")
    
    print(f"\n4. 评估指标:")
    print(f"   - BLEU分数（机器翻译）")
    print(f"   - 准确率（分类任务）")
    print(f"   - 编辑距离（序列相似度）")

if __name__ == "__main__":
    main() 