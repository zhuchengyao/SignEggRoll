import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from typing import List, Tuple, Optional
import sys

# 添加datasets目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'datasets'))
from skeleton_dataloader import load_skels_file

class PoseDataset(Dataset):
    """3D姿态数据集，用于训练diffusion model"""
    
    def __init__(
        self, 
        data_dir: str,
        file_pattern: str = "*.skels",
        num_keypoints: int = 67,
        normalize: bool = True,
        augment: bool = True,
        max_files: Optional[int] = None
    ):
        """
        Args:
            data_dir: 数据目录路径
            file_pattern: 文件匹配模式
            num_keypoints: 关键点数量
            normalize: 是否标准化数据
            augment: 是否进行数据增强
            max_files: 最大文件数量（用于调试）
        """
        self.data_dir = data_dir
        self.num_keypoints = num_keypoints
        self.normalize = normalize
        self.augment = augment
        
        # 查找所有.skels文件
        self.file_paths = glob.glob(os.path.join(data_dir, "**", file_pattern), recursive=True)
        if max_files:
            self.file_paths = self.file_paths[:max_files]
        
        print(f"找到 {len(self.file_paths)} 个数据文件")
        
        # 加载所有数据并展开为单帧
        self.poses = []
        self.load_all_data()
        
        # 计算数据统计信息
        if self.normalize:
            self.compute_stats()
        
    def load_all_data(self):
        """加载所有数据文件"""
        print("正在加载数据...")
        
        for file_path in self.file_paths:
            try:
                # 加载单个文件的数据 (num_frames, 67, 3)
                pose_sequence = load_skels_file(file_path, self.num_keypoints)
                
                # 将每一帧作为独立样本
                for frame_idx in range(pose_sequence.shape[0]):
                    pose = pose_sequence[frame_idx]  # (67, 3)
                    
                    # 检查数据有效性（去除全零帧）
                    if not np.allclose(pose, 0):
                        self.poses.append(pose.astype(np.float32))
                        
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
                continue
        
        print(f"成功加载 {len(self.poses)} 个姿态帧")
        
    def compute_stats(self):
        """计算数据集的均值和标准差用于标准化"""
        all_poses = np.stack(self.poses)  # (N, 67, 3)
        
        # 计算每个维度的统计信息
        self.mean = np.mean(all_poses, axis=0, keepdims=True)  # (1, 67, 3)
        self.std = np.std(all_poses, axis=0, keepdims=True)    # (1, 67, 3)
        
        # 避免除零
        self.std = np.maximum(self.std, 1e-8)
        
        print(f"数据统计 - 均值范围: [{self.mean.min():.4f}, {self.mean.max():.4f}]")
        print(f"数据统计 - 标准差范围: [{self.std.min():.4f}, {self.std.max():.4f}]")
        
    def normalize_pose(self, pose):
        """标准化姿态数据"""
        if self.normalize:
            return (pose - self.mean) / self.std
        return pose
    
    def denormalize_pose(self, pose):
        """反标准化姿态数据"""
        if self.normalize:
            return pose * self.std + self.mean
        return pose
    
    def augment_pose(self, pose):
        """数据增强"""
        if not self.augment:
            return pose
            
        augmented_pose = pose.copy()
        
        # 随机旋转（绕Y轴）
        if np.random.random() < 0.5:
            angle = np.random.uniform(-np.pi/6, np.pi/6)  # ±30度
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            augmented_pose = augmented_pose @ rotation_matrix.T
        
        # 随机缩放
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            augmented_pose *= scale
        
        # 随机平移
        if np.random.random() < 0.3:
            translation = np.random.normal(0, 0.01, (1, 3))
            augmented_pose += translation
        
        # 添加小量噪声
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 0.005, augmented_pose.shape)
            augmented_pose += noise
            
        return augmented_pose
    
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        pose = self.poses[idx].copy()
        
        # 数据增强
        pose = self.augment_pose(pose)
        
        # 标准化
        pose = self.normalize_pose(pose)
        
        return torch.FloatTensor(pose)

def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    normalize: bool = True,
    augment: bool = True,
    max_files: Optional[int] = None
) -> Tuple[DataLoader, PoseDataset]:
    """创建数据加载器"""
    
    dataset = PoseDataset(
        data_dir=data_dir,
        normalize=normalize,
        augment=augment,
        max_files=max_files
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset

# 用于验证数据加载的函数
def visualize_pose_data(dataset: PoseDataset, num_samples: int = 5):
    """可视化姿态数据"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 3))
    
    for i in range(min(num_samples, len(dataset))):
        pose = dataset[i].numpy()  # (67, 3)
        
        ax = fig.add_subplot(1, num_samples, i+1, projection='3d')
        
        # 绘制关键点
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], 
                  c='red', marker='o', s=20)
        
        # 设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(f'样本 {i+1}')
        
        # 设置相同的缩放比例
        max_range = np.array([pose[:,0].max()-pose[:,0].min(),
                             pose[:,1].max()-pose[:,1].min(),
                             pose[:,2].max()-pose[:,2].min()]).max() / 2.0
        mid_x = (pose[:,0].max()+pose[:,0].min()) * 0.5
        mid_y = (pose[:,1].max()+pose[:,1].min()) * 0.5
        mid_z = (pose[:,2].max()+pose[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig('pose_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 测试数据加载
    data_dir = "./datasets/processed"  # 修改为你的数据目录
    
    if os.path.exists(data_dir):
        dataset = PoseDataset(data_dir, max_files=2, normalize=True, augment=False)
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本形状: {sample.shape}")
            print(f"样本范围: [{sample.min():.4f}, {sample.max():.4f}]")
            
            # 可视化几个样本
            visualize_pose_data(dataset, num_samples=3)
    else:
        print(f"数据目录 {data_dir} 不存在") 