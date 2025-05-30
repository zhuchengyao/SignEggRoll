import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import json
import cv2
from typing import List, Tuple, Optional
import random
from PIL import Image
import torchvision.transforms as transforms

class ASLTextPoseDataset(Dataset):
    """ASL文本-姿态配对数据集，用于训练text-to-pose diffusion model"""
    
    def __init__(
        self, 
        data_dir: str,
        num_frames: int = 100,  # ASL序列通常较长
        normalize: bool = True,
        augment: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: ASL数据目录路径，包含dev_*子目录
            num_frames: 每个序列的最大帧数
            normalize: 是否标准化坐标
            augment: 是否进行数据增强
            max_samples: 最大样本数量（用于调试）
        
        预期数据结构:
        data_dir/
        ├── dev_xxx/
        │   ├── pose.json
        │   └── text.txt
        ├── dev_yyy/
        │   ├── pose.json
        │   └── text.txt
        └── ...
        """
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.normalize = normalize
        self.augment = augment
        
        # ASL数据特征维度：身体(18*2) + 左手(21*2) + 右手(21*2) = 120维
        self.feature_dim = 120
        
        # 查找所有ASL数据目录
        self.data_paths = []
        self.captions = []
        
        data_dirs = glob.glob(os.path.join(data_dir, "dev_*"))
        
        for data_path in data_dirs:
            pose_file = os.path.join(data_path, 'pose.json')
            text_file = os.path.join(data_path, 'text.txt')
            
            if os.path.exists(pose_file) and os.path.exists(text_file):
                try:
                    # 读取文本标签
                    with open(text_file, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    
                    if caption:  # 确保文本不为空
                        self.data_paths.append(data_path)
                        self.captions.append(caption)
                except Exception as e:
                    print(f"读取文本文件 {text_file} 失败: {e}")
                    continue
        
        if max_samples:
            self.data_paths = self.data_paths[:max_samples]
            self.captions = self.captions[:max_samples]
        
        print(f"找到 {len(self.data_paths)} 个有效的ASL数据样本")
        
        # 计算数据统计信息用于标准化
        if self.normalize:
            self.compute_normalization_stats()
    
    def compute_normalization_stats(self):
        """计算数据集的标准化统计信息"""
        print("计算数据标准化统计信息...")
        all_poses = []
        
        # 采样部分数据计算统计信息
        sample_indices = random.sample(range(len(self.data_paths)), 
                                     min(50, len(self.data_paths)))
        
        for idx in sample_indices:
            pose_sequence = self.load_pose_sequence(self.data_paths[idx])
            if pose_sequence is not None:
                all_poses.append(pose_sequence)
        
        if all_poses:
            # 合并所有序列
            concatenated = np.concatenate(all_poses, axis=0)
            self.pose_mean = np.mean(concatenated, axis=0)
            self.pose_std = np.std(concatenated, axis=0)
            # 避免除零
            self.pose_std = np.maximum(self.pose_std, 1e-8)
            
            print(f"姿态数据统计 - 均值范围: [{self.pose_mean.min():.3f}, {self.pose_mean.max():.3f}]")
            print(f"姿态数据统计 - 标准差范围: [{self.pose_std.min():.3f}, {self.pose_std.max():.3f}]")
        else:
            # 默认值
            self.pose_mean = np.zeros(self.feature_dim)
            self.pose_std = np.ones(self.feature_dim)
    
    def load_pose_sequence(self, data_path: str) -> np.ndarray:
        """加载姿态序列数据"""
        pose_file = os.path.join(data_path, 'pose.json')
        
        try:
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
            
            poses = pose_data['poses']
            sequence_features = []
            
            for pose in poses:
                frame_features = self.extract_frame_features(pose)
                if frame_features is not None:
                    sequence_features.append(frame_features)
            
            if len(sequence_features) == 0:
                return None
            
            return np.array(sequence_features)  # (frames, feature_dim)
            
        except Exception as e:
            print(f"加载姿态文件 {pose_file} 失败: {e}")
            return None
    
    def extract_frame_features(self, pose: dict) -> np.ndarray:
        """从单帧中提取特征"""
        features = []
        
        # 1. 身体关键点特征 (18个点 * 2坐标 = 36维)
        pose_coords = pose['pose_keypoints_2d']
        pose_points = np.array([[pose_coords[i], pose_coords[i+1], pose_coords[i+2]] 
                               for i in range(0, len(pose_coords), 3)])
        
        # 提取前18个身体关键点的x,y坐标
        for i in range(min(18, len(pose_points))):
            if pose_points[i, 2] > 0.1:  # 置信度阈值
                features.extend([pose_points[i, 0], pose_points[i, 1]])
            else:
                features.extend([0.0, 0.0])
        
        # 补齐到18个点
        while len(features) < 36:
            features.extend([0.0, 0.0])
        
        # 2. 左手关键点特征 (21个点 * 2坐标 = 42维)
        left_hand_coords = pose['hand_left_keypoints_2d']
        left_hand_points = np.array([[left_hand_coords[i], left_hand_coords[i+1], left_hand_coords[i+2]] 
                                    for i in range(0, len(left_hand_coords), 3)])
        
        for i in range(min(21, len(left_hand_points))):
            if left_hand_points[i, 2] > 0.1:
                features.extend([left_hand_points[i, 0], left_hand_points[i, 1]])
            else:
                features.extend([0.0, 0.0])
        
        # 补齐到21个点
        while len(features) < 78:  # 36 + 42
            features.extend([0.0, 0.0])
        
        # 3. 右手关键点特征 (21个点 * 2坐标 = 42维)
        right_hand_coords = pose['hand_right_keypoints_2d']
        right_hand_points = np.array([[right_hand_coords[i], right_hand_coords[i+1], right_hand_coords[i+2]] 
                                     for i in range(0, len(right_hand_coords), 3)])
        
        for i in range(min(21, len(right_hand_points))):
            if right_hand_points[i, 2] > 0.1:
                features.extend([right_hand_points[i, 0], right_hand_points[i, 1]])
            else:
                features.extend([0.0, 0.0])
        
        # 补齐到21个点
        while len(features) < 120:  # 36 + 42 + 42
            features.extend([0.0, 0.0])
        
        # 确保特征维度为120
        features = features[:120]
        assert len(features) == 120, f"Feature dimension mismatch: {len(features)}"
        
        return np.array(features)
    
    def normalize_pose(self, pose_sequence):
        """标准化姿态序列"""
        if self.normalize:
            return (pose_sequence - self.pose_mean) / self.pose_std
        return pose_sequence
    
    def denormalize_pose(self, pose_sequence):
        """反标准化姿态序列"""
        if self.normalize:
            return pose_sequence * self.pose_std + self.pose_mean
        return pose_sequence
    
    def augment_pose_sequence(self, pose_sequence: np.ndarray) -> np.ndarray:
        """姿态序列数据增强"""
        if not self.augment:
            return pose_sequence
        
        augmented_sequence = pose_sequence.copy()
        
        # 随机水平翻转（交换左右手）
        if random.random() < 0.3:
            # 身体关键点翻转
            body_coords = augmented_sequence[:, :36].reshape(-1, 18, 2)
            body_coords[:, :, 0] = -body_coords[:, :, 0]  # x坐标翻转
            
            # 交换左右手数据
            left_hand = augmented_sequence[:, 36:78].copy()
            right_hand = augmented_sequence[:, 78:120].copy()
            
            # 翻转手部x坐标
            left_hand_coords = left_hand.reshape(-1, 21, 2)
            right_hand_coords = right_hand.reshape(-1, 21, 2)
            left_hand_coords[:, :, 0] = -left_hand_coords[:, :, 0]
            right_hand_coords[:, :, 0] = -right_hand_coords[:, :, 0]
            
            # 交换左右手
            augmented_sequence[:, 36:78] = right_hand_coords.reshape(-1, 42)
            augmented_sequence[:, 78:120] = left_hand_coords.reshape(-1, 42)
            augmented_sequence[:, :36] = body_coords.reshape(-1, 36)
        
        # 随机噪声
        if random.random() < 0.2:
            noise = np.random.normal(0, 0.01, augmented_sequence.shape)
            augmented_sequence += noise
        
        # 随机时间扭曲（简单的帧跳跃）
        if random.random() < 0.2 and len(augmented_sequence) > 5:
            # 随机删除一些帧
            keep_indices = sorted(random.sample(range(len(augmented_sequence)), 
                                              int(len(augmented_sequence) * 0.9)))
            augmented_sequence = augmented_sequence[keep_indices]
        
        return augmented_sequence
    
    def augment_text(self, text: str) -> str:
        """文本数据增强"""
        if not self.augment:
            return text
        
        # ASL特定的文本增强
        replacements = {
            "hello": ["hi", "greetings", "good morning"],
            "thank you": ["thanks", "appreciate", "grateful"],
            "please": ["kindly", "would you"],
            "good": ["nice", "great", "wonderful"],
            "bad": ["poor", "terrible", "awful"]
        }
        
        augmented_text = text.lower()
        for word, synonyms in replacements.items():
            if word in augmented_text and random.random() < 0.3:
                synonym = random.choice(synonyms)
                augmented_text = augmented_text.replace(word, synonym, 1)
        
        return augmented_text
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        caption = self.captions[idx]
        
        try:
            # 加载姿态序列
            pose_sequence = self.load_pose_sequence(data_path)
            
            if pose_sequence is None or len(pose_sequence) == 0:
                # 返回空序列作为fallback
                pose_sequence = np.zeros((self.num_frames, self.feature_dim))
                caption = "empty sequence"
            else:
                # 数据增强
                pose_sequence = self.augment_pose_sequence(pose_sequence)
                caption = self.augment_text(caption)
                
                # 序列长度处理
                if len(pose_sequence) > self.num_frames:
                    # 截断
                    pose_sequence = pose_sequence[:self.num_frames]
                elif len(pose_sequence) < self.num_frames:
                    # 填充
                    padding = np.zeros((self.num_frames - len(pose_sequence), self.feature_dim))
                    pose_sequence = np.vstack([pose_sequence, padding])
            
            # 标准化
            pose_sequence = self.normalize_pose(pose_sequence)
            
            return {
                'pose_sequence': torch.FloatTensor(pose_sequence),  # (T, 120)
                'caption': caption,
                'data_path': data_path,
                'sequence_length': min(len(pose_sequence), self.num_frames)
            }
            
        except Exception as e:
            print(f"加载ASL数据 {data_path} 失败: {e}")
            # 返回默认数据作为fallback
            pose_sequence = np.zeros((self.num_frames, self.feature_dim))
            if self.normalize:
                pose_sequence = self.normalize_pose(pose_sequence)
            
            return {
                'pose_sequence': torch.FloatTensor(pose_sequence),
                'caption': "error loading data",
                'data_path': data_path,
                'sequence_length': 0
            }

# 保持原有的视频数据集类以便向后兼容
class TextVideoDataset(Dataset):
    """原有的文本-视频配对数据集"""
    
    def __init__(
        self, 
        data_dir: str,
        num_frames: int = 16,
        frame_size: int = 64,
        normalize: bool = True,
        augment: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: 数据目录路径，包含videos/和captions/子目录
            num_frames: 每个视频片段的帧数
            frame_size: 帧的尺寸 (正方形)
            normalize: 是否标准化像素值到[-1, 1]
            augment: 是否进行数据增强
            max_samples: 最大样本数量（用于调试）
        
        预期数据结构:
        data_dir/
        ├── videos/
        │   ├── video_001.mp4
        │   ├── video_002.mp4
        │   └── ...
        └── captions/
            ├── video_001.txt
            ├── video_002.txt
            └── ...
        """
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.normalize = normalize
        self.augment = augment
        
        # 查找所有视频文件
        video_dir = os.path.join(data_dir, "videos")
        caption_dir = os.path.join(data_dir, "captions")
        
        if not os.path.exists(video_dir) or not os.path.exists(caption_dir):
            raise ValueError(f"数据目录结构不正确，需要包含videos/和captions/子目录")
        
        # 找到所有有对应文本描述的视频文件
        self.video_paths = []
        self.captions = []
        
        video_files = glob.glob(os.path.join(video_dir, "*.mp4")) + \
                     glob.glob(os.path.join(video_dir, "*.avi")) + \
                     glob.glob(os.path.join(video_dir, "*.mov"))
        
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            caption_path = os.path.join(caption_dir, f"{video_name}.txt")
            
            if os.path.exists(caption_path):
                try:
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    
                    if caption:  # 确保文本不为空
                        self.video_paths.append(video_path)
                        self.captions.append(caption)
                except Exception as e:
                    print(f"读取文本文件 {caption_path} 失败: {e}")
                    continue
        
        if max_samples:
            self.video_paths = self.video_paths[:max_samples]
            self.captions = self.captions[:max_samples]
        
        print(f"找到 {len(self.video_paths)} 个有效的视频-文本配对")
        
        # 图像变换
        if self.normalize:
            self.transform = transforms.Compose([
                transforms.Resize((frame_size, frame_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化到[-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((frame_size, frame_size)),
                transforms.ToTensor()
            ])
    
    def load_video_frames(self, video_path: str) -> np.ndarray:
        """加载视频帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < self.num_frames:
                # 如果视频帧数不足，重复播放
                frame_indices = list(range(total_frames)) * (self.num_frames // total_frames + 1)
                frame_indices = frame_indices[:self.num_frames]
            else:
                # 均匀采样帧
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # 转换BGR到RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    # 如果读取失败，复制最后一帧
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        # 创建黑色帧
                        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        finally:
            cap.release()
        
        return np.array(frames)  # (num_frames, H, W, 3)
    
    def augment_video(self, frames: np.ndarray) -> np.ndarray:
        """视频数据增强"""
        if not self.augment:
            return frames
        
        augmented_frames = frames.copy()
        
        # 随机水平翻转
        if random.random() < 0.5:
            augmented_frames = np.flip(augmented_frames, axis=2)
        
        # 随机亮度调整
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.8, 1.2)
            augmented_frames = np.clip(augmented_frames * brightness_factor, 0, 255)
        
        # 随机对比度调整
        if random.random() < 0.3:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = np.mean(augmented_frames)
            augmented_frames = np.clip((augmented_frames - mean) * contrast_factor + mean, 0, 255)
        
        return augmented_frames.astype(np.uint8)
    
    def augment_text(self, text: str) -> str:
        """文本数据增强（同义词替换等）"""
        if not self.augment:
            return text
        
        # 简单的文本增强：随机替换一些常见词汇
        # 这里只是示例，实际项目中可以使用更复杂的NLP技术
        replacements = {
            "person": ["man", "woman", "individual", "human"],
            "walking": ["strolling", "moving", "stepping"],
            "running": ["jogging", "sprinting", "dashing"],
            "beautiful": ["stunning", "gorgeous", "lovely"],
            "quickly": ["rapidly", "swiftly", "fast"]
        }
        
        augmented_text = text
        for word, synonyms in replacements.items():
            if word in augmented_text and random.random() < 0.2:
                synonym = random.choice(synonyms)
                augmented_text = augmented_text.replace(word, synonym, 1)
        
        return augmented_text
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        caption = self.captions[idx]
        
        try:
            # 加载视频帧
            frames = self.load_video_frames(video_path)  # (T, H, W, 3)
            
            # 数据增强
            frames = self.augment_video(frames)
            caption = self.augment_text(caption)
            
            # 转换帧格式
            video_tensor = torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            
            for i, frame in enumerate(frames):
                frame_pil = Image.fromarray(frame)
                frame_tensor = self.transform(frame_pil)  # (3, H, W)
                video_tensor[:, i, :, :] = frame_tensor
            
            return {
                'video': video_tensor,  # (3, T, H, W)
                'caption': caption,
                'video_path': video_path
            }
            
        except Exception as e:
            print(f"加载视频 {video_path} 失败: {e}")
            # 返回黑色视频和空文本作为fallback
            video_tensor = torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            if self.normalize:
                video_tensor = video_tensor * 2 - 1  # 标准化到[-1, 1]
            
            return {
                'video': video_tensor,
                'caption': "empty video",
                'video_path': video_path
            }

def asl_collate_fn(batch):
    """ASL数据集的批次整理函数"""
    pose_sequences = []
    captions = []
    
    for item in batch:
        pose_sequences.append(item['pose_sequence'])
        captions.append(item['caption'])
    
    # 堆叠张量
    pose_sequences = torch.stack(pose_sequences, dim=0)
    
    return {
        'pose_sequences': pose_sequences,
        'captions': captions
    }

def create_asl_dataloader(
    data_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,  # Windows默认为0
    num_frames: int = 100,
    max_samples: Optional[int] = None,
    normalize: bool = True,
    augment: bool = False
) -> Tuple[DataLoader, ASLTextPoseDataset]:
    """创建ASL数据加载器"""
    
    dataset = ASLTextPoseDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        normalize=normalize,
        augment=augment,
        max_samples=max_samples
    )
    
    # Windows系统优化
    import platform
    if platform.system() == 'Windows':
        num_workers = 0  # Windows上避免multiprocessing问题
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=asl_collate_fn,  # 使用模块级别的函数
        pin_memory=torch.cuda.is_available(),
        drop_last=True if len(dataset) > batch_size else False
    )
    
    return dataloader, dataset

# 用于验证数据加载的函数
def visualize_video_data(dataset: TextVideoDataset, num_samples: int = 2):
    """可视化视频数据"""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        video = sample['video']  # (3, T, H, W)
        caption = sample['caption']
        
        # 转换为显示格式
        if dataset.normalize:
            video = (video + 1) / 2  # 从[-1, 1]转换到[0, 1]
        
        video = video.permute(1, 2, 3, 0)  # (T, H, W, 3)
        video = video.clamp(0, 1)
        
        # 创建动画
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"样本 {i+1}: {caption[:50]}...")
        ax.axis('off')
        
        def animate(frame_idx):
            ax.clear()
            ax.imshow(video[frame_idx])
            ax.set_title(f"样本 {i+1}: {caption[:50]}... (帧 {frame_idx+1}/{len(video)})")
            ax.axis('off')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(video), interval=200, repeat=True)
        
        # 保存为GIF
        anim.save(f'video_sample_{i+1}.gif', writer='pillow', fps=5)
        plt.close()
        
        print(f"保存视频样本 {i+1}: {caption}")

def create_demo_dataset(output_dir: str, num_videos: int = 10):
    """创建演示数据集（用于测试）"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "captions"), exist_ok=True)
    
    # 简单的合成视频生成
    for i in range(num_videos):
        # 创建简单的移动方块视频
        frames = []
        color = np.random.rand(3)
        
        for frame_idx in range(16):
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_aspect('equal')
            
            # 移动的方块
            x = frame_idx * 0.5
            y = 5 + 2 * np.sin(frame_idx * 0.3)
            
            rect = patches.Rectangle((x, y), 1, 1, linewidth=1, 
                                   edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            
            ax.set_title(f"Frame {frame_idx}")
            
            # 保存帧
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            
            plt.close(fig)
        
        # 保存为视频
        video_path = os.path.join(output_dir, "videos", f"video_{i:03d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 8.0, (frames[0].shape[1], frames[0].shape[0]))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        # 创建对应的文本描述
        actions = ["moving", "sliding", "bouncing", "floating"]
        colors = ["red", "blue", "green", "yellow", "purple"]
        directions = ["left to right", "right to left", "up and down"]
        
        caption = f"A {random.choice(colors)} square {random.choice(actions)} {random.choice(directions)}"
        
        caption_path = os.path.join(output_dir, "captions", f"video_{i:03d}.txt")
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption)
    
    print(f"创建了 {num_videos} 个演示视频在 {output_dir}")

if __name__ == "__main__":
    # 创建演示数据集
    demo_dir = "./demo_video_data"
    create_demo_dataset(demo_dir, num_videos=5)
    
    # 测试数据加载
    if os.path.exists(demo_dir):
        dataset = TextVideoDataset(demo_dir, num_frames=8, frame_size=32, max_samples=3)
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"视频形状: {sample['video'].shape}")
            print(f"文本描述: {sample['caption']}")
            
            # 可视化样本
            visualize_video_data(dataset, num_samples=2)
    else:
        print(f"演示数据目录 {demo_dir} 不存在") 