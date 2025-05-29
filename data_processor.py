"""
数据处理模块 - 处理Prompt2Sign数据集和多语言手语数据
支持OpenPose和DWPose格式的姿态数据
"""

import os
import json
import h5py
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from pathlib import Path
import pickle
import mediapipe as mp
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseExtractor:
    """姿态提取器 - 支持OpenPose和MediaPipe"""
    
    def __init__(self, method: str = "mediapipe"):
        self.method = method
        if method == "mediapipe":
            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands
            self.mp_face = mp.solutions.face_mesh
            
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5
            )
            self.face = self.mp_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
    
    def extract_from_video(self, video_path: str) -> List[Dict]:
        """从视频中提取姿态关键点"""
        cap = cv2.VideoCapture(video_path)
        poses = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_data = self._extract_frame_pose(frame_rgb)
            poses.append(pose_data)
        
        cap.release()
        return poses
    
    def _extract_frame_pose(self, frame: np.ndarray) -> Dict:
        """从单帧中提取姿态"""
        if self.method == "mediapipe":
            return self._extract_mediapipe_pose(frame)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
    
    def _extract_mediapipe_pose(self, frame: np.ndarray) -> Dict:
        """使用MediaPipe提取姿态"""
        h, w = frame.shape[:2]
        
        # 身体姿态
        pose_results = self.pose.process(frame)
        pose_keypoints = []
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                pose_keypoints.extend([landmark.x * w, landmark.y * h, landmark.visibility])
        else:
            pose_keypoints = [0.0] * (33 * 3)  # MediaPipe pose有33个关键点
        
        # 手部姿态
        hand_results = self.hands.process(frame)
        left_hand_keypoints = [0.0] * (21 * 3)
        right_hand_keypoints = [0.0] * (21 * 3)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_keypoints = []
                for landmark in hand_landmarks.landmark:
                    hand_keypoints.extend([landmark.x * w, landmark.y * h, landmark.z])
                
                if handedness.classification[0].label == "Left":
                    left_hand_keypoints = hand_keypoints
                else:
                    right_hand_keypoints = hand_keypoints
        
        # 面部关键点（简化版，只取部分关键点）
        face_results = self.face.process(frame)
        face_keypoints = []
        if face_results.multi_face_landmarks:
            # 只取前70个面部关键点
            for i, landmark in enumerate(face_results.multi_face_landmarks[0].landmark[:70]):
                face_keypoints.extend([landmark.x * w, landmark.y * h, landmark.z])
        else:
            face_keypoints = [0.0] * (70 * 3)
        
        return {
            "pose_keypoints_2d": pose_keypoints,
            "hand_left_keypoints_2d": left_hand_keypoints,
            "hand_right_keypoints_2d": right_hand_keypoints,
            "face_keypoints_2d": face_keypoints
        }


class PoseNormalizer:
    """姿态数据标准化器"""
    
    @staticmethod
    def normalize_pose_sequence(poses: List[Dict]) -> List[Dict]:
        """标准化姿态序列"""
        if not poses:
            return poses
        
        normalized_poses = []
        for pose in poses:
            normalized_pose = PoseNormalizer._normalize_single_pose(pose)
            normalized_poses.append(normalized_pose)
        
        return normalized_poses
    
    @staticmethod
    def _normalize_single_pose(pose: Dict) -> Dict:
        """标准化单个姿态"""
        normalized = {}
        
        for key, keypoints in pose.items():
            if not keypoints or len(keypoints) == 0:
                normalized[key] = keypoints
                continue
            
            # 转换为numpy数组
            kpts = np.array(keypoints).reshape(-1, 3)
            
            # 过滤无效点
            valid_mask = (kpts[:, 0] > 0) & (kpts[:, 1] > 0)
            if not np.any(valid_mask):
                normalized[key] = keypoints
                continue
            
            # 计算边界框
            valid_points = kpts[valid_mask]
            min_x, min_y = valid_points[:, :2].min(axis=0)
            max_x, max_y = valid_points[:, :2].max(axis=0)
            
            # 标准化到[0, 1]范围
            width = max_x - min_x
            height = max_y - min_y
            
            if width > 0 and height > 0:
                kpts[valid_mask, 0] = (kpts[valid_mask, 0] - min_x) / width
                kpts[valid_mask, 1] = (kpts[valid_mask, 1] - min_y) / height
            
            normalized[key] = kpts.flatten().tolist()
        
        return normalized


class Prompt2SignDataset(Dataset):
    """Prompt2Sign数据集"""
    
    def __init__(self, 
                 data_dir: str,
                 language: str = "ASL",
                 split: str = "train",
                 max_sequence_length: int = 256,
                 pose_dim: int = 150,
                 transform=None):
        self.data_dir = Path(data_dir)
        self.language = language
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.pose_dim = pose_dim
        self.transform = transform
        
        # 加载数据索引
        self.data_index = self._load_data_index()
        
        # 语言映射
        self.language_to_id = {
            "ASL": 0, "DGS": 1, "KSL": 2, "DSGS": 3,
            "LSF-CH": 4, "LIS-CH": 5, "LSA": 6, "TSL": 7
        }
        
        logger.info(f"Loaded {len(self.data_index)} samples for {language} {split}")
    
    def _load_data_index(self) -> List[Dict]:
        """加载数据索引"""
        index_file = self.data_dir / f"{self.language}_{self.split}_index.json"
        
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 如果索引文件不存在，扫描目录创建索引
            return self._create_data_index()
    
    def _create_data_index(self) -> List[Dict]:
        """创建数据索引"""
        data_index = []
        
        # 扫描数据目录
        language_dir = self.data_dir / self.language / self.split
        if not language_dir.exists():
            logger.warning(f"Directory {language_dir} does not exist")
            return []
        
        for sample_dir in language_dir.iterdir():
            if sample_dir.is_dir():
                # 检查必要文件
                text_file = sample_dir / "text.txt"
                pose_file = sample_dir / "pose.json"
                
                if text_file.exists() and pose_file.exists():
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    data_index.append({
                        "id": sample_dir.name,
                        "text": text,
                        "pose_file": str(pose_file),
                        "language": self.language
                    })
        
        # 保存索引
        index_file = self.data_dir / f"{self.language}_{self.split}_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(data_index, f, ensure_ascii=False, indent=2)
        
        return data_index
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.data_index[idx]
        
        # 加载文本
        text = sample["text"]
        
        # 加载姿态数据 - 修复路径问题
        pose_file_path = sample["pose_file"]
        if not Path(pose_file_path).is_absolute():
            # 如果是相对路径，则相对于数据目录
            pose_file_path = self.data_dir / pose_file_path
        
        with open(pose_file_path, 'r') as f:
            pose_data = json.load(f)
        
        # 处理姿态序列
        pose_sequence = self._process_pose_sequence(pose_data["poses"])
        
        # 获取语言ID
        language_id = self.language_to_id[sample["language"]]
        
        return {
            "text": text,
            "pose_sequence": pose_sequence,
            "language": sample["language"],
            "language_id": language_id,
            "length": len(pose_sequence),
            "sample_id": sample["id"]
        }
    
    def _process_pose_sequence(self, poses: List[Dict]) -> torch.Tensor:
        """处理姿态序列"""
        # 标准化姿态
        normalized_poses = PoseNormalizer.normalize_pose_sequence(poses)
        
        # 转换为特征向量
        feature_vectors = []
        for pose in normalized_poses:
            feature_vector = self._pose_to_feature_vector(pose)
            feature_vectors.append(feature_vector)
        
        # 截断或填充到固定长度
        if len(feature_vectors) > self.max_sequence_length:
            feature_vectors = feature_vectors[:self.max_sequence_length]
        else:
            # 填充零向量
            while len(feature_vectors) < self.max_sequence_length:
                feature_vectors.append(np.zeros(self.pose_dim))
        
        return torch.tensor(feature_vectors, dtype=torch.float32)
    
    def _pose_to_feature_vector(self, pose: Dict) -> np.ndarray:
        """将姿态转换为特征向量"""
        # 提取关键部位的关键点
        pose_kpts = np.array(pose.get("pose_keypoints_2d", []))
        left_hand_kpts = np.array(pose.get("hand_left_keypoints_2d", []))
        right_hand_kpts = np.array(pose.get("hand_right_keypoints_2d", []))
        
        # 选择重要的关键点（上身8个点 + 双手42个点）
        if len(pose_kpts) >= 24:  # 至少8个点的x,y,confidence
            # 选择上身关键点（肩膀、肘部、手腕、颈部）
            upper_body_indices = [0, 2, 5, 7, 9, 10, 11, 12]  # 对应重要的上身关键点
            selected_pose = []
            for i in upper_body_indices:
                if i * 3 + 2 < len(pose_kpts):
                    selected_pose.extend(pose_kpts[i*3:i*3+3])
                else:
                    selected_pose.extend([0.0, 0.0, 0.0])
        else:
            selected_pose = [0.0] * 24  # 8个点 * 3
        
        # 手部关键点（每只手21个点）
        if len(left_hand_kpts) >= 63:
            left_hand = left_hand_kpts[:63].tolist()
        else:
            left_hand = [0.0] * 63
        
        if len(right_hand_kpts) >= 63:
            right_hand = right_hand_kpts[:63].tolist()
        else:
            right_hand = [0.0] * 63
        
        # 组合特征向量 (24 + 63 + 63 = 150)
        feature_vector = np.array(selected_pose + left_hand + right_hand)
        
        # 确保维度正确
        if len(feature_vector) > self.pose_dim:
            feature_vector = feature_vector[:self.pose_dim]
        elif len(feature_vector) < self.pose_dim:
            padding = np.zeros(self.pose_dim - len(feature_vector))
            feature_vector = np.concatenate([feature_vector, padding])
        
        return feature_vector


class MultilingualSignDataset(Dataset):
    """多语言手语数据集"""
    
    def __init__(self, 
                 data_dirs: Dict[str, str],
                 languages: List[str] = None,
                 split: str = "train",
                 max_sequence_length: int = 256,
                 pose_dim: int = 150):
        self.data_dirs = data_dirs
        self.languages = languages or list(data_dirs.keys())
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.pose_dim = pose_dim
        
        # 加载所有语言的数据
        self.all_samples = []
        for lang in self.languages:
            if lang in data_dirs:
                dataset = Prompt2SignDataset(
                    data_dirs[lang], 
                    language=lang, 
                    split=split,
                    max_sequence_length=max_sequence_length,
                    pose_dim=pose_dim
                )
                self.all_samples.extend([(i, lang) for i in range(len(dataset))])
                setattr(self, f"{lang}_dataset", dataset)
        
        logger.info(f"Loaded {len(self.all_samples)} total samples across {len(self.languages)} languages")
    
    def __len__(self) -> int:
        return len(self.all_samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_idx, language = self.all_samples[idx]
        dataset = getattr(self, f"{language}_dataset")
        return dataset[sample_idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """数据批处理函数"""
    texts = [item["text"] for item in batch]
    pose_sequences = torch.stack([item["pose_sequence"] for item in batch])
    languages = [item["language"] for item in batch]
    language_ids = torch.tensor([item["language_id"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]
    
    return {
        "texts": texts,
        "pose_sequences": pose_sequences,
        "languages": languages,
        "language_ids": language_ids,
        "lengths": lengths,
        "sample_ids": sample_ids
    }


class DataProcessor:
    """数据处理器 - 处理原始视频数据"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pose_extractor = PoseExtractor()
    
    def process_video_dataset(self, 
                            video_dir: str, 
                            annotation_file: str, 
                            language: str,
                            split: str = "train"):
        """处理视频数据集"""
        video_dir = Path(video_dir)
        
        # 加载标注文件
        if annotation_file.endswith('.json'):
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        elif annotation_file.endswith('.csv'):
            annotations = pd.read_csv(annotation_file).to_dict('records')
        else:
            raise ValueError("Unsupported annotation file format")
        
        # 创建输出目录
        output_lang_dir = self.output_dir / language / split
        output_lang_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理每个视频
        for i, annotation in enumerate(tqdm(annotations, desc=f"Processing {language} {split}")):
            try:
                self._process_single_video(annotation, video_dir, output_lang_dir, i)
            except Exception as e:
                logger.error(f"Error processing video {i}: {e}")
                continue
    
    def _process_single_video(self, annotation: Dict, video_dir: Path, output_dir: Path, idx: int):
        """处理单个视频"""
        # 获取视频路径和文本
        video_file = annotation.get("video_file") or annotation.get("file_name")
        text = annotation.get("text") or annotation.get("sentence")
        
        if not video_file or not text:
            logger.warning(f"Missing video_file or text in annotation {idx}")
            return
        
        video_path = video_dir / video_file
        if not video_path.exists():
            logger.warning(f"Video file not found: {video_path}")
            return
        
        # 提取姿态
        poses = self.pose_extractor.extract_from_video(str(video_path))
        
        if not poses:
            logger.warning(f"No poses extracted from {video_path}")
            return
        
        # 创建样本目录
        sample_dir = output_dir / f"sample_{idx:06d}"
        sample_dir.mkdir(exist_ok=True)
        
        # 保存文本
        with open(sample_dir / "text.txt", 'w', encoding='utf-8') as f:
            f.write(text)
        
        # 保存姿态数据
        pose_data = {
            "poses": poses,
            "video_file": video_file,
            "num_frames": len(poses)
        }
        
        with open(sample_dir / "pose.json", 'w') as f:
            json.dump(pose_data, f, indent=2)


def create_dataloaders(data_dirs: Dict[str, str],
                      languages: List[str],
                      batch_size: int = 8,
                      num_workers: int = 4,
                      max_sequence_length: int = 256) -> Dict[str, DataLoader]:
    """创建数据加载器"""
    dataloaders = {}
    
    for split in ["train", "val", "test"]:
        dataset = MultilingualSignDataset(
            data_dirs=data_dirs,
            languages=languages,
            split=split,
            max_sequence_length=max_sequence_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        dataloaders[split] = dataloader
    
    return dataloaders


if __name__ == "__main__":
    # 测试数据处理
    processor = DataProcessor("./processed_data")
    
    # 示例：处理ASL数据
    # processor.process_video_dataset(
    #     video_dir="./raw_data/ASL/videos",
    #     annotation_file="./raw_data/ASL/annotations.json",
    #     language="ASL",
    #     split="train"
    # )
    
    # 测试数据集加载
    data_dirs = {
        "ASL": "./processed_data",
        "DGS": "./processed_data"
    }
    
    dataloaders = create_dataloaders(
        data_dirs=data_dirs,
        languages=["ASL", "DGS"],
        batch_size=4
    )
    
    # 测试数据加载
    for split, dataloader in dataloaders.items():
        print(f"\n{split} split:")
        for batch in dataloader:
            print(f"  Batch size: {len(batch['texts'])}")
            print(f"  Pose sequences shape: {batch['pose_sequences'].shape}")
            print(f"  Languages: {batch['languages']}")
            break 