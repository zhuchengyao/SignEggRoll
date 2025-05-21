import os
import json
import torch
from torch.utils.data import Dataset

class SignPoseDataset(Dataset):
    def __init__(self, json_folder, T_max=100, feature_dim=None):
        """
        json_folder: 处理好的 JSON 样本目录
        T_max:       除去开始/结束帧后的最大帧数
        feature_dim: 单帧特征维度（如果为 None，则第一次 __getitem__ 时自动推断）
        """
        self.files = [os.path.join(json_folder, f) 
                      for f in os.listdir(json_folder) if f.endswith(".json")]
        self.T_max = T_max
        self.feature_dim = feature_dim
        
        # 一个全零向量，用于 padding、start/end
        self._zero_frame = None  

    def __len__(self):
        return len(self.files)

    def _flatten_frame(self, frame: dict):
        # 按 pose/face/hand_left/hand_right 顺序拼接
        return (
            frame["pose_keypoints_2d"]
            + frame["face_keypoints_2d"]
            + frame["hand_left_keypoints_2d"]
            + frame["hand_right_keypoints_2d"]
        )

    def __getitem__(self, idx):
        fn = self.files[idx]
        with open(fn, "r", encoding="utf-8") as f:
            sample = json.load(f)

        # 1) 把每帧 dict -> list
        seq = [self._flatten_frame(frm) for frm in sample["pose"]]
        if self.feature_dim is None:
            self.feature_dim = len(seq[0])
            self._zero_frame = [0.0] * self.feature_dim

        # 2) 截断或 padding 到 T_max 帧
        if len(seq) > self.T_max:
            seq = seq[: self.T_max]
        else:
            pad_n = self.T_max - len(seq)
            seq += [self._zero_frame] * pad_n

        # 3) 增加起始帧和结束帧（这里用 0 向量，也可以用 learnable 参数）
        seq = [self._zero_frame] + seq + [self._zero_frame]

        # 转成 Tensor: [S, D]
        x = torch.tensor(seq, dtype=torch.float32)
        return x  # downstream 可以返回 mask、原始长度等

# 用法示例
# ds = SignPoseDataset("./datasets/processed", T_max=120)
# x = ds[0]     # x.shape == (120+2, feature_dim)
