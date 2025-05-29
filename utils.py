"""
工具函数模块
包含训练、评估、可视化等辅助功能
"""

import os
import json
import random
import math
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from datetime import datetime
import cv2
import logging

__all__ = [
    "set_seed",
    "online_mean_std",
    "compute_loss",
    "save_ckpt",
    "load_ckpt",
    "render_pose_sequence",
    "render_animation_from_json",
    "AverageMeter",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
    "get_model_size",
    "create_output_dir",
    "save_config",
    "load_config",
    "PoseVisualizer",
    "plot_training_curves",
    "plot_evaluation_metrics",
    "create_multilingual_comparison",
    "ConfigManager",
    "format_time",
    "get_device_info",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# dataset statistics (Welford online)
# ---------------------------------------------------------------------------

def online_mean_std(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    count = 0
    mean = None
    M2 = None  # sum of squares for variance
    for batch in loader:
        poses = batch["pose"]                # [B, T, D]
        B, T, D = poses.shape
        poses = poses.view(-1, D)             # [B*T, D]
        n = poses.shape[0]
        batch_mean = poses.mean(dim=0)
        batch_var = poses.var(dim=0, unbiased=False)
        if mean is None:
            mean = batch_mean
            M2 = batch_var * n
        else:
            delta = batch_mean - mean
            total = count + n
            mean = mean + delta * n / total
            M2 = M2 + batch_var * n + delta.pow(2) * count * n / total
        count += n
    std = torch.sqrt(M2 / count)
    return mean, std

# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    mse = F.mse_loss(pred, target)
    motion_pred = pred[:, 1:] - pred[:, :-1]
    motion_tgt = target[:, 1:] - target[:, :-1]
    motion = F.mse_loss(motion_pred, motion_tgt)
    vel = (motion_pred[:, 1:] - motion_pred[:, :-1]).pow(2).mean()
    total = mse + 0.1 * motion + 0.1 * vel
    return total, {"mse": mse.item(), "motion": motion.item(), "vel": vel.item()}

# ---------------------------------------------------------------------------
# checkpoint helpers
# ---------------------------------------------------------------------------

def _sd(x):
    return x.state_dict() if x is not None else {}


def save_ckpt(model, optim, sched, scaler, step: int, work: str, final=False):
    tag = "final" if final else step
    p = Path(work) / f"checkpoint_{tag}.pth"
    torch.save({
        "model": model.state_dict(),
        "optim": _sd(optim),
        "sched": _sd(sched),
        "scaler": _sd(scaler),
        "step": step,
    }, p)
    print(f"💾 Saved checkpoint to {p}")


def load_ckpt(model, optim, sched, scaler, work: str) -> int:
    ckpts = sorted(Path(work).glob("checkpoint_*.pth"))
    if not ckpts:
        return 0
    last = ckpts[-1]
    ck = torch.load(last, map_location=model.device)
    model.load_state_dict(ck["model"])
    if ck["optim"] and optim is not None:
        optim.load_state_dict(ck["optim"])
    if ck["sched"] and sched is not None:
        sched.load_state_dict(ck["sched"])
    if ck.get("scaler") and scaler is not None:
        scaler.load_state_dict(ck["scaler"])
    print(f"🔄 Resumed from {last}")
    return ck.get("step", 0)

# ---------------------------------------------------------------------------
# animation rendering
# ---------------------------------------------------------------------------

# OpenPose BODY_25 core pairs
BODY_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8),
    (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17),
    (0, 16), (16, 18),
]

# OpenPose hand pairs
HAND_PAIRS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

# OpenPose face pairs (68+2 points)
FACE_PAIRS = [
    *( (i, i+1) for i in range(0, 16) ),
    *( (i, i+1) for i in range(17, 21) ),
    *( (i, i+1) for i in range(22, 26) ),
    *( (i, i+1) for i in range(36, 41) ), (41, 36),
    *( (i, i+1) for i in range(42, 47) ), (47, 42),
    *( (i, i+1) for i in range(27, 30) ),
    *( (i, i+1) for i in range(30, 35) ), (35, 30),
    *( (i, i+1) for i in range(48, 59) ), (59, 48),
    *( (i, i+1) for i in range(60, 67) ), (67, 60),
]


def render_pose_sequence(pose_sequence: List[Dict[str, List[float]]], output_path: str, fps: int = 15) -> None:
    """
    Render animation from a sequence of frame dicts containing keypoint lists.
    Each frame dict should have keys: 'pose_keypoints_2d', 'face_keypoints_2d',
    'hand_left_keypoints_2d', 'hand_right_keypoints_2d'.
    """
    # 1) 计算整体坐标范围
    all_x, all_y = [], []
    for frame in pose_sequence:
        for key in ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            pts = np.array(frame.get(key, [])).reshape(-1, 3)
            if pts.size:
                all_x.extend(pts[:, 0].tolist())
                all_y.extend(pts[:, 1].tolist())
    if not all_x or not all_y:
        raise ValueError("No valid keypoints found in sequence.")
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    mx = (max_x - min_x) * 0.1
    my = (max_y - min_y) * 0.1
    x0, x1 = min_x - mx, max_x + mx
    y0, y1 = min_y - my, max_y + my

    # 2) 动画绘制
    T = len(pose_sequence)
    fig, ax = plt.subplots(figsize=(8, 8))

    def init():
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.invert_yaxis()
        ax.set_aspect('equal', 'box')
        ax.axis('off')

    def update(i):
        ax.clear()
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.invert_yaxis()
        ax.set_aspect('equal', 'box')
        ax.axis('off')

        frame = pose_sequence[i]
        # draw body
        body = np.array(frame['pose_keypoints_2d']).reshape(-1,3)
        for a,b in BODY_PAIRS:
            if body[a,2]>0.05 and body[b,2]>0.05:
                ax.plot([body[a,0],body[b,0]], [body[a,1],body[b,1]], 'b-', lw=2)
        ax.scatter(body[:,0], body[:,1], c='b', s=10)
        # draw face
        face = np.array(frame.get('face_keypoints_2d',[])).reshape(-1,3)
        if face.size:
            for a,b in FACE_PAIRS:
                if face[a,2]>0.01 and face[b,2]>0.01:
                    ax.plot([face[a,0],face[b,0]], [face[a,1],face[b,1]], 'm-', lw=1)
            ax.scatter(face[:,0], face[:,1], c='m', s=5)
        # draw left hand
        lh = np.array(frame['hand_left_keypoints_2d']).reshape(-1,3)
        for a,b in HAND_PAIRS:
            if lh[a,2]>0.05 and lh[b,2]>0.05:
                ax.plot([lh[a,0],lh[b,0]], [lh[a,1],lh[b,1]], 'r-', lw=2)
        ax.scatter(lh[:,0], lh[:,1], c='r', s=8)
        # draw right hand
        rh = np.array(frame['hand_right_keypoints_2d']).reshape(-1,3)
        for a,b in HAND_PAIRS:
            if rh[a,2]>0.05 and rh[b,2]>0.05:
                ax.plot([rh[a,0],rh[b,0]], [rh[a,1],rh[b,1]], 'g-', lw=2)
        ax.scatter(rh[:,0], rh[:,1], c='g', s=8)

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=1000/fps, blit=False
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='ffmpeg', fps=fps)
    plt.close()
    print(f"✅ Animation saved to: {output_path}")


def render_animation_from_json(json_path: str, output_path: str, fps: int = 15) -> None:
    """
    Load a prediction JSON and render an animation.
    Supports two formats:
      1) Raw frame dicts (with 'pose_keypoints_2d', etc.)
      2) Flat pose vectors from inference (list of floats length 411).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    seq = data.get('pose', [])
    if not seq:
        raise ValueError(f"No 'pose' key in {json_path}")

    first = seq[0]
    if isinstance(first, (list, tuple)):
        splits = [75, 210, 63, 63]
        if sum(splits) != len(first):
            raise ValueError(f"Unexpected pose_dim {len(first)}; expected {sum(splits)}")
        frames = []
        for vec in seq:
            frames.append({
                'pose_keypoints_2d':       vec[0              : splits[0]],
                'face_keypoints_2d':       vec[splits[0]      : splits[0]+splits[1]],
                'hand_left_keypoints_2d':  vec[splits[0]+splits[1]             : splits[0]+splits[1]+splits[2]],
                'hand_right_keypoints_2d': vec[splits[0]+splits[1]+splits[2]    : sum(splits)],
            })
    elif isinstance(first, dict):
        frames = seq
    else:
        raise TypeError(f"Unsupported frame type: {type(first)}")

    render_pose_sequence(frames, output_path, fps)


class AverageMeter:
    """平均值计算器"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state: Dict, filename: str, is_best: bool = False):
    """保存检查点"""
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace('.pt', '_best.pt')
        torch.save(state, best_filename)


def load_checkpoint(filename: str, model: nn.Module, optimizer=None, scheduler=None) -> Dict:
    """加载检查点"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")
    
    checkpoint = torch.load(filename, map_location='cpu')
    
    # 加载模型状态
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 加载优化器状态
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> float:
    """获取模型大小（MB）"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def create_output_dir(base_dir: str, experiment_name: str = None) -> Path:
    """创建输出目录"""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"signllm_{timestamp}"
    
    output_dir = Path(base_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "configs").mkdir(exist_ok=True)
    
    return output_dir


def save_config(config: Dict, output_dir: Path):
    """保存配置文件"""
    config_file = output_dir / "configs" / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(config_file: str) -> Dict:
    """加载配置文件"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


class PoseVisualizer:
    """姿态可视化器"""
    
    def __init__(self, pose_dim: int = 150):
        self.pose_dim = pose_dim
        # 定义关键点连接关系（用于绘制骨架）
        self.connections = self._get_pose_connections()
    
    def _get_pose_connections(self) -> List[Tuple[int, int]]:
        """获取姿态关键点连接关系"""
        # 上身连接关系（简化版）
        body_connections = [
            (0, 1), (1, 2), (2, 3),  # 右臂
            (0, 4), (4, 5), (5, 6),  # 左臂
            (0, 7),  # 颈部到头部
        ]
        
        # 手部连接关系（每只手21个点）
        hand_connections = []
        for hand_offset in [8, 29]:  # 左手和右手的起始索引
            # 拇指
            for i in range(4):
                hand_connections.append((hand_offset + i, hand_offset + i + 1))
            # 其他四指
            for finger in range(1, 5):
                base = hand_offset + 1 + finger * 4
                for i in range(3):
                    hand_connections.append((base + i, base + i + 1))
        
        return body_connections + hand_connections
    
    def visualize_pose_sequence(self, poses: np.ndarray, output_path: str = None, 
                              title: str = "Pose Sequence"):
        """可视化姿态序列"""
        if len(poses.shape) != 2:
            raise ValueError("Poses should be 2D array [seq_len, pose_dim]")
        
        seq_len, pose_dim = poses.shape
        
        # 重塑为关键点格式 [seq_len, num_keypoints, 3]
        num_keypoints = pose_dim // 3
        keypoints = poses.reshape(seq_len, num_keypoints, 3)
        
        # 创建图形
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        # 选择要显示的帧
        frame_indices = np.linspace(0, seq_len - 1, 10, dtype=int)
        
        for i, frame_idx in enumerate(frame_indices):
            ax = axes[i]
            self._plot_single_pose(keypoints[frame_idx], ax, f"Frame {frame_idx}")
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def _plot_single_pose(self, keypoints: np.ndarray, ax, title: str):
        """绘制单个姿态"""
        # 提取x, y坐标
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        confidence = keypoints[:, 2]
        
        # 只显示置信度高的关键点
        valid_mask = confidence > 0.5
        
        # 绘制关键点
        ax.scatter(x[valid_mask], y[valid_mask], c='red', s=30, alpha=0.7)
        
        # 绘制连接线
        for start_idx, end_idx in self.connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                valid_mask[start_idx] and valid_mask[end_idx]):
                ax.plot([x[start_idx], x[end_idx]], 
                       [y[start_idx], y[end_idx]], 
                       'b-', linewidth=2, alpha=0.6)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    def create_pose_video(self, poses: np.ndarray, output_path: str, fps: int = 30):
        """创建姿态视频"""
        seq_len, pose_dim = poses.shape
        num_keypoints = pose_dim // 3
        keypoints = poses.reshape(seq_len, num_keypoints, 3)
        
        # 视频参数
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_idx in range(seq_len):
            # 创建空白帧
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # 绘制姿态
            frame = self._draw_pose_on_frame(frame, keypoints[frame_idx])
            
            video_writer.write(frame)
        
        video_writer.release()
        logger.info(f"Pose video saved to {output_path}")
    
    def _draw_pose_on_frame(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """在帧上绘制姿态"""
        height, width = frame.shape[:2]
        
        # 转换坐标到像素空间
        x = (keypoints[:, 0] * width).astype(int)
        y = (keypoints[:, 1] * height).astype(int)
        confidence = keypoints[:, 2]
        
        # 绘制关键点
        for i, (px, py, conf) in enumerate(zip(x, y, confidence)):
            if conf > 0.5 and 0 <= px < width and 0 <= py < height:
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
        
        # 绘制连接线
        for start_idx, end_idx in self.connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                confidence[start_idx] > 0.5 and confidence[end_idx] > 0.5):
                
                start_point = (x[start_idx], y[start_idx])
                end_point = (x[end_idx], y[end_idx])
                
                if (0 <= start_point[0] < width and 0 <= start_point[1] < height and
                    0 <= end_point[0] < width and 0 <= end_point[1] < height):
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        
        return frame


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        output_path: str = None):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 学习率曲线（如果有的话）
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_evaluation_metrics(metrics: Dict[str, List[float]], output_path: str = None):
    """绘制评估指标"""
    num_metrics = len(metrics)
    cols = min(3, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    plt.figure(figsize=(5 * cols, 4 * rows))
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.subplot(rows, cols, i + 1)
        plt.plot(values, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} over time')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_multilingual_comparison(results_by_language: Dict[str, Dict[str, float]], 
                                 output_path: str = None):
    """创建多语言性能对比图"""
    languages = list(results_by_language.keys())
    metrics = list(next(iter(results_by_language.values())).keys())
    
    # 创建热力图数据
    data = []
    for lang in languages:
        row = [results_by_language[lang].get(metric, 0) for metric in metrics]
        data.append(row)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(data, 
                xticklabels=metrics, 
                yticklabels=languages,
                annot=True, 
                fmt='.3f', 
                cmap='viridis')
    
    plt.title('Performance Comparison Across Languages')
    plt.xlabel('Metrics')
    plt.ylabel('Languages')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def create_default_config() -> Dict:
        """创建默认配置"""
        return {
            "model": {
                "languages": ["ASL", "DGS", "KSL", "DSGS", "LSF-CH", "LIS-CH", "LSA", "TSL"],
                "gloss_vocab_size": 10000,
                "hidden_dim": 1024,
                "pose_dim": 150
            },
            "data": {
                "data_dirs": {},
                "languages": ["ASL", "DGS"],
                "batch_size": 8,
                "num_workers": 4,
                "max_sequence_length": 256
            },
            "training": {
                "mode": "mlsf",
                "epochs": 100,
                "grad_clip": 1.0,
                "log_interval": 100,
                "val_interval": 1,
                "save_interval": 10
            },
            "optimizer": {
                "type": "adamw",
                "lr": 1e-4,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999]
            },
            "scheduler": {
                "type": "cosine",
                "eta_min": 1e-6
            },
            "loss": {
                "use_rl_loss": True,
                "alpha": 0.1,
                "beta": 0.1
            },
            "output_dir": "./outputs",
            "seed": 42,
            "use_wandb": False,
            "use_tensorboard": True,
            "wandb_project": "signllm",
            "experiment_name": None
        }
    
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """验证配置文件"""
        required_keys = ["model", "data", "training", "optimizer"]
        
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required config key: {key}")
                return False
        
        # 验证模型配置
        model_config = config["model"]
        if "languages" not in model_config or not model_config["languages"]:
            logger.error("Model config must specify languages")
            return False
        
        # 验证数据配置
        data_config = config["data"]
        if "data_dirs" not in data_config:
            logger.error("Data config must specify data_dirs")
            return False
        
        return True


def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def get_device_info() -> Dict[str, Any]:
    """获取设备信息"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name()
        info["memory_allocated"] = torch.cuda.memory_allocated()
        info["memory_reserved"] = torch.cuda.memory_reserved()
    
    return info


if __name__ == "__main__":
    # 测试工具函数
    print("Testing utility functions...")
    
    # 测试设备信息
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # 测试配置管理
    config_manager = ConfigManager()
    default_config = config_manager.create_default_config()
    print(f"Default config created with {len(default_config)} sections")
    
    # 测试姿态可视化
    visualizer = PoseVisualizer()
    test_poses = np.random.randn(50, 150)
    print(f"Created test poses with shape: {test_poses.shape}")
    
    print("All tests passed!")
