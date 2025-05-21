import os
import json
import random
import math
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

__all__ = [
    "set_seed",
    "online_mean_std",
    "compute_loss",
    "save_ckpt",
    "load_ckpt",
    "render_pose_sequence",
    "render_animation_from_json",
]

# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
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
