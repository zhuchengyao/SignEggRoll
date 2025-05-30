#!/usr/bin/env python3
"""
最小化 SignLLM 训练脚本 - 解决数据格式问题 (修订版)

* 使用 DataLoader 替代手动批处理
* 重新实例化 CONFIG 而不是直接调用 __init__
* 引入混合精度训练 (AMP)
* 保存当前 CONFIG 到 checkpoint

其余逻辑保持与原脚本一致。
"""

import os
import sys
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from signllm_model import SignLLM, ModelConfig, CONFIG  # 文件名保持不变
from data_processor import MultilingualSignDataset


# --------------------------- Collate 函数 --------------------------- #

def collate_fn(batch: List[Dict]):
    """将 dataset 返回的样本列表整理成批次"""
    texts = [sample["text"] for sample in batch]
    poses = [sample["pose_sequence"] for sample in batch]
    poses = torch.stack(poses)  # 假设 dataset 已保证序列长度一致
    return {"texts": texts, "poses": poses}


# --------------------------- 主函数 --------------------------- #

def main():
    print("🚀 最小化 SignLLM 训练 (修订版)")
    print("=" * 60)

    # 设置模型大小 (可选: "tiny", "small", "medium", "large")
    MODEL_SIZE = "medium"

    # 重新实例化 CONFIG, 并替换全局引用
    global CONFIG
    CONFIG = ModelConfig(MODEL_SIZE)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")

    # 创建模型
    print("📦 创建模型…")
    model = SignLLM(languages=["ASL"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 参数量: {total_params:,} ({total_params/1_000_000:.1f}M)")

    # 创建数据集 & 数据加载器
    print("📚 构建数据集…")
    dataset = MultilingualSignDataset(
        data_dirs={"ASL": "datasets/signllm_data_complete"},
        languages=["ASL"],
        split="dev",
        max_sequence_length=256,
        pose_dim=CONFIG.pose_dim,
    )
    print(f"📊 样本数: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # 优化器 & 损失
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device.type)

    # 训练
    epoch_num = 10
    print("\n🎯 开始训练…")
    model.train()

    for epoch in range(epoch_num):
        print(f"\n📅 Epoch {epoch + 1}/{epoch_num}")
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            batch_texts = batch["texts"]
            target_poses = batch["poses"].to(device)

            with torch.amp.autocast(device.type):
                pred_poses, _ = model(
                    texts=batch_texts,
                    language="ASL",
                    mode="mlsf",
                    max_length=target_poses.size(1),
                )
                loss = criterion(pred_poses, target_poses)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"📊 Epoch {epoch + 1} 平均损失: {avg_loss:.6f}")

        # 保存检查点
        checkpoint_dir = Path("checkpoints/eggroll_train")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch + 1,
            "config": CONFIG.__dict__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }
        ckpt_path = checkpoint_dir / f"epoch_{epoch + 1}.pth"
        torch.save(checkpoint, ckpt_path)
        print(f"💾 已保存检查点: {ckpt_path}")

    print("\n✅ 训练完成！")

    # 简单推理测试
    print("\n🔍 推理测试…")
    model.eval()
    with torch.no_grad():
        test_texts = ["Hello world"]
        test_poses, _ = model(texts=test_texts, language="ASL", mode="prompt2langgloss")
        print(f"✅ 推理成功，输出形状: {test_poses.shape}")


if __name__ == "__main__":
    main()
