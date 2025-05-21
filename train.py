import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SignPoseDataset
from model import AutoRegressivePoseModel  # 请确保 model.py 中定义了这个类

# —— 扩展数据集：返回 (x, length) —— #
class PoseDatasetWithLength(SignPoseDataset):
    def __getitem__(self, idx):
        x = super().__getitem__(idx)  # [S, D]
        fn = self.files[idx]
        with open(fn, "r", encoding="utf-8") as f:
            orig = json.load(f)
        length = orig["length"]
        return x, length


def collate_fn(batch):
    xs, lengths = zip(*batch)
    xs = torch.stack(xs, dim=0)               # [B, S, D]
    lengths = torch.tensor(lengths, dtype=torch.long)
    return xs, lengths


def get_latest_checkpoint(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    epochs = []
    for fn in files:
        if fn.startswith("ar_epoch") and fn.endswith(".pt"):
            try:
                num = int(fn[len("ar_epoch"):-len(".pt")])
                epochs.append(num)
            except:
                pass
    if not epochs:
        return None, 0
    last = max(epochs)
    return os.path.join(checkpoint_dir, f"ar_epoch{last}.pt"), last


def main():
    # ——— 配置 ———
    json_folder    = "./datasets/processed"
    checkpoint_dir = "./model"
    T_max          = 256        # 同 SignPoseDataset 中的设定
    batch_size     = 8
    epochs_total   = 50
    lr             = 1e-4
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ——— 数据集 & DataLoader ———
    ds = PoseDatasetWithLength(json_folder, T_max=T_max)
    sample_x, _ = ds[0]                     # Tensor [S, D]
    S, feature_dim = sample_x.shape
    print(f"[Info] sequence length S={S}, feature_dim={feature_dim}")
    
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # ——— 模型 & 优化器 ———
    model = AutoRegressivePoseModel(
        feature_dim=feature_dim,
        T_max=T_max,
        hidden_dim=2048,
        n_layers=12,
        n_heads=16,
        ff_dim=4096
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # ——— 尝试加载已有 checkpoint —— #
    ckpt_path, last_epoch = get_latest_checkpoint(checkpoint_dir)
    start_epoch = 1
    if ckpt_path:
        print(f"[Resume] Loading checkpoint from epoch {last_epoch}: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = last_epoch + 1
        if start_epoch > epochs_total:
            print(f"Already trained {last_epoch} epochs, which exceeds total {epochs_total}. Exiting.")
            return

    # —— training loop —— #
    for epoch in range(start_epoch, epochs_total + 1):
        model.train()
        total_loss = 0.0
        total_frames = 0

        pbar = tqdm(loader, desc=f"[Epoch {epoch}/{epochs_total}]")
        for x, lengths in pbar:
            x = x.to(device)
            lengths = lengths.to(device)

            # —— 动态截断 ——  
            max_L = (lengths + 1).max().item()   # +1 包括 start token
            x = x[:, :max_L, :]                  # [B, max_L, D]
            lengths = torch.clamp(lengths, max=max_L-1)

            # Teacher forcing 前向
            y_pred = model.forward_train(x, lengths)  # [B,S,D]

            # 构造 MSE loss mask：t=1..length
            B, S, D = y_pred.shape
            mask = torch.zeros(B, S, device=device, dtype=torch.bool)
            for i, L in enumerate(lengths):
                mask[i, 1:(L + 1)] = True

            # 计算 MSE loss
            diff2 = (y_pred - x).pow(2)              # [B,S,D]
            diff2 = diff2 * mask.unsqueeze(-1)      # [B,S,D]
            loss = diff2.sum() / (mask.sum() * D)   # scalar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_frames += mask.sum().item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / total_frames
        print(f"Epoch {epoch} ─ avg MSE/frame: {avg_loss:.6f}")

        # 1) 保存完整 checkpoint（可恢复训练）
        ckpt_path = os.path.join(checkpoint_dir, f"ar_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss
        }, ckpt_path)

        # 2) 单独保存纯模型权重（部署/加载）
        weights_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}.pth")
        torch.save(model.state_dict(), weights_path)

    print("✅ 自回归模型训练完毕！")

if __name__ == "__main__":
    main()
