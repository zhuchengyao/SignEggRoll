# inference_ar.py
# python inference.py \
#   --json_file      ./datasets/processed/_-adcxjm1R4_0-8-rgb_front.json \
#   --checkpoint     ./checkpoints/ar_epoch50.pt \
#   --output_folder  ./inference_ar_output \
#   --init_frames    10 \
#   --gen_steps      50 \
#   --T_max          256 \
#   --device         cuda

import os
import argparse
import json

import torch
from dataset import SignPoseDataset
from model import AutoRegressivePoseModel

def flatten_frame(frame):
    return (
        frame["pose_keypoints_2d"]
        + frame["face_keypoints_2d"]
        + frame["hand_left_keypoints_2d"]
        + frame["hand_right_keypoints_2d"]
    )

def reconstruct_frames(vecs, dims):
    """
    vecs: numpy array of shape [N, D]
    dims: dict with keys 'pose','face','hand_left','hand_right' giving lengths
    returns list of frame dicts
    """
    out = []
    for v in vecs:
        idx = 0
        f = {}
        for key, name in [
            ("pose", "pose_keypoints_2d"),
            ("face", "face_keypoints_2d"),
            ("hand_left", "hand_left_keypoints_2d"),
            ("hand_right", "hand_right_keypoints_2d"),
        ]:
            n = dims[key]
            f[name] = v[idx:idx+n].tolist()
            idx += n
        out.append(f)
    return out

def parse_args():
    p = argparse.ArgumentParser(description="Autoregressive pose inference")
    p.add_argument("--json_file",   type=str, required=True,
                   help="Path to one processed JSON sample")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to your AR model checkpoint (.pt)")
    p.add_argument("--output_folder", type=str, default="./inference_ar_output",
                   help="Where to save the generated JSON")
    p.add_argument("--init_frames", type=int, required=True,
                   help="Number of frames from the sample to use as prompt (≤ original length)")
    p.add_argument("--gen_steps",   type=int, required=True,
                   help="How many new frames to generate")
    p.add_argument("--T_max",       type=int, default=120,
                   help="T_max used in training (must match your model)")
    p.add_argument("--device",      type=str, default=None,
                   help="cuda or cpu; default auto")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.output_folder, exist_ok=True)

    # 1) Load original JSON
    orig = json.load(open(args.json_file, "r", encoding="utf-8"))
    frames = orig["pose"]
    total_len = len(frames)
    k = args.init_frames
    if k > total_len:
        raise ValueError(f"init_frames={k} exceeds sample length {total_len}")
    
    # 2) Compute dims and feature_dim
    dims = {
        "pose":      len(frames[0]["pose_keypoints_2d"]),
        "face":      len(frames[0]["face_keypoints_2d"]),
        "hand_left": len(frames[0]["hand_left_keypoints_2d"]),
        "hand_right":len(frames[0]["hand_right_keypoints_2d"]),
    }
    feature_dim = sum(dims.values())

    # 3) Build model and load weights
    model = AutoRegressivePoseModel(
        feature_dim=feature_dim,
        T_max=args.T_max,     # 现在会是 256
        hidden_dim=2048,      # 跟训练时保持一致
        n_layers=12,          # 跟训练时保持一致
        n_heads=16,
        ff_dim=4096
    ).to(device)

    # **这里做兼容处理**
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # 4) Prepare initial frames tensor [1, k, D]
    init_list = [flatten_frame(fr) for fr in frames[:k]]
    init_tensor = torch.tensor([init_list], dtype=torch.float32, device=device)

    # 5) Generate autoregressively
    with torch.no_grad():
        seq = model.generate(init_tensor, max_gen_steps=args.gen_steps)
        # seq shape: [1, k+1+gen_steps, D], seq[0,0] is start token
    seq_np = seq[0].cpu().numpy()

    # 6) Extract generated frames: positions [k+1 : k+1+gen_steps]
    generated = seq_np[k+1 : k+1 + args.gen_steps]

    # 7) Reconstruct frame dicts
    out_frames = reconstruct_frames(generated, dims)

    # 8) Save output JSON
    out = {
        "id":     orig["id"],
        "text":   orig.get("text", ""),
        "pose":   out_frames,
        "length": len(out_frames)
    }
    base = os.path.splitext(os.path.basename(args.json_file))[0]
    out_path = os.path.join(args.output_folder, f"{base}_gen.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {len(out_frames)} frames. Saved to {out_path}")

if __name__ == "__main__":
    main()
