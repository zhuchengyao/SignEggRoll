import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# 路径配置
CSV_PATH    = "./datasets/how2sign_realigned_train.csv"
JSON_ROOT   = "./datasets/openpose_output/json"
OUTPUT_PATH = "./datasets/processed"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 读取 CSV（自动识别分隔符）
df = pd.read_csv(CSV_PATH, sep=None, engine='python')

# 使用的列名
id_col   = "SENTENCE_NAME"
text_col = "SENTENCE"

def load_pose_sequence(folder_name):
    folder_path = os.path.join(JSON_ROOT, folder_name)
    if not os.path.isdir(folder_path):
        return None
    pose_seq = []
    files = sorted(os.listdir(folder_path))
    for fname in files:
        if not fname.endswith("_keypoints.json"):
            continue
        with open(os.path.join(folder_path, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
        people = data.get("people", [])
        if not people:
            continue
        person = people[0]
        frame = {
            "pose_keypoints_2d":      person.get("pose_keypoints_2d", []),
            "face_keypoints_2d":      person.get("face_keypoints_2d", []),
            "hand_left_keypoints_2d": person.get("hand_left_keypoints_2d", []),
            "hand_right_keypoints_2d":person.get("hand_right_keypoints_2d", [])
        }
        pose_seq.append(frame)
    return pose_seq if pose_seq else None

# 遍历每一行 CSV，构建样本
for _, row in tqdm(df.iterrows(), total=len(df)):
    clip_id = row[id_col]
    out_path = os.path.join(OUTPUT_PATH, f"{clip_id}.json")

    # —— 断点续跑：跳过已存在的文件 —— #
    if os.path.exists(out_path):
        continue

    text = str(row[text_col]).strip()
    pose_seq = load_pose_sequence(clip_id)
    if not pose_seq or len(pose_seq) < 10:
        continue

    sample = {
        "id":     clip_id,
        "text":   text,
        "pose":   pose_seq,
        "length": len(pose_seq)
    }

    # —— 指定 utf-8 编码写文件 —— #
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False)

print("✅ 完成！训练样本保存在:", OUTPUT_PATH)
