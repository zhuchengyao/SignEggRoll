# visualize_pose.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

# OpenPose 骨架连线规则
POSE_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 头部和右臂
    (1, 5), (5, 6), (6, 7),                # 左臂
    (1, 8), (8, 9), (9, 10),               # 躯干
    (10, 11), (11, 24), (11, 22),          # 右腿
    (10, 12), (12, 13), (13, 14),          # 左腿
    (0, 15), (15, 17), (0, 16), (16, 18)   # 耳朵、眼睛等
]

# 手部骨架连线规则
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
]

def render_pose_sequence(pose_sequence, output_path, fps=15):
    out_dir = os.path.dirname(output_path)
    if out_dir not in ("", "."):
        os.makedirs(out_dir, exist_ok=True)

    # 计算每个部分的关键点数量
    POSE_POINTS = 25  # OpenPose 25个关键点
    HAND_POINTS = 21  # 每只手21个关键点
    
    # 重塑数据为 [T, total_points, 2]
    frames = pose_sequence.reshape(-1, POSE_POINTS + 2 * HAND_POINTS, 2)
    T = frames.shape[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 创建散点图（不同颜色区分不同部分）
    pose_scat = ax.scatter([], [], c='red', label='Pose')
    left_hand_scat = ax.scatter([], [], c='blue', label='Left Hand')
    right_hand_scat = ax.scatter([], [], c='green', label='Right Hand')
    
    # 创建连线
    pose_lines = [ax.plot([], [], c='red')[0] for _ in POSE_SKELETON]
    left_hand_lines = [ax.plot([], [], c='blue')[0] for _ in HAND_SKELETON]
    right_hand_lines = [ax.plot([], [], c='green')[0] for _ in HAND_SKELETON]

    def init():
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.legend()
        return pose_scat, left_hand_scat, right_hand_scat, *pose_lines, *left_hand_lines, *right_hand_lines

    def update(i):
        keypoints = frames[i]
        
        # 分离不同部分的关键点
        pose_points = keypoints[:POSE_POINTS]
        left_hand_points = keypoints[POSE_POINTS:POSE_POINTS + HAND_POINTS]
        right_hand_points = keypoints[POSE_POINTS + HAND_POINTS:]
        
        # 更新散点
        pose_scat.set_offsets(pose_points)
        left_hand_scat.set_offsets(left_hand_points)
        right_hand_scat.set_offsets(right_hand_points)
        
        # 更新骨架连线
        # 1. 身体骨架
        for idx, (j, k) in enumerate(POSE_SKELETON):
            if j < POSE_POINTS and k < POSE_POINTS:
                pose_lines[idx].set_data(
                    [pose_points[j, 0], pose_points[k, 0]],
                    [1000 - pose_points[j, 1], 1000 - pose_points[k, 1]]
                )
        
        # 2. 左手骨架
        for idx, (j, k) in enumerate(HAND_SKELETON):
            if j < HAND_POINTS and k < HAND_POINTS:
                left_hand_lines[idx].set_data(
                    [left_hand_points[j, 0], left_hand_points[k, 0]],
                    [1000 - left_hand_points[j, 1], 1000 - left_hand_points[k, 1]]
                )
        
        # 3. 右手骨架
        for idx, (j, k) in enumerate(HAND_SKELETON):
            if j < HAND_POINTS and k < HAND_POINTS:
                right_hand_lines[idx].set_data(
                    [right_hand_points[j, 0], right_hand_points[k, 0]],
                    [1000 - right_hand_points[j, 1], 1000 - right_hand_points[k, 1]]
                )
        
        return pose_scat, left_hand_scat, right_hand_scat, *pose_lines, *left_hand_lines, *right_hand_lines

    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, interval=1000/fps, blit=True)
    ani.save(output_path, writer='ffmpeg', fps=fps)
    plt.close()

    print(f"✅ 骨架动画已保存至: {output_path}")

if __name__ == "__main__":
    import json

    # 指定 JSON 文件路径
    json_path = "inference_ar_output/_-adcxjm1R4_0-8-rgb_front_gen.json"
    output_path = "./output/demo1.mp4"

    # 加载 pose 序列
    with open(json_path, "r") as f:
        data = json.load(f)
    pose_sequence = np.array(data["pose"])  # [T, D]

    # 渲染骨架动画
    render_pose_sequence(pose_sequence, output_path)
