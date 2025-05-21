import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# ------------------------------
# 骨架连接规则（OpenPose BODY_25 核心）
# ------------------------------
BODY_PAIRS = [
    (0, 1),  # 鼻尖–颈部
    (1, 2), (2, 3), (3, 4),      # 右臂
    (1, 5), (5, 6), (6, 7),      # 左臂
    (1, 8),                      # 颈部–躯干中心
    (8, 9), (9, 10), (10, 11),   # 右腿
    (8, 12), (12, 13), (13, 14), # 左腿
    (0, 15), (15, 17),           # 鼻尖–右眼–右耳
    (0, 16), (16, 18)            # 鼻尖–左眼–左耳
]

# ------------------------------
# 手部（左右手共用）
# ------------------------------
HAND_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# ------------------------------
# 面部 68+2 点常用连线
# ------------------------------
FACE_PAIRS = [
    *[(i, i+1) for i in range(0, 16)],             # 脸轮廓
    *[(i, i+1) for i in range(17, 21)],            # 右眉
    *[(i, i+1) for i in range(22, 26)],            # 左眉
    *[(i, i+1) for i in range(36, 41)], (41, 36),  # 右眼
    *[(i, i+1) for i in range(42, 47)], (47, 42),  # 左眼
    *[(i, i+1) for i in range(27, 30)],            # 鼻梁
    *[(i, i+1) for i in range(30, 35)], (35, 30),  # 鼻翼
    *[(i, i+1) for i in range(48, 59)], (59, 48),  # 外嘴唇
    *[(i, i+1) for i in range(60, 67)], (67, 60)   # 内嘴唇
]

def render_pose_sequence(pose_sequence, output_path, fps=15):
    T = len(pose_sequence)
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(i):
        ax.clear()
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(f'Frame {i+1}/{T}')

        frame = pose_sequence[i]
        # reshape 为 N×3
        body = np.array(frame["pose_keypoints_2d"]).reshape(-1, 3)
        face = np.array(frame.get("face_keypoints_2d", [])).reshape(-1, 3)
        lh   = np.array(frame["hand_left_keypoints_2d"]).reshape(-1, 3)
        rh   = np.array(frame["hand_right_keypoints_2d"]).reshape(-1, 3)

        # 画 body
        for a, b in BODY_PAIRS:
            if body[a,2]>0.1 and body[b,2]>0.1:
                ax.plot([body[a,0], body[b,0]], [body[a,1], body[b,1]], 'b-', lw=2)
        ax.scatter(body[:,0], body[:,1], c='b', s=20)

        # 画 face
        if face.size:
            for a, b in FACE_PAIRS:
                if face[a,2]>0.01 and face[b,2]>0.01:
                    ax.plot([face[a,0], face[b,0]], [face[a,1], face[b,1]], 'm-', lw=1)
            ax.scatter(face[:,0], face[:,1], c='m', s=8)

        # 画左手
        for a, b in HAND_PAIRS:
            if lh[a,2]>0.1 and lh[b,2]>0.1:
                ax.plot([lh[a,0], lh[b,0]], [lh[a,1], lh[b,1]], 'r-', lw=2)
        ax.scatter(lh[:,0], lh[:,1], c='r', s=10)

        # 画右手
        for a, b in HAND_PAIRS:
            if rh[a,2]>0.1 and rh[b,2]>0.1:
                ax.plot([rh[a,0], rh[b,0]], [rh[a,1], rh[b,1]], 'g-', lw=2)
        ax.scatter(rh[:,0], rh[:,1], c='g', s=10)

    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=1000/fps, blit=False
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='ffmpeg', fps=fps)
    plt.close()
    print(f"✅ 已保存动画：{output_path}")

# ------------------------------
# 运行示例
# ------------------------------
if __name__ == "__main__":
    json_path   = "./inference_ar_output/_-adcxjm1R4_1-8-rgb_front_gen.json"
    # json_path   = "./datasets/processed/_-adcxjm1R4_0-8-rgb_front.json"
    output_path = "./output/demo_fixed.mp4"
    with open(json_path, "r") as f:
        data = json.load(f)
    # 假设 data["pose"] 是 list of frames，每帧含 3 个 key
    print("顶层 keys:", data.keys())                
    print("第一帧 keys:", data["pose"][0].keys())  
    print("face_keypoints_2d 长度:", len(data["pose"][0].get("face_keypoints_2d", [])))
    render_pose_sequence(data["pose"], output_path)
