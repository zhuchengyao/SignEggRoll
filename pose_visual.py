import json
import matplotlib.pyplot as plt
import numpy as np

# ======== OpenPose 关键点连线规则 ========
BODY_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17), (0, 16), (16, 18)
]

FACE_PAIRS = [
    (17, 18), (18, 19), (19, 20), (20, 21),  # 右眉
    (22, 23), (23, 24), (24, 25), (25, 26),  # 左眉
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),  # 右眼
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),  # 左眼
    (48, 49), (49, 50), (50, 51), (51, 52),
    (52, 53), (53, 54), (54, 55), (55, 56),
    (56, 57), (57, 58), (58, 59), (59, 48)   # 外唇轮廓
]

HAND_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),      # 食指
    (0, 9), (9,10), (10,11),(11,12),     # 中指
    (0,13), (13,14),(14,15),(15,16),     # 无名指
    (0,17), (17,18),(18,19),(19,20)      # 小指
]

def draw_keypoints(kps, pairs, ax, color='b'):
    kps = np.array(kps).reshape(-1, 3)
    for pair in pairs:
        pt1, pt2 = pair
        if kps[pt1, 2] > 0.05 and kps[pt2, 2] > 0.05:
            ax.plot([kps[pt1, 0], kps[pt2, 0]], [kps[pt1, 1], kps[pt2, 1]], color=color, linewidth=2)
    # 画点
    ax.scatter(kps[:, 0], kps[:, 1], c=color, s=10)

# ========== 加载你提供的 JSON 数据 ==========
with open("./inference_ar_output/_-adcxjm1R4_0-8-rgb_front_gen.json") as f:  # 替换为你的文件名
    data = json.load(f)

# person = data['people'][0]
# pose_kps = person['pose_keypoints_2d']
# face_kps = person['face_keypoints_2d']
# hand_l_kps = person['hand_left_keypoints_2d']
# hand_r_kps = person['hand_right_keypoints_2d']

# # ========== 可视化 ==========
# fig, ax = plt.subplots(figsize=(8, 10))
# ax.set_title("Full Human Pose with Face and Hands")
# ax.invert_yaxis()  # 图像Y轴反转，符合图像坐标习惯

# # 各部位绘制
# draw_keypoints(pose_kps, BODY_PAIRS, ax, color='blue')
# draw_keypoints(hand_l_kps, HAND_PAIRS, ax, color='green')
# draw_keypoints(hand_r_kps, HAND_PAIRS, ax, color='red')

# # 面部点（不连线，只绘制点）
# face_kps_np = np.array(face_kps).reshape(-1, 3)
# ax.scatter(face_kps_np[:, 0], face_kps_np[:, 1], c='purple', s=5, label='face')

# ax.set_aspect('equal')
# ax.legend()
# plt.tight_layout()
# plt.show()



# 取第一个人的 keypoints
p = data["people"][0]

# 将 flat 列表分块成 (x, y, c) 形式
def split_kps(arr):
    return [(arr[i], arr[i+1], arr[i+2]) for i in range(0, len(arr), 3)]

pose_kps = split_kps(p["pose_keypoints_2d"])
face_kps = split_kps(p["face_keypoints_2d"])
lh_kps   = split_kps(p["hand_left_keypoints_2d"])
rh_kps   = split_kps(p["hand_right_keypoints_2d"])

# ——— ③ 绘图 ———
plt.figure(figsize=(6, 8))
ax = plt.gca()
ax.invert_yaxis()  # 原点在左上角时建议翻转 y 轴

def plot_kps(kps, pairs, marker="o", ms=4):
    xs = [x for x, y, c in kps if c>0]
    ys = [y for x, y, c in kps if c>0]
    ax.scatter(xs, ys, s=ms)
    for i, j in pairs:
        if kps[i][2]>0 and kps[j][2]>0:
            ax.plot([kps[i][0], kps[j][0]],
                    [kps[i][1], kps[j][1]],
                    linewidth=1)

# 身体
plot_kps(pose_kps, BODY_PAIRS)
# 面部
plot_kps(face_kps, FACE_PAIRS, ms=2)
# 左手
plot_kps(lh_kps, HAND_PAIRS, ms=3)
# 右手
plot_kps(rh_kps, HAND_PAIRS, ms=3)

plt.axis("off")
plt.tight_layout()
plt.show()