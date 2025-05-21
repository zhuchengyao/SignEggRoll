import json
import numpy as np
import matplotlib.pyplot as plt

# 手部骨架连线关系（OpenPose 21 点手部模型）
HAND_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),      # 食指
    (0, 9), (9,10), (10,11),(11,12),     # 中指
    (0,13), (13,14),(14,15),(15,16),     # 无名指
    (0,17), (17,18),(18,19),(19,20)      # 小指
]

# 1. 读取 JSON 文件
with open("./datasets/openpose_output/json/_-adcxjm1R4_0-8-rgb_front/_-adcxjm1R4_0-8-rgb_front_000000000092_keypoints.json") as f:  # 替换为你的文件名
    data = json.load(f)

# 2. 提取左手关键点，转换为 (21, 3) 的数组：[x, y, confidence]
left = data['people'][0]['hand_right_keypoints_2d']
kp = np.array(left).reshape(-1, 3)

# 3. 分离坐标与置信度
xs, ys, cs = kp[:,0], kp[:,1], kp[:,2]

# 4. 绘图
plt.figure(figsize=(6,6))
# 绘制点
for idx, (x, y, c) in enumerate(kp):
    if c < 0.1:  # 置信度过低的点可视化时跳过
        continue
    plt.scatter(x, y, s=30, edgecolors='k')
    plt.text(x+2, y+2, str(idx), fontsize=8)

# 绘制骨架连线
for i, j in HAND_PAIRS:
    if kp[i,2] > 0.1 and kp[j,2] > 0.1:
        plt.plot([kp[i,0], kp[j,0]], [kp[i,1], kp[j,1]], linewidth=2)

plt.gca().invert_yaxis()   # 图像坐标系原点在左上，需反转 y 轴
plt.axis('equal')
plt.title('Left Hand Keypoints')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()
