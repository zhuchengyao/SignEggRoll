# 3D人体姿态生成的扩散模型 (Diffusion Model for 3D Human Pose Generation)

这个项目实现了一个基于扩散模型的3D人体姿态生成系统。模型能够学习人体结构的分布，并通过逐步去噪的过程生成新的、合理的3D姿态。

## 🎯 项目特点

- **扩散模型架构**: 基于DDPM (Denoising Diffusion Probabilistic Models) 实现
- **3D姿态建模**: 针对67个关键点的3D坐标进行建模
- **数据增强**: 包含旋转、缩放、平移等数据增强技术
- **可视化支持**: 提供丰富的可视化和动画生成功能
- **模块化设计**: 代码结构清晰，易于扩展和修改

## 📦 项目结构

```
.
├── diffusion_model.py      # 扩散模型核心实现
├── pose_dataset.py         # 数据加载和预处理
├── train_diffusion.py      # 训练脚本
├── generate_poses.py       # 生成脚本
├── demo.py                 # 快速演示脚本
├── requirements.txt        # 依赖包列表
└── README.md              # 项目说明文档
```

## 🚀 快速开始

### 1. 环境设置

首先安装所需的依赖包：

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 1.10.0
- NumPy >= 1.21.0
- Matplotlib >= 3.3.0
- tqdm, wandb (可选)

### 2. 数据准备

确保你的数据采用 `.skels` 格式：
- 每行包含 67×3=201 个浮点数
- 代表67个关键点的x,y,z坐标
- 每一帧作为一行

数据格式示例：
```
x1 y1 z1 x2 y2 z2 ... x67 y67 z67
x1 y1 z1 x2 y2 z2 ... x67 y67 z67
...
```

### 3. 快速演示

运行演示脚本快速测试整个流程：

```bash
# 测试数据加载（不训练）
python demo.py --data_dir ./datasets/processed --skip_training

# 完整演示（包含快速训练）
python demo.py --data_dir ./datasets/processed --quick_test --epochs 50
```

## 📖 详细使用方法

### 训练模型

```bash
python train_diffusion.py --data_dir ./datasets/processed \
                          --batch_size 32 \
                          --num_epochs 1000 \
                          --learning_rate 1e-4 \
                          --model_channels 128 \
                          --device cuda
```

训练参数说明：
- `--data_dir`: 数据目录路径
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--model_channels`: 模型通道数
- `--num_timesteps`: 扩散步数 (默认1000)
- `--no_wandb`: 禁用wandb日志记录

### 生成姿态

```bash
# 生成静态姿态
python generate_poses.py --checkpoint ./checkpoints/best_model.pth \
                         --num_samples 8 \
                         --visualize \
                         --format skels

# 生成动画
python generate_poses.py --checkpoint ./checkpoints/best_model.pth \
                         --animation \
                         --num_frames 30 \
                         --visualize
```

生成参数说明：
- `--checkpoint`: 模型检查点路径
- `--num_samples`: 生成样本数量
- `--output_dir`: 输出目录
- `--format`: 保存格式 (skels/npy/json)
- `--visualize`: 可视化结果
- `--animation`: 生成动画
- `--num_frames`: 动画帧数

### 数据分析

```bash
# 测试数据加载和可视化
python pose_dataset.py
```

## 🔧 模型架构

### 扩散模型 (Diffusion Model)

模型基于高斯扩散过程：

1. **前向过程**: 逐步向数据添加高斯噪声
   ```
   q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
   ```

2. **反向过程**: 训练神经网络学习去噪
   ```
   p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
   ```

### U-Net架构

使用1D U-Net处理序列化的3D姿态数据：

- **编码器**: 下采样路径，提取特征
- **解码器**: 上采样路径，重建姿态
- **跳跃连接**: 保持细节信息
- **注意力机制**: 增强关键点间的关联

### 关键特性

- **时间嵌入**: 正弦位置编码表示扩散时间步
- **残差连接**: 稳定训练过程
- **组归一化**: 提高训练稳定性
- **余弦噪声调度**: 更好的噪声分布

## 📊 数据处理

### 预处理步骤

1. **数据加载**: 从.skels文件读取3D坐标
2. **标准化**: Z-score标准化，均值0方差1
3. **数据增强**: 
   - 随机旋转 (±30°)
   - 随机缩放 (0.9-1.1倍)
   - 随机平移
   - 高斯噪声

### 数据增强

```python
# 旋转增强
rotation_matrix = [[cos_θ, 0, sin_θ],
                   [0, 1, 0],
                   [-sin_θ, 0, cos_θ]]

# 缩放增强
pose *= scale_factor

# 平移增强
pose += translation_vector
```

## 🎨 可视化功能

### 3D姿态可视化

```python
from pose_dataset import visualize_pose_data

# 可视化数据集样本
visualize_pose_data(dataset, num_samples=5)
```

### 生成动画

```python
from generate_poses import create_gif_from_poses

# 创建GIF动画
create_gif_from_poses(pose_sequence, "animation.gif")
```

## ⚙️ 训练技巧

### 超参数调优

推荐的超参数设置：

```python
# 模型参数
model_channels = 128        # 基础通道数
channel_mult = (1, 2, 4, 8) # 通道倍数
num_res_blocks = 2          # 残差块数量

# 训练参数
batch_size = 32            # 批次大小
learning_rate = 1e-4       # 学习率
num_epochs = 1000          # 训练轮数
num_timesteps = 1000       # 扩散步数

# 调度器
scheduler = "cosine"       # 余弦调度
```

### 训练监控

使用Wandb监控训练过程：

```python
# 记录损失
wandb.log({"train_loss": loss})

# 记录生成样本
wandb.log({"samples": wandb.Image("samples.png")})
```

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   python train_diffusion.py --batch_size 16
   
   # 使用CPU
   python train_diffusion.py --device cpu
   ```

2. **数据加载失败**
   ```bash
   # 检查数据格式
   python pose_dataset.py
   
   # 限制文件数量
   python train_diffusion.py --max_files 10
   ```

3. **训练不稳定**
   ```bash
   # 降低学习率
   python train_diffusion.py --learning_rate 5e-5
   
   # 使用较少的扩散步数
   python train_diffusion.py --num_timesteps 500
   ```

### 调试模式

使用调试参数进行快速测试：

```bash
python train_diffusion.py --data_dir ./datasets/processed \
                          --max_files 2 \
                          --num_epochs 10 \
                          --batch_size 4 \
                          --no_wandb
```

## 🚀 扩展功能

### 条件生成

可以扩展模型支持条件生成：

```python
# 添加条件嵌入
class ConditionalUNet1D(UNet1D):
    def __init__(self, condition_dim, **kwargs):
        super().__init__(**kwargs)
        self.condition_proj = nn.Linear(condition_dim, self.model_channels)
    
    def forward(self, x, timesteps, condition=None):
        # 融合条件信息
        if condition is not None:
            cond_emb = self.condition_proj(condition)
            # 将条件嵌入融合到网络中
```

### 序列生成

扩展为时序姿态生成：

```python
# 时序扩散模型
class TemporalDiffusion(GaussianDiffusion):
    def __init__(self, sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
    
    def sample_sequence(self, model, num_samples=1):
        # 生成连续的姿态序列
        pass
```

## 📚 参考文献

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)

## 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件

## 🤝 贡献

欢迎提交问题和拉取请求！请确保：

1. 代码符合项目风格
2. 添加适当的测试
3. 更新相关文档

## 📞 联系方式

如有问题或建议，请提交GitHub Issue或联系项目维护者。

---

**快速开始命令总结：**

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 快速演示
python demo.py --data_dir ./datasets/processed --quick_test

# 3. 完整训练
python train_diffusion.py --data_dir ./datasets/processed

# 4. 生成姿态
python generate_poses.py --checkpoint ./checkpoints/best_model.pth --visualize
``` 