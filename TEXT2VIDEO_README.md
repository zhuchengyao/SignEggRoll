# Text-to-Video扩散模型使用指南

这个指南将帮助您将现有的3D姿态生成项目改造为text-to-video功能。

## 🔄 主要变化

### 从3D姿态生成到Text-to-Video的改造

| 方面 | 原项目 (3D姿态) | 新项目 (Text-to-Video) |
|------|----------------|----------------------|
| **数据格式** | `(67, 3)` 单帧姿态 | `(3, T, H, W)` 视频序列 |
| **模型架构** | 1D U-Net | 3D U-Net + 交叉注意力 |
| **输入条件** | 无条件生成 | 文本条件化生成 |
| **输出** | 3D关键点坐标 | RGB视频帧 |
| **训练数据** | `.skels`文件 | 视频-文本配对 |

## 📦 安装依赖

```bash
pip install -r requirements_text2video.txt
```

## 🗂️ 数据准备

### 数据目录结构

```
your_video_data/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── captions/
    ├── video_001.txt
    ├── video_002.txt
    └── ...
```

### 创建演示数据集

```bash
python video_dataset.py
```

这将创建一个包含简单移动方块的演示数据集，用于测试系统。

## 🚀 使用方法

### 1. 训练模型

```bash
# 基础训练
python train_text2video.py --data_dir ./your_video_data --batch_size 2 --num_epochs 1000

# 调试模式（小数据集）
python train_text2video.py --data_dir ./demo_video_data --batch_size 1 --num_epochs 100 --max_samples 5 --no_wandb

# 自定义参数
python train_text2video.py \
    --data_dir ./your_video_data \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_epochs 2000 \
    --num_frames 16 \
    --frame_size 64 \
    --model_channels 128
```

### 2. 生成视频

#### 命令行模式

```bash
# 单个视频生成
python generate_text2video.py \
    --checkpoint ./checkpoints/text2video_xxx/best_model.pth \
    --prompts "A red car driving on a highway" \
    --output_dir ./generated_videos \
    --save_gif --visualize

# 批量生成
python generate_text2video.py \
    --checkpoint ./checkpoints/text2video_xxx/best_model.pth \
    --prompts "A cat playing with a ball" "A person walking in a park" "Ocean waves at sunset" \
    --output_dir ./generated_videos \
    --save_frames --save_gif --visualize
```

#### 交互式模式

```bash
python generate_text2video.py
```

然后按提示输入模型路径和文本描述。

## 🏗️ 模型架构详解

### 核心组件

1. **文本编码器 (TextEncoder)**
   - 基于CLIP模型
   - 将文本转换为768维向量序列
   - 支持最长77个token

2. **3D U-Net (UNet3D)**
   - 处理5D张量 `(B, C, T, H, W)`
   - 包含时间注意力机制
   - 交叉注意力融合文本条件

3. **扩散过程 (TextToVideoDiffusion)**
   - 扩展到视频维度的DDPM
   - 支持文本条件化生成
   - 余弦噪声调度

### 关键特性

- **交叉注意力**: 在每个残差块中融合文本信息
- **时间建模**: 沿时间轴的自注意力机制
- **3D卷积**: 同时处理空间和时间维度
- **可扩展性**: 支持不同的视频长度和分辨率

## ⚙️ 配置参数

### 训练参数

```python
# 模型配置
model_channels = 128        # 基础通道数
num_frames = 16            # 视频帧数
frame_size = 64            # 帧分辨率
batch_size = 4             # 批次大小

# 训练配置
learning_rate = 1e-4       # 学习率
num_epochs = 1000          # 训练轮数
num_timesteps = 1000       # 扩散步数

# 数据配置
normalize = True           # 标准化到[-1, 1]
augment = True            # 数据增强
```

### 生成参数

```python
num_inference_steps = 1000  # 推理步数
fps = 8                    # 输出视频帧率
save_gif = True           # 保存GIF动画
save_frames = True        # 保存单独帧
```

## 📊 性能优化

### 内存优化

1. **降低分辨率**: 使用较小的`frame_size`
2. **减少帧数**: 使用较少的`num_frames`
3. **减小批次**: 降低`batch_size`
4. **梯度累积**: 在小批次上累积梯度

### 训练技巧

1. **预训练权重**: 从2D扩散模型初始化
2. **渐进训练**: 先训练低分辨率，再精调高分辨率
3. **混合精度**: 使用`accelerate`库加速训练
4. **学习率调度**: 使用余弦退火调度器

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   python train_text2video.py --batch_size 1
   
   # 减小模型大小
   python train_text2video.py --model_channels 64 --frame_size 32
   ```

2. **文本编码器错误**
   ```bash
   # 确保网络连接正常，CLIP模型会自动下载
   pip install transformers --upgrade
   ```

3. **视频加载失败**
   ```bash
   # 检查OpenCV安装
   pip install opencv-python --upgrade
   
   # 检查视频格式支持
   python -c "import cv2; print(cv2.getBuildInformation())"
   ```

### 调试模式

```bash
# 使用小数据集快速测试
python train_text2video.py \
    --data_dir ./demo_video_data \
    --max_samples 3 \
    --num_epochs 10 \
    --batch_size 1 \
    --no_wandb
```

## 📈 性能监控

### Wandb集成

```python
# 启用wandb日志
python train_text2video.py --data_dir ./data  # 默认启用

# 禁用wandb
python train_text2video.py --data_dir ./data --no_wandb
```

### 监控指标

- 训练损失 (MSE Loss)
- 验证损失
- 学习率变化
- 生成样本质量
- 视频帧序列

## 🎨 高级功能

### 自定义数据集

```python
# 继承TextVideoDataset类
class CustomVideoDataset(TextVideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def augment_text(self, text):
        # 自定义文本增强
        return enhanced_text
    
    def augment_video(self, frames):
        # 自定义视频增强
        return enhanced_frames
```

### 模型扩展

```python
# 添加更多条件信息
class ConditionalUNet3D(UNet3D):
    def __init__(self, additional_condition_dim, **kwargs):
        super().__init__(**kwargs)
        self.condition_proj = nn.Linear(additional_condition_dim, self.model_channels)
```

## 📚 参考资料

### 相关论文

1. **Video Diffusion Models** - Ho et al.
2. **Imagen Video** - Google Research
3. **Make-A-Video** - Meta AI
4. **Text2Video-Zero** - Microsoft Research

### 代码资源

- [Diffusers Library](https://github.com/huggingface/diffusers)
- [Video Diffusion PyTorch](https://github.com/lucidrains/video-diffusion-pytorch)
- [CLIP](https://github.com/openai/CLIP)

## 🚧 未来改进

### 短期目标

1. **更高分辨率**: 支持256x256或512x512
2. **更长视频**: 支持32帧或64帧
3. **更好的文本理解**: 使用更大的语言模型

### 长期目标

1. **视频编辑**: 支持局部编辑和风格转换
2. **多模态条件**: 结合音频、深度等信息
3. **实时生成**: 优化推理速度

## 💡 使用建议

### 数据准备建议

1. **视频质量**: 使用高质量、无水印的视频
2. **文本描述**: 确保描述准确、详细
3. **数据多样性**: 包含不同场景、动作、风格
4. **数据清洗**: 去除损坏或不相关的数据

### 训练建议

1. **从小规模开始**: 先用小数据集验证流程
2. **监控过拟合**: 定期检查验证损失
3. **保存检查点**: 定期保存模型以防意外中断
4. **实验记录**: 使用wandb记录实验参数和结果

---

**祝您成功实现text-to-video功能！如有问题，请检查日志输出或提交issue。** 