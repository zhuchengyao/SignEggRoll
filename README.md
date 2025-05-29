# SignLLM: Sign Language Production Large Language Models

这是SignLLM论文的完整复现实现，支持多语言手语生成，包含MLSF和Prompt2LangGloss两种模式。

## 📋 目录

- [论文简介](#论文简介)
- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [推理使用](#推理使用)
- [评估](#评估)
- [项目结构](#项目结构)
- [复现结果](#复现结果)

## 📖 论文简介

SignLLM是一个多语言手语生成大语言模型，主要特点包括：

- **两种生成模式**：
  - MLSF (Multi-Language Switching Framework): 多语言切换框架
  - Prompt2LangGloss: 提示到语言gloss模式

- **强化学习组件**：
  - Priority Learning Channel (PLC): 优先级学习通道
  - RL损失函数：提高生成质量

- **多语言支持**：支持8种手语
  - ASL (美国手语)
  - DGS (德国手语)
  - KSL (韩国手语)
  - DSGS (德语瑞士手语)
  - LSF-CH (法语瑞士手语)
  - LIS-CH (意大利语瑞士手语)
  - LSA (阿根廷手语)
  - TSL (土耳其手语)

## 🛠 环境安装

### 1. 克隆仓库
```bash
git clone <repository-url>
cd signllm
```

### 2. 创建虚拟环境
```bash
conda create -n signllm python=3.8
conda activate signllm
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 安装额外依赖
```bash
# 安装fastdtw用于DTW计算
pip install fastdtw

# 安装NLTK数据
python -c "import nltk; nltk.download('punkt')"
```

## 📊 数据准备

### 1. 数据格式

SignLLM使用标准化的姿态数据格式：
- **姿态维度**: 150维 (上身8个关键点 + 双手42个关键点)
- **数据格式**: JSON格式，包含姿态序列和对应文本

### 2. 数据处理

#### 从原始视频处理数据：
```python
from data_processor import DataProcessor

# 创建数据处理器
processor = DataProcessor("./processed_data")

# 处理ASL数据
processor.process_video_dataset(
    video_dir="./raw_data/ASL/videos",
    annotation_file="./raw_data/ASL/annotations.json",
    language="ASL",
    split="train"
)
```

#### 使用现有的Prompt2Sign数据：
```bash
# 下载Prompt2Sign数据集
# 参考：https://signllm.github.io/Prompt2Sign/

# 将数据放置在以下目录结构：
datasets/
├── processed/
│   ├── ASL/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── DGS/
│   └── ...
```

### 3. 数据目录结构
```
datasets/processed/
├── ASL/
│   ├── train/
│   │   ├── sample_000001/
│   │   │   ├── text.txt
│   │   │   └── pose.json
│   │   └── ...
│   ├── val/
│   └── test/
└── DGS/
    ├── train/
    ├── val/
    └── test/
```

## 🚀 模型训练

### 1. MLSF模式训练

```bash
python train_signllm.py \
    --config configs/signllm_mlsf_config.json \
    --output_dir ./outputs/signllm_mlsf
```

### 2. Prompt2LangGloss模式训练

```bash
python train_signllm.py \
    --config configs/signllm_prompt2langgloss_config.json \
    --output_dir ./outputs/signllm_prompt2langgloss
```

### 3. 自定义训练配置

修改配置文件中的参数：
```json
{
  "model": {
    "hidden_dim": 1024,
    "pose_dim": 150
  },
  "training": {
    "epochs": 100,
    "batch_size": 8,
    "lr": 1e-4
  }
}
```

### 4. 恢复训练

```bash
python train_signllm.py \
    --config configs/signllm_mlsf_config.json \
    --resume ./outputs/signllm_mlsf/latest_model.pt
```

## 🔮 推理使用

### 1. 单个文本生成

```bash
python inference_signllm.py \
    --model_path ./outputs/signllm_mlsf/best_model.pt \
    --text "Hello, how are you?" \
    --language ASL \
    --mode mlsf \
    --visualize
```

### 2. 批量生成

```bash
# 创建文本文件
echo -e "Hello world\nHow are you\nNice to meet you" > texts.txt

python inference_signllm.py \
    --model_path ./outputs/signllm_mlsf/best_model.pt \
    --texts_file texts.txt \
    --language ASL \
    --mode mlsf \
    --output_dir ./inference_results
```

### 3. 交互式演示

```bash
python inference_signllm.py \
    --model_path ./outputs/signllm_mlsf/best_model.pt \
    --interactive
```

### 4. Python API使用

```python
from inference_signllm import SignLLMInference

# 创建推理器
inference = SignLLMInference("./outputs/signllm_mlsf/best_model.pt")

# 生成手语姿态
result = inference.generate_single(
    text="Hello world",
    language="ASL",
    mode="mlsf",
    visualize=True,
    output_dir="./demo_output"
)

print(f"Generated {result['num_frames']} frames")
print(f"Quality score: {result['quality_scores'].mean():.3f}")
```

## 📈 评估

### 1. 运行评估

```python
from evaluation import SignLLMEvaluator

evaluator = SignLLMEvaluator()

# 评估姿态生成质量
metrics = evaluator.evaluate_poses(predictions, targets)
print(f"DTW Score: {metrics['dtw_score']:.4f}")
print(f"Pose Similarity: {metrics['pose_similarity']:.4f}")
```

### 2. 评估指标

- **DTW Score**: 动态时间规整分数
- **Pose Similarity**: 姿态相似度
- **Motion Smoothness**: 运动平滑度
- **BLEU Score**: 用于gloss评估
- **MSE/MAE**: 均方误差和平均绝对误差

## 📁 项目结构

```
signllm/
├── signllm_model.py          # SignLLM模型实现
├── data_processor.py         # 数据处理模块
├── train_signllm.py         # 训练脚本
├── inference_signllm.py     # 推理脚本
├── evaluation.py            # 评估模块
├── utils.py                 # 工具函数
├── requirements.txt         # 依赖包
├── configs/                 # 配置文件
│   ├── signllm_mlsf_config.json
│   └── signllm_prompt2langgloss_config.json
├── datasets/               # 数据集目录
├── outputs/               # 训练输出
├── Prompt2Sign/          # Prompt2Sign工具
└── README.md
```

## 🎯 复现结果

### 1. 预期性能指标

根据论文，在ASL数据集上的预期结果：

| 模式 | DTW Score | BLEU-4 | Pose Similarity |
|------|-----------|--------|-----------------|
| MLSF | 0.85+ | - | 0.80+ |
| Prompt2LangGloss | 0.83+ | 50.41 | 0.78+ |

### 2. 训练时间

- **MLSF模式**: 约24小时 (单GPU V100)
- **Prompt2LangGloss模式**: 约36小时 (单GPU V100)

### 3. 模型大小

- **参数量**: 约40M (MLSF) / 45M (Prompt2LangGloss)
- **模型文件**: 约160MB / 180MB

## 🔧 故障排除

### 1. 常见问题

**Q: 训练时显存不足**
```bash
# 减小batch_size
# 在配置文件中修改：
"batch_size": 4  # 从8改为4
```

**Q: 数据加载错误**
```bash
# 检查数据目录结构
# 确保每个样本目录包含text.txt和pose.json
```

**Q: 模型收敛慢**
```bash
# 调整学习率
"lr": 5e-5  # 从1e-4改为5e-5
```

### 2. 性能优化

- 使用混合精度训练：在配置中添加 `"use_amp": true`
- 增加数据并行：使用多GPU训练
- 优化数据加载：增加 `num_workers`

## 📚 参考资料

- [SignLLM论文](https://arxiv.org/abs/2405.10718)
- [Prompt2Sign数据集](https://signllm.github.io/Prompt2Sign/)
- [项目主页](https://signllm.github.io/)

## 📄 许可证

本项目遵循Creative Commons Attribution-NonCommercial 4.0 International License。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个实现！

## 📧 联系

如有问题，请通过以下方式联系：
- 提交GitHub Issue
- 邮箱：[项目维护者邮箱]

---

**注意**: 这是SignLLM论文的复现实现，用于学术研究目的。请确保遵循相关的数据使用协议和许可证要求。 