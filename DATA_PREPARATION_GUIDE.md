# SignLLM 数据准备指南

本指南将帮助您将已处理的手语数据转换为SignLLM框架可以使用的格式。

## 📋 数据概述

根据您提供的数据处理脚本，您的数据处理流程如下：

1. **pipeline_demo_01_json2h5.py**: 将OpenPose JSON文件打包为H5格式
2. **pipeline_demo_02_h5totxt.py**: 从H5转换为TXT，进行3D姿态处理
3. **pipeline_demo_03_txt2skels.py**: 生成最终的.skels和.text文件

最终得到的数据格式：
- `dev.text`: 文本描述（31,047行）
- `dev.skels`: 骨架数据（31,047行）
- `dev.files`: 文件列表

## 🔄 数据转换步骤

### 步骤1: 运行数据转换脚本

```bash
python convert_data_to_signllm.py \
    --data_dir datasets/final_data/final_data \
    --output_dir datasets/signllm_data \
    --splits dev train test \
    --language ASL
```

**参数说明：**
- `--data_dir`: 包含.text、.skels、.files文件的目录
- `--output_dir`: SignLLM格式数据的输出目录
- `--splits`: 要处理的数据集分割（dev, train, test）
- `--language`: 手语语言代码（ASL, DGS等）

### 步骤2: 验证数据转换

```bash
python test_data_conversion.py
```

这个脚本会：
- 验证数据格式是否正确
- 分析数据统计信息
- 测试模型兼容性

### 步骤3: 开始训练

```bash
# 使用您的数据配置
python train_signllm.py --config configs/signllm_your_data_config.json

# 或者先运行小规模测试
python demo_train.py
```

## 📊 数据格式说明

### 输入格式（您的数据）

**dev.skels格式：**
```
x1 y1 z1 x2 y2 z2 ... time1 x1 y1 z1 x2 y2 z2 ... time2 ...
```
- 每行代表一个视频序列
- 数据按帧组织，每帧包含150维姿态数据
- 时间戳用于分隔不同帧

**dev.text格式：**
```
And I call them decorative elements because basically all they're meant to do is to enrich and color the page.
So they don't really have much of a symbolic meaning other than maybe life is richer, life is beautiful...
```
- 每行对应一个文本描述
- 与.skels文件行数一一对应

### 输出格式（SignLLM格式）

转换后的目录结构：
```
datasets/signllm_data/
├── ASL/
│   ├── dev/
│   │   ├── sample_000001/
│   │   │   ├── text.txt
│   │   │   └── pose.json
│   │   ├── sample_000002/
│   │   └── ...
│   ├── train/
│   └── test/
├── ASL_dev_index.json
├── ASL_train_index.json
├── ASL_test_index.json
└── dataset_config.json
```

**pose.json格式：**
```json
{
  "poses": [
    {
      "pose_keypoints_2d": [x1, y1, c1, x2, y2, c2, ...],
      "hand_left_keypoints_2d": [x1, y1, c1, ...],
      "hand_right_keypoints_2d": [x1, y1, c1, ...],
      "face_keypoints_2d": [0.0, 0.0, 0.0, ...]
    },
    ...
  ],
  "num_frames": 50,
  "original_index": 0
}
```

## ⚙️ 配置调整

### 训练配置优化

根据您的数据特点，建议调整以下参数：

```json
{
  "data": {
    "batch_size": 4,           // 根据GPU内存调整
    "max_sequence_length": 256 // 根据数据统计调整
  },
  "training": {
    "epochs": 50,              // 可以从较少epoch开始
    "lr": 5e-5                 // 较小的学习率
  }
}
```

### 数据维度说明

您的数据处理流程生成150维姿态数据，分配如下：
- **上身关键点**: 24维 (8个点 × 3坐标)
- **左手关键点**: 63维 (21个点 × 3坐标)  
- **右手关键点**: 63维 (21个点 × 3坐标)
- **总计**: 150维

## 🔧 故障排除

### 常见问题

**Q1: 数据转换失败**
```bash
# 检查文件是否存在
ls -la datasets/final_data/final_data/

# 检查文件行数是否匹配
wc -l datasets/final_data/final_data/dev.*
```

**Q2: 内存不足**
```bash
# 减小batch_size
# 在配置文件中修改：
"batch_size": 2
```

**Q3: 序列长度过长**
```bash
# 运行数据统计分析
python test_data_conversion.py

# 根据95%分位数调整max_sequence_length
```

### 数据质量检查

```python
# 检查姿态数据范围
import numpy as np

# 加载一个pose.json文件
with open('datasets/signllm_data/ASL/dev/sample_000001/pose.json', 'r') as f:
    data = json.load(f)

poses = data['poses']
print(f"帧数: {len(poses)}")
print(f"第一帧关键点数: {len(poses[0]['pose_keypoints_2d'])}")
```

## 📈 性能优化建议

### 1. 数据预处理优化
- 确保姿态数据已正确标准化
- 移除异常值和噪声帧
- 考虑数据增强技术

### 2. 训练优化
- 使用梯度累积处理大批次
- 实施早停机制
- 监控训练指标

### 3. 模型调整
- 根据数据量调整模型大小
- 考虑使用预训练权重
- 调整学习率调度策略

## 📝 下一步

1. **数据转换**: 运行转换脚本
2. **验证测试**: 确保数据格式正确
3. **小规模训练**: 使用demo_train.py测试
4. **完整训练**: 使用完整配置训练
5. **评估推理**: 测试模型性能

## 🆘 获取帮助

如果遇到问题：
1. 检查日志输出中的错误信息
2. 验证数据文件完整性
3. 确认环境依赖已正确安装
4. 参考quick_start.py进行环境测试

---

**注意**: 这个转换过程会将您的31,047个样本转换为SignLLM可以直接使用的格式，转换完成后即可开始训练。 