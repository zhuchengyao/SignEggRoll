# SignLLM 训练完整指南

## 🎯 训练流程概览

完成数据转换后，您可以按照以下步骤开始训练SignLLM模型：

## 📋 前置条件检查

### 1. 确认数据转换完成
```bash
# 检查数据是否转换完成
ls -la datasets/signllm_data_complete/ASL/dev/
```

### 2. 检查环境依赖
```bash
# 确保所有依赖已安装
pip install -r requirements.txt
```

### 3. 检查GPU状态（推荐）
```bash
# 检查CUDA是否可用
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

## 🚀 训练方式

### 方式1: 快速验证训练（推荐新手）

适用于：验证训练流程、调试代码、快速测试

```bash
# 运行快速训练（3个epoch，小批次）
python quick_train.py
```

**特点：**
- 训练时间短（约10-30分钟）
- 使用小模型配置
- 自动检查数据和环境
- 包含推理测试

### 方式2: 完整训练

适用于：正式训练、获得最佳性能

```bash
# 使用默认配置开始训练
python start_training.py

# 或指定配置文件
python start_training.py --config configs/signllm_your_data_config.json

# 调试模式（减少数据量和epoch）
python start_training.py --debug

# 干运行模式（只检查配置，不实际训练）
python start_training.py --dry_run
```

### 方式3: 直接使用训练器

适用于：高级用户、自定义训练流程

```bash
# 直接使用SignLLM训练器
python train_signllm.py --config configs/signllm_your_data_config.json
```

## ⚙️ 配置文件说明

### 主要配置文件

1. **`configs/signllm_your_data_config.json`** - 您的数据专用配置
2. **`configs/signllm_mlsf_config.json`** - MLSF模式配置
3. **`configs/signllm_prompt2langgloss_config.json`** - Prompt2LangGloss模式配置

### 关键配置参数

```json
{
  "model": {
    "mode": "mlsf",  // 或 "prompt2langgloss"
    "text_encoder": {
      "model_name": "bert-base-multilingual-cased",
      "max_length": 128
    },
    "pose_decoder": {
      "output_dim": 150,  // 匹配您的数据维度
      "num_layers": 4
    }
  },
  "data": {
    "dataset_path": "datasets/signllm_data_complete",
    "batch_size": 8,  // 根据GPU内存调整
    "max_frames": 500
  },
  "training": {
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "save_every": 5
  }
}
```

## 📊 训练监控

### 1. TensorBoard监控
```bash
# 启动TensorBoard
tensorboard --logdir logs/signllm_your_data

# 在浏览器中访问
# http://localhost:6006
```

### 2. 日志文件
- **训练日志**: `logs/signllm_your_data/train.log`
- **检查点**: `checkpoints/signllm_your_data/`

### 3. 关键指标
- **损失函数**: 总损失、重构损失、RL损失
- **评估指标**: DTW分数、姿态相似度、运动平滑度
- **学习率**: 当前学习率变化

## 🔧 常见问题解决

### 1. 内存不足
```bash
# 减少批次大小
# 在配置文件中设置：
"batch_size": 2  # 或更小
```

### 2. 训练速度慢
```bash
# 检查是否使用GPU
python -c "import torch; print(torch.cuda.is_available())"

# 减少数据加载器进程数
"num_workers": 0  # 在配置文件中
```

### 3. 数据加载错误
```bash
# 检查数据路径
python start_training.py --dry_run

# 重新转换数据
python final_convert_data.py --data_dir datasets/final_data --output_dir datasets/signllm_data_complete --splits dev --language ASL --max_samples 100
```

### 4. 模型收敛问题
```bash
# 调整学习率
"learning_rate": 1e-3  # 增大学习率

# 启用梯度裁剪
"gradient_clip": 1.0

# 增加warmup步数
"warmup_steps": 1000
```

## 📈 训练阶段

### 阶段1: 快速验证（1-2小时）
```bash
python quick_train.py
```
- 验证数据加载正常
- 验证模型前向传播
- 验证训练循环
- 验证保存和加载

### 阶段2: 小规模训练（4-8小时）
```bash
python start_training.py --debug
```
- 使用部分数据
- 训练10-20个epoch
- 观察损失下降趋势
- 调整超参数

### 阶段3: 完整训练（1-3天）
```bash
python start_training.py
```
- 使用全部数据
- 训练50-100个epoch
- 定期评估和保存
- 监控过拟合

## 🎯 训练目标

### 短期目标（快速验证）
- ✅ 训练流程无错误
- ✅ 损失函数下降
- ✅ 模型能生成姿态序列

### 中期目标（小规模训练）
- ✅ 训练损失稳定下降
- ✅ 验证损失不发散
- ✅ DTW分数逐步改善

### 长期目标（完整训练）
- ✅ 达到论文中的性能指标
- ✅ 生成高质量手语动作
- ✅ 支持多语言手语生成

## 📁 输出文件结构

```
checkpoints/signllm_your_data/
├── best_model.pth          # 最佳模型
├── last_model.pth          # 最新模型
├── config_used.json        # 使用的配置
├── epoch_5.pth            # 定期保存的检查点
└── training_stats.json     # 训练统计

logs/signllm_your_data/
├── train.log              # 训练日志
├── events.out.tfevents.*  # TensorBoard事件
└── samples/               # 生成样本
```

## 🔄 恢复训练

```bash
# 从最新检查点恢复
python start_training.py --resume checkpoints/signllm_your_data/last_model.pth

# 从最佳检查点恢复
python start_training.py --resume checkpoints/signllm_your_data/best_model.pth
```

## 🎉 训练完成后

### 1. 模型评估
```bash
python evaluation.py --model_path checkpoints/signllm_your_data/best_model.pth
```

### 2. 推理测试
```bash
python inference_signllm.py --model_path checkpoints/signllm_your_data/best_model.pth --text "Hello world"
```

### 3. 可视化结果
```bash
python visualize_skeleton_data.py --mode animation
```

---

## 💡 训练建议

1. **从小开始**: 先用`quick_train.py`验证流程
2. **监控指标**: 密切关注损失函数和评估指标
3. **定期保存**: 确保训练进度不丢失
4. **调整参数**: 根据训练效果调整学习率和批次大小
5. **耐心等待**: 完整训练需要较长时间，请保持耐心

**祝您训练顺利！** 🚀 