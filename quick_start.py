#!/usr/bin/env python3
"""
SignLLM快速启动脚本
用于测试模型的基本功能和验证环境配置
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signllm_model import SignLLM, RLLoss
from data_processor import PoseExtractor, PoseNormalizer
from evaluation import SignLLMEvaluator
from utils import set_seed, get_device_info, ConfigManager, PoseVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_environment():
    """测试环境配置"""
    logger.info("=== 环境测试 ===")
    
    # 检查PyTorch
    logger.info(f"PyTorch版本: {torch.__version__}")
    
    # 检查设备
    device_info = get_device_info()
    logger.info(f"CUDA可用: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        logger.info(f"GPU设备: {device_info['device_name']}")
        logger.info(f"GPU数量: {device_info['cuda_device_count']}")
    
    # 检查依赖包
    try:
        import transformers
        logger.info(f"Transformers版本: {transformers.__version__}")
    except ImportError:
        logger.warning("Transformers未安装")
    
    try:
        import mediapipe
        logger.info(f"MediaPipe版本: {mediapipe.__version__}")
    except ImportError:
        logger.warning("MediaPipe未安装")
    
    logger.info("环境测试完成\n")
    return True


def test_model_creation():
    """测试模型创建"""
    logger.info("=== 模型创建测试 ===")
    
    try:
        # 创建模型
        model = SignLLM(
            languages=["ASL", "DGS"],
            gloss_vocab_size=1000,  # 减小词汇表大小用于测试
            hidden_dim=256,         # 减小隐藏维度用于测试
            pose_dim=150
        )
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"模型创建成功")
        logger.info(f"总参数数量: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        return None


def test_model_forward():
    """测试模型前向传播"""
    logger.info("=== 模型前向传播测试 ===")
    
    model = test_model_creation()
    if model is None:
        return False
    
    try:
        # 设置为评估模式
        model.eval()
        
        # 测试MLSF模式
        logger.info("测试MLSF模式...")
        texts = ["Hello world", "How are you"]
        
        with torch.no_grad():
            poses, quality_scores = model(texts, "ASL", mode="mlsf", max_length=50)
        
        logger.info(f"MLSF输出形状: {poses.shape}")
        logger.info(f"质量分数形状: {quality_scores.shape}")
        
        # 测试Prompt2LangGloss模式
        logger.info("测试Prompt2LangGloss模式...")
        
        with torch.no_grad():
            poses, gloss_logits, quality_scores = model(texts, "ASL", mode="prompt2langgloss", max_pose_length=50)
        
        logger.info(f"Prompt2LangGloss姿态输出形状: {poses.shape}")
        logger.info(f"Gloss输出形状: {gloss_logits.shape}")
        logger.info(f"质量分数形状: {quality_scores.shape}")
        
        logger.info("模型前向传播测试成功\n")
        return True
        
    except Exception as e:
        logger.error(f"模型前向传播测试失败: {e}")
        return False


def test_loss_function():
    """测试损失函数"""
    logger.info("=== 损失函数测试 ===")
    
    try:
        # 创建RL损失函数
        criterion = RLLoss(alpha=0.1, beta=0.1)
        
        # 创建测试数据
        batch_size, seq_len, pose_dim = 2, 50, 150
        pred_poses = torch.randn(batch_size, seq_len, pose_dim)
        target_poses = torch.randn(batch_size, seq_len, pose_dim)
        quality_scores = torch.rand(batch_size, seq_len)
        
        # 计算损失
        loss = criterion(pred_poses, target_poses, quality_scores)
        
        logger.info(f"RL损失计算成功: {loss.item():.4f}")
        logger.info("损失函数测试成功\n")
        return True
        
    except Exception as e:
        logger.error(f"损失函数测试失败: {e}")
        return False


def test_data_processing():
    """测试数据处理"""
    logger.info("=== 数据处理测试 ===")
    
    try:
        # 测试姿态提取器
        extractor = PoseExtractor(method="mediapipe")
        logger.info("姿态提取器创建成功")
        
        # 测试姿态标准化
        test_poses = [
            {
                "pose_keypoints_2d": [100, 200, 0.9] * 8,
                "hand_left_keypoints_2d": [150, 250, 0.8] * 21,
                "hand_right_keypoints_2d": [200, 300, 0.7] * 21,
                "face_keypoints_2d": [175, 225, 0.6] * 70
            }
        ]
        
        normalized_poses = PoseNormalizer.normalize_pose_sequence(test_poses)
        logger.info(f"姿态标准化成功，处理了{len(normalized_poses)}帧")
        
        logger.info("数据处理测试成功\n")
        return True
        
    except Exception as e:
        logger.error(f"数据处理测试失败: {e}")
        return False


def test_evaluation():
    """测试评估模块"""
    logger.info("=== 评估模块测试 ===")
    
    try:
        evaluator = SignLLMEvaluator()
        
        # 创建测试数据
        predictions = [np.random.randn(50, 150) for _ in range(5)]
        targets = [np.random.randn(50, 150) for _ in range(5)]
        
        # 评估姿态
        metrics = evaluator.evaluate_poses(predictions, targets)
        
        logger.info("姿态评估指标:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # 测试gloss评估
        pred_gloss = [["hello", "world"], ["how", "are", "you"]]
        target_gloss = [["hello", "world"], ["how", "are", "you", "today"]]
        
        gloss_metrics = evaluator.evaluate_gloss_generation(pred_gloss, target_gloss)
        
        logger.info("Gloss评估指标:")
        for key, value in gloss_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info("评估模块测试成功\n")
        return True
        
    except Exception as e:
        logger.error(f"评估模块测试失败: {e}")
        return False


def test_visualization():
    """测试可视化功能"""
    logger.info("=== 可视化测试 ===")
    
    try:
        visualizer = PoseVisualizer(pose_dim=150)
        
        # 创建测试姿态数据
        test_poses = np.random.randn(30, 150)
        
        # 测试姿态序列可视化
        output_dir = Path("./test_output")
        output_dir.mkdir(exist_ok=True)
        
        visualizer.visualize_pose_sequence(
            test_poses, 
            str(output_dir / "test_poses.png"),
            title="Test Pose Sequence"
        )
        
        logger.info(f"姿态可视化保存到: {output_dir / 'test_poses.png'}")
        
        # 测试视频创建
        visualizer.create_pose_video(
            test_poses,
            str(output_dir / "test_poses.mp4")
        )
        
        logger.info(f"姿态视频保存到: {output_dir / 'test_poses.mp4'}")
        logger.info("可视化测试成功\n")
        return True
        
    except Exception as e:
        logger.error(f"可视化测试失败: {e}")
        return False


def test_config_management():
    """测试配置管理"""
    logger.info("=== 配置管理测试 ===")
    
    try:
        config_manager = ConfigManager()
        
        # 创建默认配置
        default_config = config_manager.create_default_config()
        logger.info(f"默认配置创建成功，包含{len(default_config)}个部分")
        
        # 验证配置
        is_valid = config_manager.validate_config(default_config)
        logger.info(f"配置验证结果: {'通过' if is_valid else '失败'}")
        
        logger.info("配置管理测试成功\n")
        return True
        
    except Exception as e:
        logger.error(f"配置管理测试失败: {e}")
        return False


def create_demo_data():
    """创建演示数据"""
    logger.info("=== 创建演示数据 ===")
    
    try:
        # 创建演示目录
        demo_dir = Path("./demo_data")
        demo_dir.mkdir(exist_ok=True)
        
        # 创建ASL演示数据
        asl_dir = demo_dir / "ASL" / "train"
        asl_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建几个演示样本
        demo_texts = [
            "Hello world",
            "How are you",
            "Nice to meet you",
            "Thank you very much",
            "Have a good day"
        ]
        
        for i, text in enumerate(demo_texts):
            sample_dir = asl_dir / f"sample_{i:06d}"
            sample_dir.mkdir(exist_ok=True)
            
            # 保存文本
            with open(sample_dir / "text.txt", 'w', encoding='utf-8') as f:
                f.write(text)
            
            # 创建随机姿态数据
            num_frames = np.random.randint(20, 80)
            poses = []
            
            for frame in range(num_frames):
                pose = {
                    "pose_keypoints_2d": np.random.randn(24).tolist(),
                    "hand_left_keypoints_2d": np.random.randn(63).tolist(),
                    "hand_right_keypoints_2d": np.random.randn(63).tolist(),
                    "face_keypoints_2d": np.random.randn(210).tolist()
                }
                poses.append(pose)
            
            pose_data = {
                "poses": poses,
                "num_frames": num_frames
            }
            
            import json
            with open(sample_dir / "pose.json", 'w') as f:
                json.dump(pose_data, f, indent=2)
        
        logger.info(f"演示数据创建成功，包含{len(demo_texts)}个样本")
        logger.info(f"数据保存在: {demo_dir}")
        logger.info("演示数据创建完成\n")
        return True
        
    except Exception as e:
        logger.error(f"演示数据创建失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("🚀 SignLLM快速启动测试")
    logger.info("=" * 50)
    
    # 设置随机种子
    set_seed(42)
    
    # 运行所有测试
    tests = [
        ("环境配置", test_environment),
        ("模型创建", test_model_creation),
        ("模型前向传播", test_model_forward),
        ("损失函数", test_loss_function),
        ("数据处理", test_data_processing),
        ("评估模块", test_evaluation),
        ("可视化功能", test_visualization),
        ("配置管理", test_config_management),
        ("演示数据创建", create_demo_data),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"🧪 运行测试: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"❌ 失败: {test_name} - {e}")
        
        logger.info("-" * 30)
    
    # 总结结果
    logger.info("📊 测试结果总结:")
    passed = sum(1 for result in results.values() if result is True)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        logger.info(f"  {status} {test_name}")
    
    logger.info(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！SignLLM环境配置正确。")
        logger.info("\n📝 下一步:")
        logger.info("1. 准备训练数据 (参考README.md)")
        logger.info("2. 运行训练: python train_signllm.py --config configs/signllm_mlsf_config.json")
        logger.info("3. 运行推理: python inference_signllm.py --model_path <model_path> --interactive")
    else:
        logger.warning("⚠️  部分测试失败，请检查环境配置。")
        logger.info("💡 建议:")
        logger.info("1. 检查依赖包安装: pip install -r requirements.txt")
        logger.info("2. 检查CUDA环境 (如果使用GPU)")
        logger.info("3. 查看错误日志并修复问题")


if __name__ == "__main__":
    main() 