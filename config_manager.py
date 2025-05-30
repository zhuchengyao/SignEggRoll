#!/usr/bin/env python3
"""
配置管理系统 - 支持YAML/JSON配置文件和命令行参数
"""

import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置 - 使用dataclass自动生成"""
    
    # 模型规模
    model_size: str = "small"
    
    # 网络架构
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    ff_multiplier: int = 2
    dropout: float = 0.1
    
    # 数据维度
    pose_dim: int = 150
    gloss_vocab_size: int = 2000
    max_sequence_length: int = 512
    
    # 训练参数
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    patience: int = 10
    
    # 数据设置
    min_frames: int = 30
    max_frames: int = 500
    default_max_frames: int = 256
    
    # 多语言设置
    num_priorities: int = 8
    num_languages: int = 8
    bert_model: str = "bert-base-multilingual-cased"
    
    # 损失函数权重
    loss_alpha: float = 1.0  # MSE权重
    loss_beta: float = 0.5   # 运动平滑度权重
    loss_gamma: float = 0.3  # 姿态一致性权重
    
    # 数据增强
    use_data_augmentation: bool = True
    augmentation_noise_std: float = 0.01
    augmentation_speed_factors: list = None
    
    # 设备和性能
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    
    def __post_init__(self):
        """后处理：根据model_size自动设置参数"""
        if self.augmentation_speed_factors is None:
            self.augmentation_speed_factors = [0.8, 1.2]
            
        # 根据模型大小自动调整参数
        size_configs = {
            "tiny": {
                "hidden_dim": 256,
                "num_layers": 2,
                "num_heads": 4,
                "ff_multiplier": 2,
                "gloss_vocab_size": 1000,
                "batch_size": 8
            },
            "small": {
                "hidden_dim": 512,
                "num_layers": 4,
                "num_heads": 8,
                "ff_multiplier": 2,
                "gloss_vocab_size": 2000,
                "batch_size": 4
            },
            "medium": {
                "hidden_dim": 768,
                "num_layers": 6,
                "num_heads": 12,
                "ff_multiplier": 3,
                "gloss_vocab_size": 5000,
                "batch_size": 2
            },
            "large": {
                "hidden_dim": 1024,
                "num_layers": 8,
                "num_heads": 16,
                "ff_multiplier": 4,
                "gloss_vocab_size": 10000,
                "batch_size": 1
            }
        }
        
        if self.model_size in size_configs:
            for key, value in size_configs[self.model_size].items():
                if not hasattr(self, '_manually_set') or key not in getattr(self, '_manually_set', set()):
                    setattr(self, key, value)
        
        # 计算衍生参数
        self.dim_feedforward = self.hidden_dim * self.ff_multiplier
    
    @property
    def device_auto(self) -> str:
        """自动检测设备"""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """估算内存使用"""
        # 简化的内存估算（MB）
        param_memory = (self.hidden_dim ** 2 * self.num_layers * 4) / (1024 ** 2)  # 参数内存
        batch_memory = (self.batch_size * self.max_sequence_length * self.hidden_dim * 4) / (1024 ** 2)  # 批次内存
        
        return {
            "parameter_memory_mb": param_memory,
            "batch_memory_mb": batch_memory,
            "total_estimated_mb": param_memory + batch_memory * 2  # 梯度占用额外内存
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建配置"""
        return cls(**config_dict)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config_paths = []
    
    def load_from_file(self, config_path: str) -> ModelConfig:
        """从文件加载配置"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 根据文件扩展名选择加载方式
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        logger.info(f"从 {config_path} 加载配置")
        return ModelConfig.from_dict(config_dict)
    
    def save_to_file(self, config: ModelConfig, config_path: str):
        """保存配置到文件"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.to_dict()
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        logger.info(f"配置已保存到 {config_path}")
    
    def create_argparser(self) -> argparse.ArgumentParser:
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(description="SignLLM 训练配置")
        
        # 基础参数
        parser.add_argument("--config", type=str, help="配置文件路径")
        parser.add_argument("--model-size", type=str, choices=["tiny", "small", "medium", "large"],
                          default="small", help="模型大小")
        
        # 训练参数
        parser.add_argument("--batch-size", type=int, help="批次大小")
        parser.add_argument("--learning-rate", type=float, help="学习率")
        parser.add_argument("--num-epochs", type=int, help="训练轮数")
        parser.add_argument("--patience", type=int, help="早停耐心值")
        
        # 数据参数
        parser.add_argument("--data-dir", type=str, help="数据目录")
        parser.add_argument("--max-frames", type=int, help="最大帧数")
        
        # 设备参数
        parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], 
                          default="auto", help="计算设备")
        parser.add_argument("--no-mixed-precision", action="store_true", 
                          help="禁用混合精度训练")
        
        # 输出参数
        parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
        parser.add_argument("--save-config", type=str, help="保存当前配置到文件")
        
        return parser
    
    def parse_args_and_config(self, args=None) -> ModelConfig:
        """解析命令行参数并合并配置"""
        parser = self.create_argparser()
        parsed_args = parser.parse_args(args)
        
        # 首先加载基础配置
        if parsed_args.config:
            config = self.load_from_file(parsed_args.config)
        else:
            config = ModelConfig()
        
        # 用命令行参数覆盖配置
        config._manually_set = set()
        
        for arg_name, arg_value in vars(parsed_args).items():
            if arg_value is not None and arg_name != 'config' and arg_name != 'save_config':
                # 转换参数名称格式
                config_name = arg_name.replace('-', '_')
                if config_name == 'no_mixed_precision':
                    config.mixed_precision = False
                    config._manually_set.add('mixed_precision')
                elif hasattr(config, config_name):
                    setattr(config, config_name, arg_value)
                    config._manually_set.add(config_name)
        
        # 重新初始化以应用模型大小相关的设置
        config.__post_init__()
        
        # 保存配置（如果需要）
        if parsed_args.save_config:
            self.save_to_file(config, parsed_args.save_config)
        
        return config
    
    def create_default_configs(self, output_dir: str = "configs"):
        """创建默认配置文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 为每种模型大小创建配置
        for size in ["tiny", "small", "medium", "large"]:
            config = ModelConfig(model_size=size)
            config_file = output_path / f"{size}_config.yaml"
            self.save_to_file(config, config_file)
        
        # 创建训练配置示例
        train_config = ModelConfig(
            model_size="small",
            num_epochs=100,
            batch_size=8,
            learning_rate=5e-5,
            use_data_augmentation=True
        )
        self.save_to_file(train_config, output_path / "train_config.yaml")
        
        logger.info(f"默认配置文件已创建在: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 创建默认配置文件
    config_manager.create_default_configs()
    
    # 从命令行解析配置
    config = config_manager.parse_args_and_config()
    
    print("🔧 配置信息:")
    print(f"  模型大小: {config.model_size}")
    print(f"  隐藏维度: {config.hidden_dim}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  设备: {config.device_auto}")
    print(f"  估算内存使用: {config.estimate_memory_usage()}") 