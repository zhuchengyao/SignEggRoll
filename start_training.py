#!/usr/bin/env python3
"""
SignLLM 训练启动脚本
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from train_signllm import SignLLMTrainer
from utils import setup_logging


def check_data_availability(config):
    """检查数据是否可用"""
    dataset_path = Path(config['data']['dataset_path'])
    
    print(f"🔍 检查数据路径: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"❌ 数据路径不存在: {dataset_path}")
        return False
    
    # 检查语言目录
    for language in config['data']['languages']:
        lang_dir = dataset_path / language
        if not lang_dir.exists():
            print(f"❌ 语言目录不存在: {lang_dir}")
            return False
        
        # 检查分割目录
        for split_name, split_value in config['data']['splits'].items():
            split_dir = lang_dir / split_value
            if not split_dir.exists():
                print(f"❌ 分割目录不存在: {split_dir}")
                return False
            
            # 检查样本数量
            samples = list(split_dir.iterdir())
            sample_count = len([s for s in samples if s.is_dir()])
            print(f"✅ {language}/{split_value}: {sample_count} 个样本")
    
    return True


def check_environment():
    """检查训练环境"""
    print("🔧 检查训练环境...")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name()}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
    
    # 检查内存
    import psutil
    memory = psutil.virtual_memory()
    print(f"💾 系统内存: {memory.total / 1e9:.1f} GB (可用: {memory.available / 1e9:.1f} GB)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="启动SignLLM训练")
    parser.add_argument("--config", type=str, default="configs/signllm_eggroll_config.json",
                       help="训练配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                       help="从检查点恢复训练")
    parser.add_argument("--debug", action="store_true",
                       help="调试模式（使用更少数据）")
    parser.add_argument("--dry_run", action="store_true",
                       help="干运行模式（只检查配置，不实际训练）")
    
    args = parser.parse_args()
    
    print("🚀 SignLLM 训练启动器")
    print("=" * 50)
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"📋 加载配置: {config_path}")
    
    # 调试模式调整
    if args.debug:
        print("🐛 调试模式启用")
        config['data']['batch_size'] = 2
        config['training']['num_epochs'] = 2
        config['training']['save_every'] = 1
        config['training']['eval_every'] = 1
        config['logging']['log_every'] = 10
    
    # 检查环境
    if not check_environment():
        return
    
    # 检查数据
    if not check_data_availability(config):
        print("\n💡 提示: 请先完成数据转换:")
        print("python final_convert_data.py --data_dir datasets/final_data --output_dir datasets/signllm_data_complete --splits dev --language ASL")
        return
    
    # 设置恢复训练
    if args.resume:
        config['checkpoint']['resume_from'] = args.resume
        print(f"🔄 从检查点恢复: {args.resume}")
    
    if args.dry_run:
        print("✅ 配置检查完成，干运行模式结束")
        return
    
    # 创建输出目录
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # 设置日志
    setup_logging(config['logging']['log_dir'])
    
    # 保存使用的配置
    used_config_path = Path(config['checkpoint']['save_dir']) / "config_used.json"
    with open(used_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 配置已保存到: {used_config_path}")
    
    # 开始训练
    print("\n🎯 开始训练...")
    print("=" * 50)
    
    try:
        trainer = SignLLMTrainer(config)
        trainer.train()
        
        print("\n🎉 训练完成！")
        print(f"📁 检查点保存在: {config['checkpoint']['save_dir']}")
        print(f"📊 日志保存在: {config['logging']['log_dir']}")
        
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 