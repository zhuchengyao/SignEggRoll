#!/usr/bin/env python3
"""
帧数调整演示脚本
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from signllm_model import SignLLM, ModelConfig, CONFIG
import torch

def demo_frame_adjustment():
    """演示如何调整生成帧数"""
    print("🎬 帧数调整演示")
    print("=" * 50)
    
    # 使用tiny模型进行演示
    global CONFIG
    CONFIG.__init__("tiny")
    
    print(f"📊 当前配置:")
    CONFIG.print_config()
    print()
    
    # 创建模型
    model = SignLLM(languages=["ASL"])
    model.eval()
    
    test_text = ["Hello world, how are you today?"]
    
    print("🔄 测试不同帧数设置:")
    print("-" * 30)
    
    frame_settings = [
        (None, "默认"),           # 使用默认256帧
        (50, "短序列"),           # 短序列
        (128, "中等长度"),        # 中等长度
        (400, "长序列"),          # 长序列 (会被限制到max_frames)
        (600, "超长序列"),        # 超长序列 (会被限制)
    ]
    
    with torch.no_grad():
        for max_frames, description in frame_settings:
            try:
                if max_frames is None:
                    # 使用默认帧数
                    poses, quality = model(
                        texts=test_text,
                        language="ASL",
                        mode="mlsf"
                    )
                    actual_frames = poses.shape[1]
                    print(f"✅ {description}: 生成 {actual_frames} 帧 (默认)")
                else:
                    # 指定帧数
                    poses, quality = model(
                        texts=test_text,
                        language="ASL",
                        mode="mlsf",
                        max_length=max_frames
                    )
                    actual_frames = poses.shape[1]
                    print(f"✅ {description}: 请求 {max_frames} 帧 → 实际生成 {actual_frames} 帧")
                    
            except Exception as e:
                print(f"❌ {description}: 失败 - {e}")
    
    print("\n📝 调整帧数的方法:")
    print("1. 修改默认帧数:")
    print("   CONFIG.default_max_frames = 128  # 改为128帧")
    print()
    print("2. 在推理时指定:")
    print("   model(texts=texts, language='ASL', max_length=100)")
    print()
    print("3. 修改配置范围:")
    print("   CONFIG.min_frames = 20")
    print("   CONFIG.max_frames = 300")

def show_current_settings():
    """显示当前帧数设置"""
    print("\n🔍 当前帧数设置:")
    print(f"  默认帧数: {CONFIG.default_max_frames}")
    print(f"  最小帧数: {CONFIG.min_frames}")
    print(f"  最大帧数: {CONFIG.max_frames}")

if __name__ == "__main__":
    demo_frame_adjustment()
    show_current_settings() 