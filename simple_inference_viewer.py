#!/usr/bin/env python3
"""
简单的SignLLM推理可视化器 - 加载模型，输入文本，生成并可视化手语姿态
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from pathlib import Path
import sys
import argparse

# 设置交互式后端
matplotlib.use('TkAgg')
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from signllm_model import SignLLM, ModelConfig, CONFIG

# 真实的50关节点骨架连接
REAL_SKELETON_STRUCTURE = [
    # head
    (0, 1, 0),
    # left shoulder
    (1, 2, 1),
    # left arm
    (2, 3, 2), (3, 4, 3),
    # right shoulder
    (1, 5, 1),
    # right arm
    (5, 6, 2), (6, 7, 3),
    # left hand - wrist
    (7, 8, 4),
    # left hand - palm
    (8, 9, 5), (8, 13, 9), (8, 17, 13), (8, 21, 17), (8, 25, 21),
    # left hand - fingers
    (9, 10, 6), (10, 11, 7), (11, 12, 8),
    (13, 14, 10), (14, 15, 11), (15, 16, 12),
    (17, 18, 14), (18, 19, 15), (19, 20, 16),
    (21, 22, 18), (22, 23, 19), (23, 24, 20),
    (25, 26, 22), (26, 27, 23), (27, 28, 24),
    # right hand - wrist
    (4, 29, 4),
    # right hand - palm
    (29, 30, 5), (29, 34, 9), (29, 38, 13), (29, 42, 17), (29, 46, 21),
    # right hand - fingers
    (30, 31, 6), (31, 32, 7), (32, 33, 8),
    (34, 35, 10), (35, 36, 11), (36, 37, 12),
    (38, 39, 14), (39, 40, 15), (40, 41, 16),
    (42, 43, 18), (43, 44, 19), (44, 45, 20),
    (46, 47, 22), (47, 48, 23), (48, 49, 24),
]

REAL_CONNECTIONS = [(start, end) for start, end, _ in REAL_SKELETON_STRUCTURE]


class SimpleInferenceViewer:
    """简单的SignLLM推理可视化器"""
    
    def __init__(self, model_path: str = None, model_size: str = "tiny"):
        self.model = self.load_model(model_path, model_size)
        self.results = []
        self.current_result_idx = 0
        self.current_frame = 0
        
    def load_model(self, model_path: str, model_size: str):
        """加载训练好的SignLLM模型"""
        print("🚀 简单SignLLM推理可视化器")
        print("=" * 40)
        
        # 声明全局变量
        global CONFIG
        
        # 首先检查checkpoint中的配置
        if model_path and Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # 尝试从checkpoint获取配置信息
                if 'config' in checkpoint:
                    # 使用保存的配置
                    saved_config = checkpoint['config']
                    print(f"📦 使用checkpoint中的配置...")
                    CONFIG = saved_config
                elif 'model_size' in checkpoint:
                    # 使用保存的模型大小
                    saved_size = checkpoint['model_size']
                    print(f"📦 检测到保存的模型大小: {saved_size}，覆盖命令行参数")
                    CONFIG = ModelConfig(saved_size)
                else:
                    # 根据模型参数推断配置
                    state_dict = checkpoint['model_state_dict']
                    
                    # 检查隐藏维度
                    if 'mlsf_mode.text_encoders.ASL.projection.weight' in state_dict:
                        hidden_dim = state_dict['mlsf_mode.text_encoders.ASL.projection.weight'].shape[0]
                        
                        if hidden_dim == 256:
                            inferred_size = "tiny"
                        elif hidden_dim == 384:
                            inferred_size = "small"  
                        elif hidden_dim == 512:
                            inferred_size = "medium"
                        elif hidden_dim == 768:
                            inferred_size = "large"
                        else:
                            inferred_size = model_size  # 使用默认值
                        
                        print(f"📦 根据参数推断模型大小: {inferred_size} (hidden_dim={hidden_dim})")
                        CONFIG = ModelConfig(inferred_size)
                    else:
                        # 使用命令行参数
                        print(f"📦 使用命令行指定的模型大小: {model_size}")
                        CONFIG = ModelConfig(model_size)
                
            except Exception as e:
                print(f"⚠️  读取checkpoint配置失败: {e}")
                CONFIG = ModelConfig(model_size)
        else:
            # 使用命令行参数
            print(f"📦 使用命令行指定的模型大小: {model_size}")
            CONFIG = ModelConfig(model_size)
        
        print(f"📦 加载模型 ({CONFIG.model_size})...")
        model = SignLLM(languages=["ASL"])
        
        if model_path and Path(model_path).exists():
            try:
                # 先运行一次前向传播来创建动态层
                dummy_text = ["hello"]
                with torch.no_grad():
                    model(dummy_text, "ASL", max_length=16)
                
                checkpoint = torch.load(model_path, map_location='cpu')
                state_dict = checkpoint['model_state_dict']
                model.load_state_dict(state_dict)
                
                epoch = checkpoint.get('epoch', 'Unknown')
                loss = checkpoint.get('loss', 'Unknown')
                print(f"✅ 成功加载训练模型: Epoch {epoch}, Loss {loss}")
                
            except Exception as e:
                print(f"⚠️  加载模型失败，使用随机初始化: {e}")
        else:
            print(f"⚠️  未找到模型文件，使用随机初始化模型")
        
        model.eval()
        return model
    
    def inference_texts(self, texts: list, language: str = "ASL", mode: str = "mlsf", max_length: int = 64):
        """对输入文本进行推理"""
        print(f"\n🎯 开始推理...")
        print(f"   语言: {language}")
        print(f"   模式: {mode.upper()}")
        print(f"   最大长度: {max_length}")
        
        results = []
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                print(f"   正在处理: '{text}'")
                
                try:
                    # 模型推理
                    pred_poses, quality_scores = self.model(
                        texts=[text],
                        language=language,
                        mode=mode,
                        max_length=max_length
                    )
                    
                    # 转换为numpy
                    pose_data = pred_poses[0].detach().cpu().numpy()  # [seq_len, 150]
                    pose_3d = pose_data.reshape(-1, 50, 3)  # [seq_len, 50, 3]
                    
                    # 计算质量分数
                    avg_quality = quality_scores.mean().item()
                    
                    results.append({
                        'text': text,
                        'poses': pose_3d,
                        'quality': avg_quality,
                        'mode': mode,
                        'frames': pose_3d.shape[0]
                    })
                    
                    print(f"     ✅ 成功: {pose_3d.shape[0]} 帧, 质量: {avg_quality:.4f}")
                    
                except Exception as e:
                    print(f"     ❌ 失败: {e}")
                    continue
        
        self.results = results
        print(f"\n📊 推理完成，成功生成 {len(results)} 个结果")
        return results
    
    def visualize_results(self):
        """可视化推理结果"""
        if len(self.results) == 0:
            print("❌ 没有推理结果可以显示")
            return
        
        print(f"\n🎮 启动交互式可视化器...")
        
        # 创建图形窗口
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle("SignLLM推理结果可视化器\n使用键盘控制：← → 切换帧，↑ ↓ 切换文本", fontsize=14)
        
        ax = fig.add_subplot(111, projection='3d')
        
        # 显示第一个结果
        self.current_result_idx = 0
        self.current_frame = 0
        self.update_display(ax)
        
        # 绑定键盘事件
        def on_key_press(event):
            current_result = self.results[self.current_result_idx]
            max_frames = current_result['poses'].shape[0]
            
            if event.key == 'left':  # 上一帧
                self.current_frame = (self.current_frame - 1) % max_frames
                self.update_display(ax)
            elif event.key == 'right':  # 下一帧
                self.current_frame = (self.current_frame + 1) % max_frames
                self.update_display(ax)
            elif event.key == 'up':  # 上一个文本
                self.current_result_idx = (self.current_result_idx - 1) % len(self.results)
                self.current_frame = 0
                self.update_display(ax)
            elif event.key == 'down':  # 下一个文本
                self.current_result_idx = (self.current_result_idx + 1) % len(self.results)
                self.current_frame = 0
                self.update_display(ax)
            elif event.key == 'q':  # 退出
                plt.close(fig)
        
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # 添加控制说明
        instruction_text = """
控制说明:
← → 切换帧
↑ ↓ 切换文本
Q 退出
鼠标拖拽旋转, 滚轮缩放
        """
        fig.text(0.02, 0.02, instruction_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        print("🎮 控制说明:")
        print("  ← → : 切换帧")
        print("  ↑ ↓ : 切换文本")
        print("  Q   : 退出")
        print("  鼠标拖拽: 旋转视角")
        print("  鼠标滚轮: 缩放")
        
        plt.show()
    
    def update_display(self, ax):
        """更新显示内容"""
        ax.clear()
        
        # 获取当前结果
        current_result = self.results[self.current_result_idx]
        poses = current_result['poses']
        text = current_result['text']
        quality = current_result['quality']
        mode = current_result['mode']
        
        # 获取当前帧的关节数据
        joints = poses[self.current_frame]  # [50, 3]
        
        x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
        
        # 绘制关节点 - 不同部位用不同颜色
        # 上身 (0-7)
        ax.scatter(x[:8], y[:8], z[:8], c='red', s=80, alpha=0.9, 
                  label='上身', edgecolors='darkred', linewidth=1)
        
        # 左手 (8-28)
        ax.scatter(x[8:29], y[8:29], z[8:29], c='blue', s=50, alpha=0.8, 
                  label='左手', edgecolors='darkblue', linewidth=0.5)
        
        # 右手 (29-49)
        ax.scatter(x[29:50], y[29:50], z[29:50], c='green', s=50, alpha=0.8, 
                  label='右手', edgecolors='darkgreen', linewidth=0.5)
        
        # 绘制骨架连接
        for start, end in REAL_CONNECTIONS:
            if start < len(joints) and end < len(joints):
                if not (np.allclose(joints[start], 0) or np.allclose(joints[end], 0)):
                    # 根据连接类型使用不同颜色
                    if start < 8 and end < 8:  # 上身连接
                        color = 'red'
                        linewidth = 3
                    elif 8 <= start < 29 and 8 <= end < 29:  # 左手连接
                        color = 'blue'
                        linewidth = 1.5
                    elif 29 <= start < 50 and 29 <= end < 50:  # 右手连接
                        color = 'green'
                        linewidth = 1.5
                    else:  # 跨部位连接
                        color = 'black'
                        linewidth = 3
                    
                    ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 
                           color=color, alpha=0.7, linewidth=linewidth)
        
        # 设置坐标轴
        ax.set_xlabel('X轴', fontsize=12)
        ax.set_ylabel('Y轴', fontsize=12)
        ax.set_zlabel('Z轴', fontsize=12)
        
        # 设置标题
        title = f"'{text}'\n帧 {self.current_frame+1}/{poses.shape[0]} | 文本 {self.current_result_idx+1}/{len(self.results)} | 质量: {quality:.4f} | 模式: {mode.upper()}"
        ax.set_title(title, fontsize=12, pad=20)
        
        # 设置相等的坐标轴比例
        ranges = [x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]
        max_range = max(ranges) / 2.0 if max(ranges) > 0 else 0.1
        center = [x.mean(), y.mean(), z.mean()]
        
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        # 添加图例
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        # 刷新显示
        plt.draw()
    
    def print_summary(self):
        """打印推理结果摘要"""
        if len(self.results) == 0:
            print("❌ 没有推理结果")
            return
        
        print(f"\n📊 推理结果摘要:")
        print("=" * 60)
        
        for i, result in enumerate(self.results):
            text = result['text']
            frames = result['frames']
            quality = result['quality']
            mode = result['mode']
            
            print(f"{i+1:2d}. '{text[:40]}{'...' if len(text) > 40 else ''}'")
            print(f"     帧数: {frames:3d} | 质量: {quality:.4f} | 模式: {mode.upper()}")
        
        print("=" * 60)
        
        # 统计信息
        total_frames = sum(r['frames'] for r in self.results)
        avg_quality = sum(r['quality'] for r in self.results) / len(self.results)
        avg_frames = total_frames / len(self.results)
        
        print(f"📈 统计:")
        print(f"   总文本数: {len(self.results)}")
        print(f"   总帧数: {total_frames}")
        print(f"   平均帧数: {avg_frames:.1f}")
        print(f"   平均质量: {avg_quality:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简单的SignLLM推理可视化器")
    parser.add_argument("--model_path", type=str, default="checkpoints/eggroll_train/epoch_10.pth",
                       help="训练模型路径")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small", "medium", "large"],
                       help="模型大小")
    parser.add_argument("--texts", nargs="+", 
                       default=[
                           "Hello, how are you?",
                           "Nice to meet you",
                           "Thank you very much",
                           "Good morning",
                           "Have a great day"
                       ],
                       help="要推理的文本列表")
    parser.add_argument("--language", type=str, default="ASL", help="目标语言")
    parser.add_argument("--mode", type=str, default="mlsf", choices=["mlsf", "prompt2langgloss"],
                       help="推理模式")
    parser.add_argument("--max_length", type=int, default=64, help="最大生成长度")
    
    args = parser.parse_args()
    
    try:
        # 创建推理器
        viewer = SimpleInferenceViewer(args.model_path, args.model_size)
        
        # 显示输入文本
        print(f"\n📝 输入文本:")
        for i, text in enumerate(args.texts, 1):
            print(f"   {i}. {text}")
        
        # 执行推理
        results = viewer.inference_texts(
            args.texts, 
            args.language, 
            args.mode, 
            args.max_length
        )
        
        if len(results) == 0:
            print("❌ 没有成功的推理结果")
            return
        
        # 打印摘要
        viewer.print_summary()
        
        # 启动可视化
        viewer.visualize_results()
        
    except KeyboardInterrupt:
        print("\n👋 用户退出")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 