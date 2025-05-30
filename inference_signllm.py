"""
SignLLM推理脚本
用于从文本生成手语姿态 - 支持优化版本模型
"""

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

from signllm_model_optimized import OptimizedSignLLM, ModelConfig, CONFIG
from utils import load_checkpoint, PoseVisualizer, get_device_info
from data_processor import PoseNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignLLMInference:
    """SignLLM推理器"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 加载配置
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # 使用默认配置
            self.config = self._get_default_config()
        
        # 初始化模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 初始化可视化器
        self.visualizer = PoseVisualizer(pose_dim=self.config["model"]["pose_dim"])
        
        logger.info("SignLLM inference ready!")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "model": {
                "languages": ["ASL"],
                "model_size": "medium",  # 恢复为medium配置
                "pose_dim": 150
            }
        }
    
    def _load_model(self, model_path: str) -> OptimizedSignLLM:
        """加载模型"""
        # 设置全局配置
        global CONFIG
        CONFIG = ModelConfig(self.config["model"]["model_size"])
        
        # 创建模型
        model = OptimizedSignLLM(languages=self.config["model"]["languages"])
        
        # 先运行一次前向传播来创建动态层
        dummy_text = ["hello"]
        with torch.no_grad():
            model(dummy_text, "ASL", max_length=16)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def generate_poses(self, 
                      texts: List[str], 
                      language: str = "ASL",
                      mode: str = "mlsf",
                      max_length: int = 256) -> np.ndarray:
        """生成手语姿态"""
        if language not in self.config["model"]["languages"]:
            raise ValueError(f"Unsupported language: {language}")
        
        with torch.no_grad():
            # 优化模型只支持基本生成，不区分mode
            poses, quality_scores = self.model(texts, language, max_length=max_length)
            return poses.cpu().numpy(), quality_scores.cpu().numpy()
    
    def generate_single(self, 
                       text: str, 
                       language: str = "ASL",
                       mode: str = "mlsf",
                       max_length: int = 256,
                       visualize: bool = True,
                       output_dir: str = None) -> Dict:
        """生成单个文本的手语姿态"""
        logger.info(f"Generating sign language for: '{text}' in {language}")
        
        # 生成姿态
        poses, quality_scores = self.generate_poses([text], language, mode, max_length)
        poses = poses[0]  # 取第一个样本
        quality_scores = quality_scores[0]
        
        result = {
            "text": text,
            "language": language,
            "mode": mode,
            "poses": poses,
            "quality_scores": quality_scores,
            "num_frames": len(poses)
        }
        
        # 可视化
        if visualize and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存姿态序列图像
            img_path = output_path / f"{language}_{mode}_poses.png"
            self.visualizer.visualize_pose_sequence(
                poses, 
                str(img_path),
                title=f"{language} - {text}"
            )
            
            # 创建姿态视频
            video_path = output_path / f"{language}_{mode}_poses.mp4"
            self.visualizer.create_pose_video(poses, str(video_path))
            
            result["visualization_paths"] = {
                "image": str(img_path),
                "video": str(video_path)
            }
        
        return result
    
    def batch_generate(self, 
                      texts: List[str], 
                      language: str = "ASL",
                      mode: str = "mlsf",
                      max_length: int = 256,
                      output_dir: str = None) -> List[Dict]:
        """批量生成手语姿态"""
        logger.info(f"Batch generating {len(texts)} texts in {language}")
        
        results = []
        
        poses_batch, quality_scores_batch = self.generate_poses(texts, language, mode, max_length)
        
        for i, (text, poses, quality_scores) in enumerate(zip(texts, poses_batch, quality_scores_batch)):
            result = {
                "text": text,
                "language": language,
                "mode": mode,
                "poses": poses,
                "quality_scores": quality_scores,
                "num_frames": len(poses)
            }
            results.append(result)
        
        # 保存结果
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存JSON结果（不包含poses数组，太大）
            json_results = []
            for i, result in enumerate(results):
                json_result = {k: v for k, v in result.items() if k != "poses"}
                json_result["poses_shape"] = result["poses"].shape
                json_results.append(json_result)
            
            with open(output_path / "batch_results.json", 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            # 保存poses数据
            poses_data = {f"sample_{i}": result["poses"] for i, result in enumerate(results)}
            np.savez_compressed(output_path / "batch_poses.npz", **poses_data)
        
        return results
    
    def interactive_demo(self):
        """交互式演示"""
        print("=== SignLLM Interactive Demo ===")
        print("Supported languages:", ", ".join(self.config["model"]["languages"]))
        print("Supported modes: mlsf, prompt2langgloss")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                # 获取用户输入
                text = input("Enter text: ").strip()
                if text.lower() == 'quit':
                    break
                
                language = input(f"Language (default: ASL): ").strip() or "ASL"
                mode = input(f"Mode (default: mlsf): ").strip() or "mlsf"
                
                if language not in self.config["model"]["languages"]:
                    print(f"Unsupported language: {language}")
                    continue
                
                if mode not in ["mlsf", "prompt2langgloss"]:
                    print(f"Unsupported mode: {mode}")
                    continue
                
                # 生成姿态
                print(f"\nGenerating sign language for: '{text}'...")
                result = self.generate_single(
                    text, 
                    language, 
                    mode,
                    visualize=True,
                    output_dir=f"./demo_output/{language}_{mode}"
                )
                
                print(f"Generated {result['num_frames']} frames")
                print(f"Average quality score: {np.mean(result['quality_scores']):.3f}")
                
                if "visualization_paths" in result:
                    print(f"Visualization saved to: {result['visualization_paths']['image']}")
                    print(f"Video saved to: {result['visualization_paths']['video']}")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        print("Demo ended.")


def main():
    parser = argparse.ArgumentParser(description="SignLLM Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--text", type=str, help="Text to convert to sign language")
    parser.add_argument("--texts_file", type=str, help="File containing texts (one per line)")
    parser.add_argument("--language", type=str, default="ASL", help="Target sign language")
    parser.add_argument("--mode", type=str, default="mlsf", choices=["mlsf", "prompt2langgloss"], help="Generation mode")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="./inference_output", help="Output directory")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = SignLLMInference(args.model_path, args.config_path)
    
    if args.interactive:
        # 交互式模式
        inference.interactive_demo()
    
    elif args.text:
        # 单个文本
        result = inference.generate_single(
            args.text,
            args.language,
            args.mode,
            args.max_length,
            args.visualize,
            args.output_dir
        )
        
        print(f"Generated {result['num_frames']} frames for: '{args.text}'")
        print(f"Average quality score: {np.mean(result['quality_scores']):.3f}")
    
    elif args.texts_file:
        # 批量处理
        with open(args.texts_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = inference.batch_generate(
            texts,
            args.language,
            args.mode,
            args.max_length,
            args.output_dir
        )
        
        print(f"Processed {len(results)} texts")
        avg_quality = np.mean([np.mean(r['quality_scores']) for r in results])
        print(f"Average quality score: {avg_quality:.3f}")
    
    else:
        print("Please specify --text, --texts_file, or --interactive")


if __name__ == "__main__":
    main() 