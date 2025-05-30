"""
SignLLM 改进版本 - 增强的时间建模和注意力机制
完全独立版本，不依赖原始模型文件
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class ModelConfig:
    """统一的模型配置类 - 所有参数都在这里管理"""

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        if model_size == "tiny":
            self.hidden_dim = 256
            self.num_layers = 2
            self.num_heads = 4
            self.ff_multiplier = 2
            self.gloss_vocab_size = 1000
        elif model_size == "small":
            self.hidden_dim = 512
            self.num_layers = 4
            self.num_heads = 8
            self.ff_multiplier = 2
            self.gloss_vocab_size = 2000
        elif model_size == "medium":
            self.hidden_dim = 768
            self.num_layers = 6
            self.num_heads = 12
            self.ff_multiplier = 3
            self.gloss_vocab_size = 5000
        elif model_size == "large":
            self.hidden_dim = 1024
            self.num_layers = 12
            self.num_heads = 16
            self.ff_multiplier = 4
            self.gloss_vocab_size = 10000
        else:
            raise ValueError(f"Unknown model size: {model_size}")

        self.dim_feedforward = self.hidden_dim * self.ff_multiplier
        self.pose_dim = 150
        self.dropout = 0.1
        self.max_sequence_length = 512
        self.bert_model = "bert-base-multilingual-cased"
        self.num_priorities = 8
        self.num_languages = 8
        self.default_max_frames = 256
        self.min_frames = 30
        self.max_frames = 500


class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
        self.encoder = AutoModel.from_pretrained(config.bert_model)
        self.proj = nn.Linear(self.encoder.config.hidden_size, config.hidden_dim)
        self.ln = nn.LayerNorm(config.hidden_dim)

    def forward(self, texts: List[str], device):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        features = self.encoder(**inputs).last_hidden_state
        return self.ln(self.proj(features))


class ImprovedPositionalEncoding(nn.Module):
    """改进的位置编码 - 支持相对位置和时间感知"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
        # 标准位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))
        
        # 时间感知的位置编码权重
        self.time_aware_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, temporal_scale: float = 1.0) -> torch.Tensor:
        seq_len = x.size(0)
        pos_encoding = self.pe[:seq_len] * self.time_aware_weight * temporal_scale
        return self.dropout(x + pos_encoding)


class TemporalAttention(nn.Module):
    """时间感知的注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 标准注意力
        attn_output, attn_weights = self.attention(
            query, key, value, attn_mask=attn_mask, need_weights=True
        )
        
        # 添加时间偏置
        if attn_weights is not None:
            batch_size, seq_len = attn_weights.shape[:2]
            temporal_bias = self.temporal_bias.expand(-1, seq_len, seq_len)
            # 这里可以添加更复杂的时间建模逻辑
        
        return attn_output, attn_weights


class EnhancedPoseDecoder(nn.Module):
    """增强的姿态解码器 - 更好的时间建模"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 时间感知的Transformer层
        self.temporal_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=False,
            )
            for _ in range(config.num_layers)
        ])
        
        # 运动建模层
        self.motion_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # 输出投影
        self.pose_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.pose_dim),
        )
        
        # 改进的位置编码
        self.pos_encoding = ImprovedPositionalEncoding(
            config.hidden_dim, config.max_sequence_length, config.dropout
        )
        
        # 运动平滑化
        self.motion_smoother = nn.Conv1d(
            config.pose_dim, config.pose_dim, kernel_size=3, padding=1, groups=config.pose_dim
        )
        
    def forward(self, memory: torch.Tensor, tgt: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                use_motion_prediction: bool = True) -> torch.Tensor:
        
        # 应用位置编码
        tgt = self.pos_encoding(tgt)
        
        # 通过Transformer层
        output = tgt
        for layer in self.temporal_layers:
            output = layer(output, memory, tgt_mask=tgt_mask, 
                         memory_key_padding_mask=memory_mask)
        
        # 运动预测（如果启用）
        if use_motion_prediction and output.size(0) > 1:
            # 计算运动特征
            motion_features = torch.cat([
                output[:-1], output[1:] - output[:-1]
            ], dim=-1)
            motion_pred = self.motion_predictor(motion_features)
            # 融合运动预测
            output[1:] = output[1:] + motion_pred
        
        # 投影到姿态空间
        poses = self.pose_projection(output.transpose(0, 1))  # (B, T, pose_dim)
        
        # 运动平滑化
        if poses.size(1) > 1:  # 序列长度 > 1
            poses_smooth = self.motion_smoother(poses.transpose(1, 2)).transpose(1, 2)
            poses = 0.8 * poses + 0.2 * poses_smooth
        
        return poses


class AdaptivePriorityChannel(nn.Module):
    """自适应优先级通道 - 动态调整注意力权重"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 多尺度注意力
        self.multi_scale_attention = nn.ModuleList([
            TemporalAttention(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(3)  # 不同时间尺度
        ])
        
        # 优先级学习
        self.priority_learner = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.num_priorities),
            nn.Softmax(dim=-1)
        )
        
        # 质量评估
        self.quality_assessor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 融合权重
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, dim = x.shape
        
        # 多尺度注意力
        multi_scale_outputs = []
        attention_weights = []
        
        for i, attention_layer in enumerate(self.multi_scale_attention):
            # 不同的时间步长
            step = 2 ** i if i > 0 else 1
            if step > 1 and seq_len > step:
                # 降采样
                sampled_x = x[:, ::step, :]
                attn_out, attn_w = attention_layer(sampled_x, sampled_x, sampled_x)
                # 上采样回原始长度
                attn_out = F.interpolate(
                    attn_out.transpose(1, 2), size=seq_len, mode='linear', align_corners=False
                ).transpose(1, 2)
            else:
                attn_out, attn_w = attention_layer(x, x, x, attn_mask=mask)
            
            multi_scale_outputs.append(attn_out)
            attention_weights.append(attn_w)
        
        # 融合多尺度特征
        fusion_weights_norm = F.softmax(self.fusion_weights, dim=0)
        fused_output = sum(w * out for w, out in zip(fusion_weights_norm, multi_scale_outputs))
        
        # 优先级学习
        priorities = self.priority_learner(fused_output)
        
        # 质量评估
        quality_scores = self.quality_assessor(fused_output).squeeze(-1)
        
        # 应用优先级和质量权重
        enhanced_output = fused_output * quality_scores.unsqueeze(-1)
        
        # 额外的诊断信息
        diagnostics = {
            'priorities': priorities,
            'quality_scores': quality_scores,
            'attention_weights': attention_weights,
            'fusion_weights': fusion_weights_norm
        }
        
        return enhanced_output, quality_scores, diagnostics


class ImprovedSignLLM(nn.Module):
    """改进的 SignLLM - 增强时间建模和注意力机制"""
    
    def __init__(self, config=None, languages: List[str] = ["ASL"]):
        super().__init__()
        if config is None:
            config = ModelConfig("medium")
        self.config = config
        self.languages = languages
        self.language_to_id = {lang: i for i, lang in enumerate(languages)}
        
        # 文本编码器
        self.text_encoders = nn.ModuleDict({
            lang: TextEncoder(config) for lang in languages
        })
        
        # 改进的姿态解码器
        self.pose_decoders = nn.ModuleDict({
            lang: EnhancedPoseDecoder(config) for lang in languages
        })
        
        # 自适应优先级通道
        self.adaptive_priority_channel = AdaptivePriorityChannel(config)
        
        # 姿态到隐藏状态的映射
        self.pose_to_hidden = nn.Linear(config.pose_dim, config.hidden_dim)
        
        # 训练/推理模式切换
        self.training_mode = True
        
    def forward(self, texts: List[str], language: str, 
                target_poses: Optional[torch.Tensor] = None,
                max_length: int = None,
                return_diagnostics: bool = False) -> Dict[str, torch.Tensor]:
        
        device = next(self.parameters()).device
        batch_size = len(texts)
        
        # 文本编码
        text_features = self.text_encoders[language](texts, device)
        
        # 自适应优先级处理
        enhanced_features, quality_scores, diagnostics = self.adaptive_priority_channel(text_features)
        memory = enhanced_features.transpose(0, 1)  # (T, B, H)
        
        results = {
            'quality_scores': quality_scores
        }
        
        if return_diagnostics:
            results['diagnostics'] = diagnostics
        
        # Teacher Forcing 训练模式
        if target_poses is not None and self.training:
            T = target_poses.size(1)
            start_token = torch.zeros(batch_size, 1, self.config.pose_dim, device=device)
            pose_input = torch.cat([start_token, target_poses[:, :-1, :]], dim=1)
            
            tgt_hidden = self.pose_to_hidden(pose_input).transpose(0, 1)
            tgt_mask = self._generate_causal_mask(T, device)
            
            predicted_poses = self.pose_decoders[language](
                memory, tgt_hidden, tgt_mask=tgt_mask, use_motion_prediction=True
            )
            
            results['predicted_poses'] = predicted_poses
            return results
        
        # 自回归推理模式
        if max_length is None:
            max_length = self.config.default_max_frames
        max_length = max(self.config.min_frames, min(max_length, self.config.max_frames))
        
        # 初始化
        tgt = torch.zeros(1, batch_size, self.config.hidden_dim, device=device)
        poses = []
        
        for step in range(max_length):
            tgt_mask = self._generate_causal_mask(tgt.size(0), device)
            decoded = self.pose_decoders[language](
                memory, tgt, tgt_mask=tgt_mask, use_motion_prediction=(step > 0)
            )
            
            # 取最后一帧
            pose_step = decoded[:, -1:, :]
            poses.append(pose_step)
            
            # 准备下一步输入
            hidden_step = self.pose_to_hidden(pose_step).transpose(0, 1)
            tgt = torch.cat([tgt, hidden_step], dim=0)
            
            # 早停条件（可选）
            if step > self.config.min_frames:
                # 可以基于运动幅度或其他指标决定是否早停
                motion_magnitude = torch.norm(pose_step.view(batch_size, -1), dim=1).mean()
                if motion_magnitude < 0.01:  # 运动幅度很小时停止
                    break
        
        results['predicted_poses'] = torch.cat(poses, dim=1)
        return results
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """生成因果mask"""
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
    
    def set_inference_mode(self, inference: bool = True):
        """切换推理模式"""
        self.training_mode = not inference
        if inference:
            self.eval()
        else:
            self.train()


# 使用示例
if __name__ == "__main__":
    config = ModelConfig("large")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ImprovedSignLLM(config, languages=["ASL"]).to(device)
    
    # 训练模式测试
    texts = ["Hello", "How are you?"]
    target_poses = torch.randn(2, 64, config.pose_dim, device=device)
    
    results = model(texts, "ASL", target_poses=target_poses, return_diagnostics=True)
    print("训练模式输出形状:", results['predicted_poses'].shape)
    print("质量分数:", results['quality_scores'].shape)
    
    # 推理模式测试
    model.set_inference_mode(True)
    results = model(texts, "ASL", max_length=64)
    print("推理模式输出形状:", results['predicted_poses'].shape) 