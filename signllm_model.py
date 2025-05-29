"""
SignLLM: Sign Language Production Large Language Models
完整的模型实现，包含MLSF和Prompt2LangGloss模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np


class ModelConfig:
    """统一的模型配置类 - 所有参数都在这里管理"""
    def __init__(self, model_size: str = "small"):
        """
        Args:
            model_size: 模型大小 ("tiny", "small", "medium", "large")
        """
        self.model_size = model_size
        
        # 根据模型大小设置所有参数
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
            self.num_layers = 8
            self.num_heads = 16
            self.ff_multiplier = 4
            self.gloss_vocab_size = 10000
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        # 计算派生参数
        self.dim_feedforward = self.hidden_dim * self.ff_multiplier
        
        # 固定参数
        self.pose_dim = 150
        self.dropout = 0.1
        self.max_sequence_length = 512
        self.bert_model = "bert-base-multilingual-cased"
        self.num_priorities = 8
        self.num_languages = 8
        
        # 帧数控制参数
        self.default_max_frames = 256    # 默认生成帧数
        self.min_frames = 30            # 最小帧数
        self.max_frames = 500           # 最大帧数
        
    def print_config(self):
        """打印配置信息"""
        print(f"🔧 模型配置 ({self.model_size}):")
        print(f"  Hidden Dim: {self.hidden_dim}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Attention Heads: {self.num_heads}")
        print(f"  FF Dim: {self.dim_feedforward}")
        print(f"  Gloss Vocab: {self.gloss_vocab_size}")
        print(f"  Pose Dim: {self.pose_dim}")
        print(f"  默认生成帧数: {self.default_max_frames}")
        print(f"  帧数范围: {self.min_frames}-{self.max_frames}")
        
    def estimate_params(self):
        """估算模型参数数量"""
        # 这是一个粗略估算
        bert_params = 110_000_000  # BERT base参数
        projection_params = 768 * self.hidden_dim
        
        # Transformer层参数 (粗略估算)
        attention_params = 4 * self.hidden_dim * self.hidden_dim * self.num_layers
        ff_params = 2 * self.hidden_dim * self.dim_feedforward * self.num_layers
        
        # 其他组件
        embedding_params = self.gloss_vocab_size * self.hidden_dim
        other_params = self.hidden_dim * 1000  # 其他小组件
        
        total = bert_params + projection_params + attention_params + ff_params + embedding_params + other_params
        return total


# 全局配置实例
CONFIG = ModelConfig("small")  # 默认使用small模型


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class PriorityLearningChannel(nn.Module):
    """Priority Learning Channel (PLC) - 强化学习模块"""
    def __init__(self):
        super().__init__()
        
        # 优先级注意力机制
        self.priority_attention = nn.MultiheadAttention(
            embed_dim=CONFIG.hidden_dim,
            num_heads=CONFIG.num_heads,
            dropout=CONFIG.dropout
        )
        
        # 优先级权重网络
        self.priority_weights = nn.Sequential(
            nn.Linear(CONFIG.hidden_dim, CONFIG.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(CONFIG.hidden_dim // 2, CONFIG.num_priorities),
            nn.Softmax(dim=-1)
        )
        
        # 质量评估网络
        self.quality_estimator = nn.Sequential(
            nn.Linear(CONFIG.hidden_dim, CONFIG.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(CONFIG.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len]
        Returns:
            enhanced_x: 增强后的特征
            quality_scores: 质量分数
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # 优先级注意力
        attn_output, _ = self.priority_attention(x, x, x, key_padding_mask=mask)
        
        # 计算优先级权重
        priority_weights = self.priority_weights(attn_output)  # [batch_size, seq_len, num_priorities]
        
        # 质量评估
        quality_scores = self.quality_estimator(attn_output).squeeze(-1)  # [batch_size, seq_len]
        
        # 加权融合
        enhanced_x = attn_output * quality_scores.unsqueeze(-1)
        
        return enhanced_x, quality_scores


class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.bert_model)
        self.encoder = AutoModel.from_pretrained(CONFIG.bert_model)
        self.projection = nn.Linear(self.encoder.config.hidden_size, CONFIG.hidden_dim)
        self.layer_norm = nn.LayerNorm(CONFIG.hidden_dim)
        
    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Args:
            texts: 输入文本列表
            device: 设备
        Returns:
            encoded_texts: [batch_size, seq_len, hidden_dim]
        """
        # 分词和编码
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        # 获取文本特征
        outputs = self.encoder(**inputs)
        text_features = outputs.last_hidden_state  # [batch_size, seq_len, bert_hidden]
        
        # 投影到目标维度
        projected = self.projection(text_features)
        return self.layer_norm(projected)


class GlossEncoder(nn.Module):
    """Gloss编码器"""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG.gloss_vocab_size, CONFIG.hidden_dim)
        self.pos_encoding = PositionalEncoding(CONFIG.hidden_dim, CONFIG.max_sequence_length)
        self.layer_norm = nn.LayerNorm(CONFIG.hidden_dim)
        
    def forward(self, gloss_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gloss_ids: [batch_size, seq_len]
        Returns:
            encoded_gloss: [batch_size, seq_len, hidden_dim]
        """
        embedded = self.embedding(gloss_ids)
        positioned = self.pos_encoding(embedded)
        return self.layer_norm(positioned)


class PoseDecoder(nn.Module):
    """姿态解码器"""
    def __init__(self):
        super().__init__()
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=CONFIG.hidden_dim,
            nhead=CONFIG.num_heads,
            dim_feedforward=CONFIG.dim_feedforward,
            dropout=CONFIG.dropout,
            activation="gelu"
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, CONFIG.num_layers)
        
        # 姿态投影层
        self.pose_projection = nn.Sequential(
            nn.Linear(CONFIG.hidden_dim, CONFIG.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(CONFIG.dropout),
            nn.Linear(CONFIG.hidden_dim // 2, CONFIG.pose_dim)
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(CONFIG.hidden_dim)
        
    def forward(self, memory: torch.Tensor, tgt: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            memory: 编码器输出 [seq_len, batch_size, hidden_dim]
            tgt: 目标序列 [tgt_len, batch_size, hidden_dim]
            tgt_mask: 目标掩码
            memory_mask: 记忆掩码
        Returns:
            poses: [batch_size, tgt_len, pose_dim]
        """
        # 添加位置编码
        tgt = self.pos_encoding(tgt)
        
        # Transformer解码
        decoded = self.transformer_decoder(
            tgt, memory, 
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_mask
        )
        
        # 转换维度并投影到姿态空间
        decoded = decoded.transpose(0, 1)  # [batch_size, tgt_len, hidden_dim]
        poses = self.pose_projection(decoded)
        
        return poses


class MLSFMode(nn.Module):
    """Multi-Language Switching Framework (MLSF) 模式"""
    def __init__(self, languages: List[str]):
        super().__init__()
        self.languages = languages
        
        # 每种语言的文本编码器
        self.text_encoders = nn.ModuleDict({
            lang: TextEncoder() for lang in languages
        })
        
        # 每种语言的姿态解码器
        self.pose_decoders = nn.ModuleDict({
            lang: PoseDecoder() for lang in languages
        })
        
        # Priority Learning Channel
        self.plc = PriorityLearningChannel()
        
    def forward(self, texts: List[str], language: str, max_length: int = None) -> torch.Tensor:
        """
        Args:
            texts: 输入文本列表
            language: 目标语言
            max_length: 最大生成长度 (None时使用默认值)
        Returns:
            poses: [batch_size, max_length, pose_dim]
        """
        if max_length is None:
            max_length = CONFIG.default_max_frames
            
        # 限制帧数范围
        max_length = max(CONFIG.min_frames, min(max_length, CONFIG.max_frames))
        
        device = next(self.parameters()).device
        batch_size = len(texts)
        
        # 文本编码
        text_features = self.text_encoders[language](texts, device)
        
        # PLC增强
        enhanced_features, quality_scores = self.plc(text_features)
        
        # 准备解码器输入
        memory = enhanced_features.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        
        # 初始化目标序列（起始token）
        tgt = torch.zeros(1, batch_size, CONFIG.hidden_dim, device=device)
        
        # 自回归生成
        poses = []
        for _ in range(max_length):
            # 解码一步
            decoded = self.pose_decoders[language](memory, tgt)
            pose_step = decoded[:, -1:, :]  # 取最后一步
            poses.append(pose_step)
            
            # 更新目标序列 - 需要将pose_step投影回hidden_dim
            # 创建一个投影层将pose_dim映射回hidden_dim
            if not hasattr(self, 'pose_to_hidden'):
                self.pose_to_hidden = nn.Linear(CONFIG.pose_dim, CONFIG.hidden_dim).to(device)
            
            hidden_step = self.pose_to_hidden(pose_step).transpose(0, 1)  # [1, batch_size, hidden_dim]
            tgt = torch.cat([tgt, hidden_step], dim=0)
        
        return torch.cat(poses, dim=1), quality_scores


class Prompt2LangGlossMode(nn.Module):
    """Prompt2LangGloss模式"""
    def __init__(self):
        super().__init__()
        
        # 文本编码器
        self.text_encoder = TextEncoder()
        
        # Gloss生成器
        self.gloss_generator = nn.Sequential(
            nn.Linear(CONFIG.hidden_dim, CONFIG.hidden_dim),
            nn.ReLU(),
            nn.Dropout(CONFIG.dropout),
            nn.Linear(CONFIG.hidden_dim, CONFIG.gloss_vocab_size)
        )
        
        # Gloss编码器
        self.gloss_encoder = GlossEncoder()
        
        # 姿态解码器
        self.pose_decoder = PoseDecoder()
        
        # Priority Learning Channel
        self.plc = PriorityLearningChannel()
        
        # 语言标记嵌入
        self.language_embedding = nn.Embedding(CONFIG.num_languages, CONFIG.hidden_dim)
        
    def forward(self, texts: List[str], language_ids: torch.Tensor, 
                max_gloss_length: int = 128, max_pose_length: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            texts: 输入文本列表
            language_ids: 语言ID [batch_size]
            max_gloss_length: 最大gloss长度
            max_pose_length: 最大姿态长度 (None时使用默认值)
        Returns:
            poses: [batch_size, max_pose_length, pose_dim]
            gloss_logits: [batch_size, max_gloss_length, gloss_vocab_size]
        """
        if max_pose_length is None:
            max_pose_length = CONFIG.default_max_frames
            
        # 限制帧数范围
        max_pose_length = max(CONFIG.min_frames, min(max_pose_length, CONFIG.max_frames))
        
        device = next(self.parameters()).device
        batch_size = len(texts)
        
        # 确保language_ids在正确的设备上
        language_ids = language_ids.to(device)
        
        # 文本编码
        text_features = self.text_encoder(texts, device)
        
        # 添加语言标记
        lang_embeds = self.language_embedding(language_ids).unsqueeze(1)
        text_features = text_features + lang_embeds
        
        # PLC增强
        enhanced_features, quality_scores = self.plc(text_features)
        
        # 生成Gloss
        gloss_logits = self.gloss_generator(enhanced_features)
        gloss_ids = torch.argmax(gloss_logits, dim=-1)
        
        # Gloss编码
        gloss_features = self.gloss_encoder(gloss_ids)
        
        # 姿态解码
        memory = gloss_features.transpose(0, 1)
        tgt = torch.zeros(1, batch_size, CONFIG.hidden_dim, device=device)
        
        poses = []
        for _ in range(max_pose_length):
            decoded = self.pose_decoder(memory, tgt)
            pose_step = decoded[:, -1:, :]
            poses.append(pose_step)
            
            # 更新目标序列 - 需要将pose_step投影回hidden_dim
            if not hasattr(self, 'pose_to_hidden_p2lg'):
                self.pose_to_hidden_p2lg = nn.Linear(CONFIG.pose_dim, CONFIG.hidden_dim).to(device)
            
            hidden_step = self.pose_to_hidden_p2lg(pose_step).transpose(0, 1)
            tgt = torch.cat([tgt, hidden_step], dim=0)
        
        return torch.cat(poses, dim=1), gloss_logits, quality_scores


class SignLLM(nn.Module):
    """SignLLM主模型"""
    def __init__(self, languages: List[str] = ["ASL"]):
        super().__init__()
        self.languages = languages
        self.language_to_id = {lang: i for i, lang in enumerate(languages)}
        
        # 打印模型配置
        CONFIG.print_config()
        estimated_params = CONFIG.estimate_params()
        print(f"  估计参数量: {estimated_params:,} ({estimated_params/1_000_000:.1f}M)")
        
        # MLSF模式
        self.mlsf_mode = MLSFMode(languages)
        
        # Prompt2LangGloss模式
        self.prompt2langgloss_mode = Prompt2LangGlossMode()
        
    def forward(self, texts: List[str], language: str, mode: str = "mlsf", **kwargs):
        """
        Args:
            texts: 输入文本列表
            language: 目标语言
            mode: 模式 ("mlsf" 或 "prompt2langgloss")
        Returns:
            根据模式返回不同的输出
        """
        if mode == "mlsf":
            return self.mlsf_mode(texts, language, **kwargs)
        elif mode == "prompt2langgloss":
            language_ids = torch.tensor([self.language_to_id[language]] * len(texts))
            if torch.cuda.is_available():
                language_ids = language_ids.cuda()
            return self.prompt2langgloss_mode(texts, language_ids, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class RLLoss(nn.Module):
    """强化学习损失函数"""
    def __init__(self, alpha: float = 0.1, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha  # 质量权重
        self.beta = beta   # 多样性权重
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, pred_poses: torch.Tensor, target_poses: torch.Tensor, 
                quality_scores: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_poses: 预测姿态 [batch_size, seq_len, pose_dim]
            target_poses: 目标姿态 [batch_size, seq_len, pose_dim]
            quality_scores: 质量分数 [batch_size, seq_len]
            mask: 掩码 [batch_size, seq_len]
        Returns:
            loss: 总损失
        """
        # 基础MSE损失
        mse_loss = self.mse_loss(pred_poses, target_poses).mean(dim=-1)  # [batch_size, seq_len]
        
        # 质量加权损失
        quality_weighted_loss = mse_loss * (1.0 + self.alpha * (1.0 - quality_scores))
        
        # 多样性正则化
        diversity_loss = -self.beta * torch.var(quality_scores, dim=1).mean()
        
        # 应用掩码
        if mask is not None:
            quality_weighted_loss = quality_weighted_loss * mask
            quality_weighted_loss = quality_weighted_loss.sum() / mask.sum()
        else:
            quality_weighted_loss = quality_weighted_loss.mean()
        
        return quality_weighted_loss + diversity_loss


def create_causal_mask(size: int) -> torch.Tensor:
    """创建因果掩码"""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """创建填充掩码"""
    batch_size = lengths.size(0)
    mask = torch.arange(max_len).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    return mask


if __name__ == "__main__":
    # 测试代码
    model = SignLLM()
    texts = ["Hello world", "How are you?"]
    
    # 测试MLSF模式
    poses_mlsf, quality_scores = model(texts, "ASL", mode="mlsf", max_length=100)
    print(f"MLSF输出形状: {poses_mlsf.shape}")
    
    # 测试Prompt2LangGloss模式
    poses_p2lg, gloss_logits, quality_scores = model(texts, "ASL", mode="prompt2langgloss")
    print(f"Prompt2LangGloss姿态输出形状: {poses_p2lg.shape}")
    print(f"Gloss输出形状: {gloss_logits.shape}") 