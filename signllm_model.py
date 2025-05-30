"""
SignLLM: Sign Language Production Large Language Models
⚠️ 2025‑05 teacher‑forcing 版本
  * 与 2025‑05 修复版保持接口兼容，新增 Teacher‑Forcing 训练路径
  * 当 forward 接收 target_poses 时使用并行解码；否则保持原逐帧自回归推理
"""

import math
from typing import Dict, List, Optional

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

    # print_config/estimate_params 同前版本，省略


CONFIG = ModelConfig("medium")


def causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    """生成大小为 (sz, sz) 的上三角布尔 mask"""
    return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len,1,d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)]


class PriorityLearningChannel(nn.Module):
    def __init__(self):
        super().__init__()
        self.priority_attention = nn.MultiheadAttention(
            embed_dim=CONFIG.hidden_dim,
            num_heads=CONFIG.num_heads,
            dropout=CONFIG.dropout,
            batch_first=True,
        )
        self.priority_weights = nn.Sequential(
            nn.Linear(CONFIG.hidden_dim, CONFIG.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(CONFIG.hidden_dim // 2, CONFIG.num_priorities),
            nn.Softmax(dim=-1),
        )
        self.quality_estimator = nn.Sequential(
            nn.Linear(CONFIG.hidden_dim, CONFIG.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(CONFIG.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_output, _ = self.priority_attention(x, x, x, key_padding_mask=mask)
        priority_weights = self.priority_weights(attn_output)
        quality_scores = self.quality_estimator(attn_output).squeeze(-1)
        enhanced_x = attn_output * quality_scores.unsqueeze(-1)
        return enhanced_x, quality_scores


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.bert_model)
        self.encoder = AutoModel.from_pretrained(CONFIG.bert_model)
        self.proj = nn.Linear(self.encoder.config.hidden_size, CONFIG.hidden_dim)
        self.ln = nn.LayerNorm(CONFIG.hidden_dim)

    def forward(self, texts: List[str], device):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        features = self.encoder(**inputs).last_hidden_state
        return self.ln(self.proj(features))


class GlossEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG.gloss_vocab_size, CONFIG.hidden_dim)
        self.pos = PositionalEncoding(CONFIG.hidden_dim, CONFIG.max_sequence_length)
        self.ln = nn.LayerNorm(CONFIG.hidden_dim)

    def forward(self, gloss_ids: torch.Tensor):
        emb = self.embedding(gloss_ids).transpose(0, 1)  # (T,B,H)
        return self.ln(self.pos(emb)).transpose(0, 1)


class PoseDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=CONFIG.hidden_dim,
            nhead=CONFIG.num_heads,
            dim_feedforward=CONFIG.dim_feedforward,
            dropout=CONFIG.dropout,
            activation="gelu",
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(layer, CONFIG.num_layers)
        self.proj = nn.Sequential(
            nn.Linear(CONFIG.hidden_dim, CONFIG.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(CONFIG.dropout),
            nn.Linear(CONFIG.hidden_dim // 2, CONFIG.pose_dim),
        )
        self.pos = PositionalEncoding(CONFIG.hidden_dim)

    def forward(self, memory, tgt, tgt_mask=None, memory_mask=None):
        dec = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        return self.proj(dec.transpose(0, 1))


class MLSFMode(nn.Module):
    def __init__(self, languages: List[str]):
        super().__init__()
        self.text_encoders = nn.ModuleDict({lang: TextEncoder() for lang in languages})
        self.pose_decoders = nn.ModuleDict({lang: PoseDecoder() for lang in languages})
        self.plc = PriorityLearningChannel()
        self.pose_to_hidden = nn.Linear(CONFIG.pose_dim, CONFIG.hidden_dim)

    def forward(
        self,
        texts: List[str],
        language: str,
        target_poses: Optional[torch.Tensor] = None,
        max_length: int = None,
    ):
        device = next(self.parameters()).device
        batch = len(texts)
        text_feat = self.text_encoders[language](texts, device)
        memory, quality_scores = self.plc(text_feat)
        memory = memory.transpose(0, 1)  # (T,B,H)

        # --- Teacher‑Forcing 路径 ---
        if target_poses is not None:
            T = target_poses.size(1)
            start_token = torch.zeros(batch, 1, CONFIG.pose_dim, device=device)
            pose_in = torch.cat([start_token, target_poses[:, :-1, :]], dim=1)  # (B,T,P)
            tgt_hidden = self.pose_to_hidden(pose_in).transpose(0, 1)  # (T,B,H)
            tgt_hidden = self.pose_decoders[language].pos(tgt_hidden)
            tgt_mask = causal_mask(tgt_hidden.size(0), device)
            pred = self.pose_decoders[language](memory, tgt_hidden, tgt_mask=tgt_mask)
            return pred, quality_scores  # pred: (B,T,P)

        # --- 推理自回归路径（与原版本保持一致） ---
        if max_length is None:
            max_length = CONFIG.default_max_frames
        max_length = max(CONFIG.min_frames, min(max_length, CONFIG.max_frames))
        tgt = torch.zeros(1, batch, CONFIG.hidden_dim, device=device)
        tgt = self.pose_decoders[language].pos(tgt)
        poses = []
        for _ in range(max_length):
            tgt_mask = causal_mask(tgt.size(0), device)
            decoded = self.pose_decoders[language](memory, tgt, tgt_mask=tgt_mask)
            pose_step = decoded[:, -1:, :]
            poses.append(pose_step)
            hidden_step = self.pose_to_hidden(pose_step).transpose(0, 1)
            hidden_step = self.pose_decoders[language].pos(hidden_step)
            tgt = torch.cat([tgt, hidden_step], dim=0)
        return torch.cat(poses, dim=1), quality_scores


class Prompt2LangGlossMode(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.gloss_generator = nn.Sequential(
            nn.Linear(CONFIG.hidden_dim, CONFIG.hidden_dim),
            nn.GELU(),
            nn.Dropout(CONFIG.dropout),
            nn.Linear(CONFIG.hidden_dim, CONFIG.gloss_vocab_size),
        )
        self.gloss_encoder = GlossEncoder()
        self.pose_decoder = PoseDecoder()
        self.plc = PriorityLearningChannel()
        self.language_embedding = nn.Embedding(CONFIG.num_languages, CONFIG.hidden_dim)
        self.pose_to_hidden = nn.Linear(CONFIG.pose_dim, CONFIG.hidden_dim)

    def forward(
        self,
        texts: List[str],
        language_ids: torch.Tensor,
        target_poses: Optional[torch.Tensor] = None,
        max_pose_length: int = None,
    ):
        device = next(self.parameters()).device
        batch = len(texts)
        language_ids = language_ids.to(device)
        text_feat = self.text_encoder(texts, device) + self.language_embedding(language_ids).unsqueeze(1)
        enhanced, quality_scores = self.plc(text_feat)

        gloss_logits = self.gloss_generator(enhanced)
        gloss_ids = gloss_logits.argmax(dim=-1)
        gloss_feat = self.gloss_encoder(gloss_ids).transpose(0, 1)  # (T,B,H)

        # --- Teacher‑Forcing path ---
        if target_poses is not None:
            T = target_poses.size(1)
            start_token = torch.zeros(batch, 1, CONFIG.pose_dim, device=device)
            pose_in = torch.cat([start_token, target_poses[:, :-1, :]], dim=1)
            tgt_hidden = self.pose_to_hidden(pose_in).transpose(0, 1)
            tgt_hidden = self.pose_decoder.pos(tgt_hidden)
            tgt_mask = causal_mask(T, device)
            pred = self.pose_decoder(gloss_feat, tgt_hidden, tgt_mask=tgt_mask)
            return pred, gloss_logits, quality_scores

        # --- Autoregressive inference ---
        if max_pose_length is None:
            max_pose_length = CONFIG.default_max_frames
        max_pose_length = max(CONFIG.min_frames, min(max_pose_length, CONFIG.max_frames))
        tgt = torch.zeros(1, batch, CONFIG.hidden_dim, device=device)
        tgt = self.pose_decoder.pos(tgt)
        poses = []
        for _ in range(max_pose_length):
            tgt_mask = causal_mask(tgt.size(0), device)
            decoded = self.pose_decoder(gloss_feat, tgt, tgt_mask=tgt_mask)
            pose_step = decoded[:, -1:, :]
            poses.append(pose_step)
            hidden_step = self.pose_to_hidden(pose_step).transpose(0, 1)
            hidden_step = self.pose_decoder.pos(hidden_step)
            tgt = torch.cat([tgt, hidden_step], dim=0)
        return torch.cat(poses, dim=1), gloss_logits, quality_scores


class SignLLM(nn.Module):
    def __init__(self, languages: List[str] = ["ASL"]):
        super().__init__()
        self.languages = languages
        self.language_to_id = {lang: i for i, lang in enumerate(languages)}
        self.mlsf_mode = MLSFMode(languages)
        self.prompt2langgloss_mode = Prompt2LangGlossMode()

    def forward(
        self,
        texts: List[str],
        language: str,
        mode: str = "mlsf",
        target_poses: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if mode == "mlsf":
            return self.mlsf_mode(texts, language, target_poses=target_poses, **kwargs)
        elif mode == "prompt2langgloss":
            language_ids = torch.tensor([self.language_to_id[language]] * len(texts))
            return self.prompt2langgloss_mode(texts, language_ids, target_poses=target_poses, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ---------------- demo ----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SignLLM().to(device)
    texts = ["Hello", "How are you?"]
    # Fake ground‑truth poses (batch, T, P)
    gt = torch.randn(2, 64, CONFIG.pose_dim, device=device)
    pred, q = model(texts, "ASL", mode="mlsf", target_poses=gt)
    print("Teacher‑forcing pred:", pred.shape, q.shape)
    # Inference
    poses, q_inf = model(texts, "ASL", mode="mlsf", max_length=64)
    print("Inference poses:", poses.shape)
