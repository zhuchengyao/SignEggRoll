import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Tuple, Union
from transformers import CLIPTextModel, CLIPTokenizer

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TextEncoder(nn.Module):
    """文本编码器，基于CLIP"""
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.text_projection = nn.Linear(512, 768)  # 将CLIP输出投影到所需维度
        
    def encode_text(self, text_prompts):
        """编码文本提示"""
        # 文本标记化
        tokens = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(next(self.parameters()).device)
        
        # 获取文本特征
        text_features = self.text_model(**tokens).last_hidden_state
        text_features = self.text_projection(text_features)
        
        return text_features  # (batch_size, seq_len, 768)

class CrossAttention(nn.Module):
    """交叉注意力机制，用于文本-视觉条件融合"""
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], h, -1).transpose(1, 2), (q, k, v))

        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).contiguous().view(out.shape[0], out.shape[2], -1)
        
        return self.to_out(out)

class ResidualBlock1D(nn.Module):
    """1D残差块，用于序列处理"""
    def __init__(self, in_channels, out_channels, time_emb_dim, text_emb_dim=768, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # 确保GroupNorm的group数不超过通道数
        group_size = min(8, in_channels, out_channels)
        
        # 1D卷积层
        self.block1 = nn.Sequential(
            nn.GroupNorm(group_size, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
        )
        
        # 交叉注意力用于文本条件
        self.cross_attn = CrossAttention(out_channels, text_emb_dim)
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(group_size, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb, text_context=None):
        # x: (B, C, T)
        h = self.block1(x)
        
        # 时间嵌入
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None]
        
        # 文本条件（如果提供）
        if text_context is not None:
            B, C, T = h.shape
            # 重塑为序列形式进行注意力计算
            h_reshaped = h.transpose(1, 2)  # (B, T, C)
            h_attended = self.cross_attn(h_reshaped, text_context)
            h = h_attended.transpose(1, 2)  # (B, C, T)
        
        h = self.block2(h)
        return h + self.shortcut(x)

class TemporalTransformerLayer(nn.Module):
    """时间注意力Transformer层"""
    def __init__(self, d_model, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x

class PoseUNet1D(nn.Module):
    """1D U-Net网络，用于文本条件化的姿态序列生成"""
    def __init__(
        self, 
        pose_dim=120,  # ASL姿态特征维度
        model_channels=256,
        num_res_blocks=2,
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        num_frames=100,
        text_emb_dim=768,
        use_transformer=True
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_frames = num_frames
        self.use_transformer = use_transformer

        # 文本编码器
        self.text_encoder = TextEncoder()
        
        # 时间嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 输入投影: (B, T, 120) -> (B, model_channels, T)
        self.input_proj = nn.Linear(pose_dim, model_channels)

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_sample_layers = nn.ModuleList()
        ch = model_channels
        self.skip_channels = [ch]  # 记录跳跃连接的通道数
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock1D(ch, mult * model_channels, time_embed_dim, text_emb_dim, dropout)
                )
                ch = mult * model_channels
                self.skip_channels.append(ch)
            
            if level != len(channel_mult) - 1:
                # 时间维度下采样
                self.down_sample_layers.append(nn.Conv1d(ch, ch, 3, stride=2, padding=1))
                self.skip_channels.append(ch)
            else:
                self.down_sample_layers.append(None)

        # 中间层
        self.middle = nn.ModuleList([
            ResidualBlock1D(ch, ch, time_embed_dim, text_emb_dim, dropout),
        ])
        
        # 添加Transformer层用于长序列建模
        if use_transformer:
            self.middle.append(
                TemporalTransformerLayer(ch, num_heads=8, dim_feedforward=ch*2, dropout=dropout)
            )
        
        self.middle.append(
            ResidualBlock1D(ch, ch, time_embed_dim, text_emb_dim, dropout)
        )

        # 上采样路径 - 重新设计
        self.up_blocks = nn.ModuleList()
        self.up_sample_layers = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # 计算输入通道数（当前通道数 + 跳跃连接通道数）
                skip_ch = self.skip_channels.pop() if self.skip_channels else 0
                input_ch = ch + skip_ch
                output_ch = mult * model_channels
                
                self.up_blocks.append(
                    ResidualBlock1D(input_ch, output_ch, time_embed_dim, text_emb_dim, dropout)
                )
                ch = output_ch
            
            if level != 0:
                # 时间维度上采样
                self.up_sample_layers.append(nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1))
            else:
                self.up_sample_layers.append(None)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, model_channels), model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, pose_dim),
        )

    def forward(self, x, timesteps, text_prompts=None):
        """
        Args:
            x: (batch_size, num_frames, pose_dim) - 输入姿态序列
            timesteps: (batch_size,) - 时间步
            text_prompts: List[str] - 文本提示
        """
        # 文本编码
        text_context = None
        if text_prompts is not None:
            text_context = self.text_encoder.encode_text(text_prompts)
        
        # 时间嵌入
        t_emb = self.time_embed(timesteps)
        
        # 输入投影: (B, T, pose_dim) -> (B, T, model_channels) -> (B, model_channels, T)
        h = self.input_proj(x)  # (B, T, model_channels)
        h = h.transpose(1, 2)   # (B, model_channels, T)
        
        # 保存跳跃连接
        hs = [h]
        
        # 下采样
        down_block_idx = 0
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[down_block_idx](h, t_emb, text_context)
                hs.append(h)
                down_block_idx += 1
            
            if level != len(self.channel_mult) - 1:
                h = self.down_sample_layers[level](h)
                hs.append(h)
        
        # 中间层
        for module in self.middle:
            if isinstance(module, ResidualBlock1D):
                h = module(h, t_emb, text_context)
            elif isinstance(module, TemporalTransformerLayer):
                # Transformer需要 (B, T, C) 格式
                h_reshaped = h.transpose(1, 2)  # (B, T, C)
                h_reshaped = module(h_reshaped)
                h = h_reshaped.transpose(1, 2)  # (B, C, T)
            else:
                h = module(h)
        
        # 上采样
        up_block_idx = 0
        up_sample_idx = 0
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                # 添加跳跃连接
                if hs:
                    skip_h = hs.pop()
                    # 确保时间维度匹配
                    if h.shape[2] != skip_h.shape[2]:
                        # 使用插值调整时间维度
                        h = F.interpolate(h, size=skip_h.shape[2], mode='linear', align_corners=False)
                    h = torch.cat([h, skip_h], dim=1)
                
                # 应用残差块
                h = self.up_blocks[up_block_idx](h, t_emb, text_context)
                up_block_idx += 1
            
            # 上采样层（除了最后一层）
            if level != 0:
                if self.up_sample_layers[up_sample_idx] is not None:
                    h = self.up_sample_layers[up_sample_idx](h)
                up_sample_idx += 1
        
        # 输出投影: (B, model_channels, T) -> (B, T, model_channels) -> (B, T, pose_dim)
        h = h.transpose(1, 2)  # (B, T, model_channels)
        
        # 对于GroupNorm，我们需要转回(B, C, T)格式
        h_for_norm = h.transpose(1, 2)  # (B, model_channels, T)
        h_for_norm = self.output_proj[0](h_for_norm)  # GroupNorm
        h_for_norm = self.output_proj[1](h_for_norm)  # SiLU
        h = h_for_norm.transpose(1, 2)  # (B, T, model_channels)
        
        h = self.output_proj[2](h)  # Linear: (B, T, pose_dim)
        
        return h

class TextToPoseDiffusion:
    """文本到姿态序列的扩散过程"""
    def __init__(self, num_timesteps=1000, beta_schedule='cosine'):
        self.num_timesteps = num_timesteps
        
        if beta_schedule == 'cosine':
            betas = self.cosine_beta_schedule(num_timesteps)
        else:
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """余弦噪声调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start, t, noise=None):
        """前向过程：给x_0添加噪声得到x_t"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 确保所有张量在同一设备上
        device = x_start.device
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().to(device)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().to(device)
        
        # 调整维度匹配 (B, T, pose_dim)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, text_prompts=None, noise=None):
        """计算去噪损失"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, text_prompts)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, text_prompts=None):
        """从x_t采样x_{t-1}"""
        device = x.device
        betas_t = self.betas[t].to(device)[:, None, None]
        sqrt_one_minus_alphas_cumprod_t = ((1 - self.alphas_cumprod[t]) ** 0.5).to(device)[:, None, None]
        sqrt_recip_alphas_t = (1.0 / self.alphas[t].sqrt()).to(device)[:, None, None]
        
        # 预测的噪声
        predicted_noise = model(x, t, text_prompts)
        
        # 均值
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].to(device)[:, None, None]
            noise = torch.randn_like(x)
            return model_mean + (0.5 * posterior_variance_t).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, text_prompts=None):
        """完整的采样循环"""
        device = next(model.parameters()).device
        b = shape[0]
        
        # 从纯噪声开始
        pose_sequence = torch.randn(shape, device=device)
        pose_sequences = []
        
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            pose_sequence = self.p_sample(model, pose_sequence, t, i, text_prompts)
            pose_sequences.append(pose_sequence.cpu())
        
        return pose_sequences

    @torch.no_grad()
    def sample(self, model, text_prompts, num_frames=100, pose_dim=120):
        """生成姿态序列样本"""
        batch_size = len(text_prompts) if isinstance(text_prompts, list) else 1
        shape = (batch_size, num_frames, pose_dim)
        
        return self.p_sample_loop(model, shape, text_prompts)

# 保持向后兼容性，为原有的3D视频模型提供别名
UNet3D = PoseUNet1D
TextToVideoDiffusion = TextToPoseDiffusion 