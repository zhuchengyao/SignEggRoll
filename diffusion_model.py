import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Tuple, Union

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

class ResidualBlock(nn.Module):
    """残差块，用于U-Net的基本构建单元"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None]
        h = self.block2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv1d(channels, channels * 3, 1)
        self.to_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b, c, n = x.shape
        h = self.group_norm(x)
        qkv = self.to_qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 计算注意力
        scale = 1 / math.sqrt(c)
        attn = torch.bmm(q.transpose(1, 2), k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        h = torch.bmm(v, attn.transpose(1, 2))
        h = self.to_out(h)
        return h + x

class UNet1D(nn.Module):
    """1D U-Net网络，用于处理序列化的3D姿态数据"""
    def __init__(
        self, 
        in_channels=3,  # x, y, z坐标
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        num_keypoints=67
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_keypoints = num_keypoints

        # 时间嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 输入投影 - 将(batch, 67, 3)转换为(batch, model_channels, 67)
        self.input_proj = nn.Conv1d(in_channels, model_channels, 1)

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                )
                ch = mult * model_channels
                if ds in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(ch))
            if level != len(channel_mult) - 1:  # 不在最后一层下采样
                self.down_blocks.append(nn.Conv1d(ch, ch, 3, stride=2, padding=1))
                ds *= 2

        # 中间层
        self.middle = nn.ModuleList([
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch),
            ResidualBlock(ch, ch, time_embed_dim, dropout),
        ])

        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(ch + mult * model_channels, mult * model_channels, time_embed_dim, dropout)
                )
                ch = mult * model_channels
                if ds in attention_resolutions:
                    self.up_blocks.append(AttentionBlock(ch))
            if level != 0:
                self.up_blocks.append(nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1))
                ds //= 2

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, 1),
        )

    def forward(self, x, timesteps):
        """
        Args:
            x: (batch_size, num_keypoints, 3) - 输入的3D姿态数据
            timesteps: (batch_size,) - 时间步
        """
        # 调整输入维度: (batch, 67, 3) -> (batch, 3, 67)
        x = x.transpose(1, 2)
        
        # 时间嵌入
        t_emb = self.time_embed(timesteps)
        
        # 输入投影
        h = self.input_proj(x)
        
        # 保存跳跃连接
        hs = [h]
        
        # 下采样
        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
            hs.append(h)
        
        # 中间层
        for module in self.middle:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # 上采样
        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # 输出投影
        h = self.output_proj(h)
        
        # 调整输出维度: (batch, 3, 67) -> (batch, 67, 3)
        h = h.transpose(1, 2)
        
        return h

def cosine_beta_schedule(timesteps, s=0.008):
    """余弦噪声调度"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class GaussianDiffusion:
    """高斯扩散过程"""
    def __init__(self, num_timesteps=1000, beta_schedule='cosine'):
        self.num_timesteps = num_timesteps
        
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        else:
            # 线性调度
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 后验方差的计算
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """前向过程：给x_0添加噪声得到x_t"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt()
        
        # 调整维度匹配
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, noise=None):
        """计算去噪损失"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """从x_t采样x_{t-1}"""
        betas_t = self.betas[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = ((1 - self.alphas_cumprod[t]) ** 0.5)[:, None, None]
        sqrt_recip_alphas_t = (1.0 / self.alphas[t].sqrt())[:, None, None]
        
        # 预测的噪声
        predicted_noise = model(x, t)
        
        # 均值
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None]
            noise = torch.randn_like(x)
            return model_mean + (0.5 * posterior_variance_t).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """完整的采样循环"""
        device = next(model.parameters()).device
        b = shape[0]
        
        # 从纯噪声开始
        pose = torch.randn(shape, device=device)
        poses = []
        
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            pose = self.p_sample(model, pose, t, i)
            poses.append(pose.cpu())
        
        return poses

    @torch.no_grad()
    def sample(self, model, num_samples=1, num_keypoints=67):
        """生成样本"""
        return self.p_sample_loop(model, shape=(num_samples, num_keypoints, 3)) 