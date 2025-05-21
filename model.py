# model.py

import torch
import torch.nn as nn

class FrameEncoderCNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim=1024, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act  = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)              # -> [B, feature_dim, S]
        x = self.conv(x)                  # -> [B, hidden_dim, S]
        x = x.transpose(1, 2)             # -> [B, S, hidden_dim]
        x = self.norm(x)
        return self.act(x)                # -> [B, S, hidden_dim]


class PoseSeqModel(nn.Module):
    def __init__(self, feature_dim, T_max, hidden_dim=1024,
                 n_layers=8, n_heads=16, ff_dim=2048):
        super().__init__()
        self.S = T_max + 2   # +2 for start/end
        self.hidden_dim = hidden_dim

        self.encoder = FrameEncoderCNN(feature_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.S, hidden_dim))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        B, S, _ = x.shape
        enc = self.encoder(x) + self.pos_emb[:, :S, :]
        tgt = enc.transpose(0, 1)
        memory = tgt
        dec_out = self.decoder(tgt, memory)
        dec_out = dec_out.transpose(0, 1)
        y = self.out_proj(dec_out)
        return y


class AutoRegressivePoseModel(PoseSeqModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _causal_mask(self, size):
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    def forward_train(self, x, lengths):
        B, S, D = x.shape
        enc = self.encoder(x) + self.pos_emb[:, :S, :]
        tgt = enc.transpose(0, 1)
        memory = tgt

        # 因果掩码
        mask_causal = self._causal_mask(S).to(x.device)
        # padding 掩码：True 表示该位置是 padding
        idxs = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        padding_mask = idxs >= (lengths.unsqueeze(1) + 1)

        dec_out = self.decoder(
            tgt,
            memory,
            tgt_mask=mask_causal,
            tgt_key_padding_mask=padding_mask
        )
        out = self.out_proj(dec_out.transpose(0, 1))
        return out

    @torch.no_grad()
    def generate(self, init_frames, max_gen_steps):
        device = init_frames.device
        B, k, D = init_frames.shape
        start = torch.zeros(B, 1, D, device=device)
        seq = torch.cat([start, init_frames], dim=1)

        for step in range(max_gen_steps):
            S = seq.shape[1]
            enc = self.encoder(seq) + self.pos_emb[:, :S, :]
            tgt = enc.transpose(0, 1)
            memory = tgt
            mask = self._causal_mask(S).to(device)
            dec = self.decoder(tgt, memory, tgt_mask=mask)
            out = self.out_proj(dec.transpose(0, 1))
            next_frame = out[:, -1:, :]
            seq = torch.cat([seq, next_frame], dim=1)

        return seq
