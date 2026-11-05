import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from chunkers import CHUNKERS


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope(x, base=10000.0):
    # x: (B, H, L, Dh)
    L, Dh = x.shape[-2], x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, Dh, 2, device=x.device).float() / Dh)
    t = torch.arange(L, device=x.device).float()
    f = torch.outer(t, inv)
    cos, sin = f.cos(), f.sin()
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, nh, window=None):
        super().__init__()
        self.nh, self.window = nh, window
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, 8 * d, bias=False)   # SwiGLU, hidden 4d
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.nh, D // self.nh).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        i = torch.arange(L, device=x.device)
        am = i[:, None] >= i[None, :]
        if self.window is not None:
            am = am & (i[:, None] - i[None, :] < self.window)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        x = x + self.out_proj(a.transpose(1, 2).reshape(B, L, D))
        g, u = self.fc1(self.n2(x)).chunk(2, -1)
        return x + self.fc2(F.silu(g) * u)


class Stack(nn.Module):
    def __init__(self, d, nh, n, window=None):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, nh, window) for _ in range(n))
        self.norm = RMSNorm(d)

    def forward(self, x, mask=None):
        for b in self.blocks:
            x = b(x, mask)
        return self.norm(x)


class ByteHNet(nn.Module):
    """1-stage H-Net style byte LM with a pluggable chunker."""

    def __init__(self, chunker="hnet", d_outer=192, d_inner=320, n_enc=2, n_main=6, n_dec=2,
                 nh=4, N=4, window=128):
        super().__init__()
        self.emb = nn.Embedding(256, d_outer)
        self.enc = Stack(d_outer, nh, n_enc, window)
        self.chunker = CHUNKERS[chunker](d_outer, N)
        self.up_dim = nn.Linear(d_outer, d_inner, bias=False)
        self.main = Stack(d_inner, nh, n_main)
        self.down_dim = nn.Linear(d_inner, d_outer, bias=False)
        self.residual_proj = nn.Linear(d_outer, d_outer)
        self.dec = Stack(d_outer, nh, n_dec, window)
        self.head = nn.Linear(d_outer, 256, bias=False)
        self.apply(self._init)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.enc(self.emb(x))
        res = self.residual_proj(h)
        z, zmask, st = self.chunker.down(h)
        y = self.down_dim(self.main(self.up_dim(z), zmask))
        u = self.chunker.up(y, st)
        out = self.dec(u + res)
        return self.head(out), st
