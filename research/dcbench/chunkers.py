"""
Pure-PyTorch dynamic chunking modules for small-scale CPU experiments.

All chunkers share one interface:

    z, zmask, st = chunker.down(h)          # h: (B, L, D) encoder output
    y = main_network(z, zmask)              # (B, M, D)
    u = chunker.up(y, st)                   # (B, L, D) aligned with bytes
    aux = chunker.aux_loss(st)              # scalar

Causality contract: u[:, t] may only depend on h[:, :t+1].
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def ste_one(x):
    """Forward: 1. Backward: identity (H-Net's STE on the confidence)."""
    return 1.0 + x - x.detach()


def gather_chunks(h, b):
    """Select positions where b is True, left-packed, padded. Returns z (B,M,D), mask (B,M)."""
    B, L, D = h.shape
    n = b.sum(1)
    M = int(n.max())
    idx = torch.arange(L, device=h.device)[None].expand(B, L) + (~b).long() * L
    order = idx.argsort(1)[:, :M]
    z = h.gather(1, order[..., None].expand(-1, -1, D))
    mask = torch.arange(M, device=h.device)[None] < n[:, None]
    return z, mask


# ---------------------------------------------------------------------------
# 0. Fixed-stride chunking (baseline)
# ---------------------------------------------------------------------------
class FixedChunker(nn.Module):
    def __init__(self, d, N=4, **kw):
        super().__init__()
        self.N = N

    def down(self, h):
        B, L, D = h.shape
        t = torch.arange(L, device=h.device, dtype=torch.float64)
        # works for fractional strides too (e.g. N=3.4 to match H-Net's realised ratio)
        b = (torch.floor(t / self.N) != torch.floor((t - 1) / self.N))[None].expand(B, L)
        z, mask = gather_chunks(h, b)
        return z, mask, dict(b=b)

    def up(self, y, st):
        idx = st["b"].long().cumsum(1) - 1
        return y.gather(1, idx[..., None].expand(-1, -1, y.shape[-1]))

    def aux_loss(self, st):
        return torch.zeros(())

    def boundaries(self, st):
        return st["b"]


# ---------------------------------------------------------------------------
# 1. H-Net dynamic chunking (faithful re-implementation of hnet/modules/dc.py)
# ---------------------------------------------------------------------------
class HNetChunker(nn.Module):
    """
    Routing: p_t = (1 - cos(Wq h_{t-1}, Wk h_t)) / 2, p_0 = 1, b_t = p_t >= 0.5.
    Down:    select h_t where b_t.
    Up:      EMA smoothing over chunk outputs with rate p, repeat to bytes,
             multiply by STE(confidence).
    Loss:    H-Net ratio loss with target N.
    `ema=False` / `ste=False` turn individual pieces off (ablations).
    """

    def __init__(self, d, N=4, ema=True, ste=True, **kw):
        super().__init__()
        self.N, self.ema, self.ste = N, ema, ste
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.q.weight.copy_(torch.eye(d))
            self.k.weight.copy_(torch.eye(d))
        self.q.weight._no_reinit = True
        self.k.weight._no_reinit = True

    def down(self, h):
        cos = F.cosine_similarity(self.q(h[:, :-1]), self.k(h[:, 1:]), dim=-1)
        p = ((1 - cos) / 2).clamp(0, 1)
        p = F.pad(p, (1, 0), value=1.0)
        b = p >= 0.5
        z, mask = gather_chunks(h, b)
        P, _ = gather_chunks(p[..., None], b)
        return z, mask, dict(p=p, b=b, P=P[..., 0])

    def up(self, y, st):
        p, b, P = st["p"], st["b"], st["P"]
        if self.ema:
            # ybar_k = P_k y_k + (1 - P_k) ybar_{k-1}, done in closed form.
            P = P.float().clamp(1e-4, 1 - 1e-4)
            A = torch.log1p(-P).cumsum(1)                       # (B, M)
            W = A[:, :, None] - A[:, None, :]                   # A_k - A_j
            M = P.shape[1]
            tri = torch.ones(M, M, dtype=torch.bool, device=y.device).tril()
            W = torch.where(tri, W, torch.full_like(W, -float("inf"))).exp() * P[:, None, :]
            y = torch.einsum("bkj,bjd->bkd", W.to(y.dtype), y)
        idx = b.long().cumsum(1) - 1
        u = y.gather(1, idx[..., None].expand(-1, -1, y.shape[-1]))
        if self.ste:
            conf = torch.where(b, p, 1 - p)
            u = u * ste_one(conf)[..., None]
        return u

    def aux_loss(self, st):
        p, b, N = st["p"], st["b"], self.N
        F_ = b.float().mean()
        G = p.float().mean()
        return N / (N - 1) * ((N - 1) * F_ * G + (1 - F_) * (1 - G))

    def boundaries(self, st):
        return st["b"]


# ---------------------------------------------------------------------------
# 2. NEW: Integrate-and-Fire chunking (IF-DC)
# ---------------------------------------------------------------------------
class IFChunker(nn.Module):
    """
    Each byte emits a scalar "weight" a_t in (0,1). Weights are integrated
    (C_t = sum_{s<=t} a_s); a chunk is emitted every time C crosses an integer.
    The chunk vector is the a-weighted *average* of the bytes it integrated:
    the byte that straddles a crossing is split between the two chunks.

        z_k = sum_t w_{t,k} h_t,   w_{t,k} = |[C_{t-1}, C_t] ∩ [k, k+1]|

    Properties:
      * the number of chunks is exactly floor(sum_t a_t), so the compression
        ratio is controlled *directly* by mean(a) (simple quadratic loss, no
        mismatch between "probability" and "decision" as in H-Net);
      * the pooling weights are an exact, differentiable function of a, so the
        router gets gradient through the downsampling path itself — no STE and
        no EMA smoothing are required;
      * each chunk summarises all its bytes (pooling), not just one byte;
      * fully causal: chunk k is complete at the byte where it fires, and byte
        t only reads chunks fired at <= t.
    Upsampling: byte t uses the output of the most recent fired chunk.
    """

    def __init__(self, d, N=4, lam=10.0, ste=False, cos=False, last=False, frozen=False, **kw):
        super().__init__()
        self.N, self.lam, self.ste, self.cos, self.last = N, lam, ste, cos, last
        self.frozen = frozen  # ablation: constant a = 1/N (fixed stride) but keep pooling-gradient path
        if cos:
            # optional H-Net-style neighbour-dissimilarity feature for the router
            self.q = nn.Linear(d, d, bias=False)
            self.k = nn.Linear(d, d, bias=False)
            with torch.no_grad():
                self.q.weight.copy_(torch.eye(d))
                self.k.weight.copy_(torch.eye(d))
            self.q.weight._no_reinit = True
            self.k.weight._no_reinit = True
            self.g = nn.Parameter(torch.tensor(4.0))
        self.w = nn.Linear(d, 1)
        self.w.weight._no_reinit = True
        with torch.no_grad():
            self.w.weight.zero_()
            self.w.bias.fill_(-math.log(N - 1))   # sigmoid(b) = 1/N at init
        self.norm = nn.LayerNorm(d)

    def down(self, h):
        B, L, D = h.shape
        logit = self.w(self.norm(h))[..., 0]
        if self.cos:
            cos = F.cosine_similarity(self.q(h[:, :-1]), self.k(h[:, 1:]), dim=-1)
            dis = F.pad((1 - cos) / 2, (1, 0), value=0.0)
            logit = logit + self.g * (dis - 0.5)
        a = torch.sigmoid(logit)
        if self.frozen:
            a = torch.full_like(a, 1.0 / self.N)
        a = torch.cat([torch.ones_like(a[:, :1]), a[:, 1:]], 1)  # byte 0 forms chunk 0
        C = a.cumsum(1)
        Cp = C - a
        k0 = Cp.detach().floor()
        w0 = torch.minimum(C, k0 + 1) - Cp
        w1 = (C - (k0 + 1)).clamp(min=0)
        nfire = C.detach().floor().long()             # chunks complete at byte t
        M = int(nfire[:, -1].max())
        z = h.new_zeros(B, M + 1, D)
        k0l = k0.long()
        z.scatter_add_(1, k0l[..., None].expand(-1, -1, D), (w0[..., None] * h).to(h.dtype))
        z.scatter_add_(1, (k0l + 1).clamp(max=M)[..., None].expand(-1, -1, D),
                       (w1[..., None] * h).to(h.dtype))
        z = z[:, :M]
        mask = torch.arange(M, device=h.device)[None] < nfire[:, -1:]
        b = torch.zeros_like(nfire, dtype=torch.bool)
        b[:, 1:] = nfire[:, 1:] > nfire[:, :-1]
        b[:, 0] = True
        if self.last:
            # forward: the state of the byte where the chunk fires (selection, like H-Net);
            # backward: gradient w.r.t. the router through the pooling weights.
            zl, _ = gather_chunks(h, b)
            zl = zl[:, :M]
            z = zl + (z - z.detach())
        return z, mask, dict(a=a, nfire=nfire, b=b, frac=C - C.detach().floor())

    def up(self, y, st):
        idx = (st["nfire"] - 1).clamp(min=0)
        u = y.gather(1, idx[..., None].expand(-1, -1, y.shape[-1]))
        if self.ste:
            # optional: confidence = how "decisively" the byte fired / did not
            u = u * ste_one(st["a"])[..., None]
        return u

    def aux_loss(self, st):
        a = st["a"][:, 1:]
        return self.lam * (a.mean(1) * self.N - 1).pow(2).mean()

    def boundaries(self, st):
        return st["b"]


CHUNKERS = {
    "fixed": lambda d, N: FixedChunker(d, N),
    "hnet": lambda d, N: HNetChunker(d, N),
    "hnet_noema": lambda d, N: HNetChunker(d, N, ema=False),
    "hnet_noste": lambda d, N: HNetChunker(d, N, ste=False),
    "if": lambda d, N: IFChunker(d, N),
    "if_ste": lambda d, N: IFChunker(d, N, ste=True),
    "if_cos": lambda d, N: IFChunker(d, N, cos=True),
    "if_last": lambda d, N: IFChunker(d, N, last=True),
    "if_last_frozen": lambda d, N: IFChunker(d, N, last=True, frozen=True),
}
