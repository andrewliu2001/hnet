"""Check that every model is causal: changing byte j must not change logits at t < j."""
import torch
from model import ByteHNet
from chunkers import CHUNKERS

torch.manual_seed(0)
for name in CHUNKERS:
    m = ByteHNet(name, d_outer=32, d_inner=48, n_enc=1, n_main=1, n_dec=1, nh=2, window=16).double()
    # make routers non-trivial
    for p in m.chunker.parameters():
        p.data += 0.3 * torch.randn_like(p)
    m.eval()
    x = torch.randint(0, 256, (3, 64))
    l1, st = m(x)
    worst = 0.0
    for j in [5, 17, 40, 63]:
        x2 = x.clone(); x2[:, j] = (x2[:, j] + 7) % 256
        l2, _ = m(x2)
        worst = max(worst, (l1[:, :j] - l2[:, :j]).abs().max().item())
    # gradient reaches the router
    m.train()
    l, st = m(x)
    (l.logsumexp(-1).mean() + m.chunker.aux_loss(st)).backward()
    g = sum(p.grad.abs().sum().item() for p in m.chunker.parameters() if p.grad is not None)
    b = m.chunker.boundaries(st)
    print(f"{name:12s} max leak {worst:.2e}  router grad {g:.2e}  ratio {b.numel()/b.sum().item():.2f}")
    assert worst < 1e-9, name
print("ok")
