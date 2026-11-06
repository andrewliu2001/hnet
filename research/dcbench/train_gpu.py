"""
GPU training for dcbench at larger scale (FineWeb-Edu bytes, ~150M params).

  python train_gpu.py --chunker if_last --train data/train.bin --val data/val.bin --steps 20000 --seed 1

Writes <out>/<name>.json after every eval (so partial results survive preemption).
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from model import ByteHNet

p = argparse.ArgumentParser()
p.add_argument("--chunker", default="hnet")
p.add_argument("--train", required=True)
p.add_argument("--val", required=True)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--L", type=int, default=2048)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--warmup", type=int, default=1000)
p.add_argument("--N", type=float, default=4)
p.add_argument("--aux", type=float, default=0.03)
p.add_argument("--d_outer", type=int, default=512)
p.add_argument("--d_inner", type=int, default=768)
p.add_argument("--n_enc", type=int, default=4)
p.add_argument("--n_main", type=int, default=12)
p.add_argument("--n_dec", type=int, default=4)
p.add_argument("--nh", type=int, default=8)
p.add_argument("--nh_inner", type=int, default=12)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval_every", type=int, default=2000)
p.add_argument("--eval_batches", type=int, default=100)
p.add_argument("--out", default="results/gpu")
p.add_argument("--tag", default="")
p.add_argument("--probe", type=int, default=0, help="only time this many steps, print s/step, exit")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"  # cpu only for smoke tests
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

train = np.memmap(args.train, dtype=np.uint8, mode="r")
val = np.memmap(args.val, dtype=np.uint8, mode="r")
space = torch.zeros(256, dtype=torch.bool, device=dev)
for c in b" \n\t.,;:!?\"'()-\x00":
    space[c] = True

model = ByteHNet(args.chunker, d_outer=args.d_outer, d_inner=args.d_inner, n_enc=args.n_enc,
                 n_main=args.n_main, n_dec=args.n_dec, nh=args.nh, nh_inner=args.nh_inner,
                 N=args.N, window=None).to(dev)
nparams = sum(p.numel() for p in model.parameters())
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.95), fused=dev == "cuda")


def lr_at(s):
    if s < args.warmup:
        return args.lr * s / args.warmup
    t = (s - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def batch(arr, bs, rng):
    ix = rng.integers(0, len(arr) - args.L - 1, bs)
    x = np.stack([np.asarray(arr[i:i + args.L + 1]) for i in ix]).astype(np.int64)
    x = torch.from_numpy(x)
    x = x.pin_memory().to(dev, non_blocking=True) if dev == "cuda" else x
    return x[:, :-1], x[:, 1:]


@torch.no_grad()
def evaluate(arr, n_batches):
    model.eval()
    rng = np.random.default_rng(1234)
    tot, cnt, nb, hits, starts, at_sp = 0.0, 0, 0, 0, 0, 0
    for _ in range(n_batches):
        x, y = batch(arr, args.bs, rng)
        with torch.autocast(dev, dtype=torch.bfloat16):
            logits, st = model(x)
        tot += F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1), reduction="sum").item()
        cnt += y.numel()
        b = model.chunker.boundaries(st)
        nb += b.sum().item()
        prev = space[x[:, :-1]]
        hits += (prev & b[:, 1:]).sum().item()
        starts += b[:, 1:].sum().item()
        at_sp += (space[x[:, 1:]] & b[:, 1:]).sum().item()
    model.train()
    return dict(bpb=tot / cnt / math.log(2), ratio=cnt / nb, word_prec=hits / max(1, starts),
                at_space=at_sp / max(1, starts))


rng = np.random.default_rng(args.seed)
name = f"{args.chunker}{args.tag}_s{args.seed}"
os.makedirs(args.out, exist_ok=True)
log, t0 = [], time.time()
for step in range(1, args.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = batch(train, args.bs, rng)
    with torch.autocast(dev, dtype=torch.bfloat16):
        logits, st = model(x)
        ce = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))
        aux = model.chunker.aux_loss(st).to(ce.device)
    loss = ce + args.aux * aux
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if args.probe and step == 20:
        tp = time.time()
    if args.probe and step == args.probe:
        if dev == "cuda":
            torch.cuda.synchronize()
        print(f"PROBE {args.chunker} {(time.time() - tp) / (args.probe - 20):.4f} s/step "
              f"params {nparams}", flush=True)
        raise SystemExit
    if step % 100 == 0:
        b = model.chunker.boundaries(st)
        print(f"{name} step {step} ce {ce.item() / math.log(2):.3f}bpb aux {aux.item():.3f} "
              f"ratio {b.numel() / b.sum().item():.2f} {time.time() - t0:.0f}s", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        ev = evaluate(val, args.eval_batches if step == args.steps else args.eval_batches // 4)
        ev.update(step=step, time=time.time() - t0, train_bpb=evaluate(train, 10)["bpb"])
        log.append(ev)
        print(f"{name} EVAL {ev}", flush=True)
        sweep = []
        if step == args.steps:
            # inference-time compute dial: same weights, different compression
            ch = model.chunker
            if hasattr(ch, "thr"):
                knobs = [("thr", v) for v in (0.3, 0.4, 0.5, 0.6, 0.7)]
            else:
                knobs = [("dial", v) for v in (0.67, 0.8, 1.0, 1.25, 1.5)]
            for k, v in knobs:
                old = getattr(ch, k); setattr(ch, k, v)
                ev2 = evaluate(val, args.eval_batches // 2); ev2[k] = v
                sweep.append(ev2); setattr(ch, k, old)
                print(f"{name} SWEEP {ev2}", flush=True)
        with open(os.path.join(args.out, name + ".json"), "w") as f:
            json.dump(dict(args=vars(args), params=nparams, log=log, final=log[-1], sweep=sweep,
                           done=step == args.steps), f, indent=1)
