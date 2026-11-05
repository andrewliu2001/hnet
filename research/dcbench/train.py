import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from model import ByteHNet

p = argparse.ArgumentParser()
p.add_argument("--chunker", default="hnet")
p.add_argument("--data", required=True)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--L", type=int, default=512)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--N", type=float, default=4)
p.add_argument("--aux", type=float, default=0.03)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--eval_every", type=int, default=500)
p.add_argument("--out", default="results")
p.add_argument("--tag", default="")
args = p.parse_args()

torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

raw = np.frombuffer(open(args.data, "rb").read(), dtype=np.uint8)
n_val = len(raw) // 20
train, val = raw[:-n_val], raw[-n_val:]
space = np.zeros(256, bool)
for c in b" \n\t.,;:!?\"'()-":
    space[c] = True

model = ByteHNet(args.chunker, N=args.N)
nparams = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
sched = lambda s: min(1, s / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))


def batch(arr, bs, rng):
    ix = rng.integers(0, len(arr) - args.L - 1, bs)
    x = np.stack([arr[i:i + args.L + 1] for i in ix]).astype(np.int64)
    x = torch.from_numpy(x)
    return x[:, :-1], x[:, 1:]


@torch.no_grad()
def evaluate(n_batches=40):
    model.eval()
    rng = np.random.default_rng(1234)
    tot, cnt, nb, nc, word = 0.0, 0, 0, 0, []
    for _ in range(n_batches):
        x, y = batch(val, 16, rng)
        logits, st = model(x)
        tot += F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum").item()
        cnt += y.numel()
        b = model.chunker.boundaries(st)
        nb += b.sum().item()
        # boundary precision: does the chunk start right after a space/punct byte?
        prev = torch.from_numpy(space[x[:, :-1].numpy()])
        bb = b[:, 1:]
        word.append((prev & bb).sum().item() / max(1, bb.sum().item()))
        nc += prev.sum().item()
    model.train()
    return dict(bpb=tot / cnt / math.log(2), ratio=cnt / nb, word_prec=float(np.mean(word)))


rng = np.random.default_rng(args.seed)
log = []
t0 = time.time()
name = f"{args.chunker}{args.tag}_s{args.seed}"
os.makedirs(args.out, exist_ok=True)
for step in range(1, args.steps + 1):
    for g in opt.param_groups:
        g["lr"] = args.lr * sched(step)
    x, y = batch(train, args.bs, rng)
    logits, st = model(x)
    ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
    aux = model.chunker.aux_loss(st)
    loss = ce + args.aux * aux
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        b = model.chunker.boundaries(st)
        print(f"{name} step {step} ce {ce.item()/math.log(2):.3f}bpb aux {aux.item():.3f} "
              f"ratio {b.numel()/b.sum().item():.2f} {time.time()-t0:.0f}s", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        ev = evaluate()
        ev.update(step=step, time=time.time() - t0)
        log.append(ev)
        print(f"{name} EVAL {ev}", flush=True)

with open(os.path.join(args.out, name + ".json"), "w") as f:
    json.dump(dict(args=vars(args), params=nparams, log=log, final=log[-1]), f, indent=1)
