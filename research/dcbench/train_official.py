"""
Main-network-collapse check on the OFFICIAL H-Net implementation (hnet/ package: Mamba-2
encoder/decoder, triton RMSNorm, flash-attention main network, official lr multipliers and
load-balancing loss), trained exactly like train_gpu.py (same data, batch, steps, eval).

  python train_official.py --variant hnet  --train data/train.bin --val data/val.bin --seed 1
  python train_official.py --variant fixed ...   # same architecture, routing forced to every N-th byte

At the end it reports ctx_gain = val-bpb increase when the main network's output is replaced by
its per-sequence mean over chunks (same diagnostic as train_gpu.py).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from hnet.models.config_hnet import HNetConfig, AttnConfig, SSMConfig
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.modules.dc import RoutingModuleOutput
from hnet.utils.train import load_balancing_loss, group_params

p = argparse.ArgumentParser()
p.add_argument("--variant", default="hnet", choices=["hnet", "fixed"])
p.add_argument("--train", required=True)
p.add_argument("--val", required=True)
p.add_argument("--steps", type=int, default=9000)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--L", type=int, default=2048)
p.add_argument("--lr", type=float, default=5e-4, help="base lr; stage multipliers applied on top")
p.add_argument("--lr_mult", type=str, default="2.0,1.0", help="official per-stage lr multipliers (outer,inner)")
p.add_argument("--warmup", type=int, default=1000)
p.add_argument("--N", type=int, default=4)
p.add_argument("--aux", type=float, default=0.03)
p.add_argument("--layout", default="m4,T12,m4")
p.add_argument("--d_model", default="512,768")
p.add_argument("--d_ff", default="0,2048")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval_every", type=int, default=2000)
p.add_argument("--eval_batches", type=int, default=100)
p.add_argument("--out", default="results/official")
p.add_argument("--probe", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True

enc, main, dec = args.layout.split(",")
dm = [int(v) for v in args.d_model.split(",")]
cfg = HNetConfig(arch_layout=[enc, [main], dec], d_model=dm, d_intermediate=[int(v) for v in args.d_ff.split(",")],
                 vocab_size=256, ssm_cfg=SSMConfig(d_conv=4, expand=2, d_state=128, chunk_size=256),
                 attn_cfg=AttnConfig(num_heads=[dm[0] // 64, dm[1] // 64], rotary_emb_dim=[32, 32],
                                     window_size=[1023, -1]),
                 tie_embeddings=False)
model = HNetForCausalLM(cfg, device=dev, dtype=torch.float32)
model.init_weights()
model.apply_lr_multiplier([float(v) for v in args.lr_mult.split(",")])
groups = group_params(model)
for g in groups:
    g["mult"] = g.pop("lr_multiplier", 1.0)
    g.setdefault("weight_decay", 0.1)
opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=True)
nparams = sum(q.numel() for q in model.parameters())
bb = model.backbone

# ---- variant: fixed-stride routing in the official architecture ---------------------------
if args.variant == "fixed":
    rm = bb.routing_module

    def fixed_routing(hidden_states, cu_seqlens=None, mask=None, inference_params=None):
        T = hidden_states.shape[0]                         # packed (T, D)
        pos = torch.arange(T, device=hidden_states.device) % args.L
        b = pos % args.N == 0
        pr = b.float()
        return RoutingModuleOutput(boundary_prob=torch.stack([1 - pr, pr], -1), boundary_mask=b,
                                   selected_probs=torch.ones(T, 1, device=hidden_states.device))
    rm.forward = fixed_routing

# ---- ctx_gain hook: replace the main network's output by its per-sequence mean -------------
main_fwd = bb.main_network.forward
MEAN_REPLACE = [False]


def main_wrapper(hidden_states, cu_seqlens=None, max_seqlen=None, mask=None, inference_params=None, **kw):
    out, prev = main_fwd(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, mask=mask,
                         inference_params=inference_params, **kw)
    if MEAN_REPLACE[0]:
        seg = torch.repeat_interleave(torch.arange(len(cu_seqlens) - 1, device=out.device),
                                      (cu_seqlens[1:] - cu_seqlens[:-1]).long())
        s = torch.zeros(len(cu_seqlens) - 1, out.shape[-1], device=out.device, dtype=torch.float32)
        s.index_add_(0, seg, out.float())
        cnt = (cu_seqlens[1:] - cu_seqlens[:-1]).float().clamp(min=1)[:, None]
        out = (s / cnt)[seg].to(out.dtype)
    return out, prev
bb.main_network.forward = main_wrapper

train = np.memmap(args.train, dtype=np.uint8, mode="r")
val = np.memmap(args.val, dtype=np.uint8, mode="r")


def lr_at(s):
    if s < args.warmup:
        return args.lr * s / args.warmup
    t = (s - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def batch(arr, bs, rng):
    ix = rng.integers(0, len(arr) - args.L - 1, bs)
    x = torch.from_numpy(np.stack([np.asarray(arr[i:i + args.L + 1]) for i in ix]).astype(np.int64))
    x = x.pin_memory().to(dev, non_blocking=True)
    return x[:, :-1], x[:, 1:]


def run(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(x)
    return o.logits.float(), o.bpred_output


@torch.no_grad()
def evaluate(arr, n):
    model.eval()
    rng = np.random.default_rng(1234)
    tot, cnt, nb = 0.0, 0, 0
    for _ in range(n):
        x, y = batch(arr, args.bs, rng)
        logits, bp = run(x)
        tot += F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum").item()
        cnt += y.numel()
        nb += bp[0].boundary_mask.sum().item()
    model.train()
    return dict(bpb=tot / cnt / math.log(2), ratio=cnt / max(1, nb))


rng = np.random.default_rng(args.seed)
name = f"official_{args.variant}_s{args.seed}"
os.makedirs(args.out, exist_ok=True)
log, t0 = [], time.time()
for step in range(1, args.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step) * g["mult"]
    x, y = batch(train, args.bs, rng)
    logits, bp = run(x)
    ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
    aux = sum(load_balancing_loss(o, N=args.N) for o in bp) if args.variant == "hnet" else torch.zeros((), device=dev)
    loss = ce + args.aux * aux
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if args.probe and step == 20:
        torch.cuda.synchronize(); tp = time.time()
    if args.probe and step == args.probe:
        torch.cuda.synchronize()
        print(f"PROBE official_{args.variant} {(time.time() - tp) / (args.probe - 20):.4f} s/step params {nparams}", flush=True)
        raise SystemExit
    if step % 100 == 0:
        r = bp[0].boundary_mask.numel() / max(1, bp[0].boundary_mask.sum().item())
        print(f"{name} step {step} ce {ce.item() / math.log(2):.3f}bpb aux {float(aux):.3f} ratio {r:.2f} "
              f"{time.time() - t0:.0f}s", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        last = step == args.steps
        ev = evaluate(val, args.eval_batches if last else args.eval_batches // 4)
        ev.update(step=step, time=time.time() - t0)
        if last:
            MEAN_REPLACE[0] = True
            ev["bpb_main_meanreplaced"] = evaluate(val, args.eval_batches // 2)["bpb"]
            MEAN_REPLACE[0] = False
            ev["bpb_half"] = evaluate(val, args.eval_batches // 2)["bpb"]
            ev["ctx_gain"] = ev["bpb_main_meanreplaced"] - ev["bpb_half"]
            print(f"{name} CTX {ev['ctx_gain']:.4f} bpb", flush=True)
        log.append(ev)
        print(f"{name} EVAL {ev}", flush=True)
        with open(os.path.join(args.out, name + ".json"), "w") as f:
            json.dump(dict(args=vars(args), params=nparams, log=log, final=log[-1], done=last), f, indent=1)
