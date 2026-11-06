"""Figures and numbers for the paper, from research/dcbench/results/*.

  cd research/dcbench && python paper/make_figures.py
"""
import glob, json, os, collections, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# validated categorical slots (dataviz reference palette, light mode, all-pairs safe for 3)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3de"
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

STATIC = {"fixed", "fixed_N3.6", "fixed_ema", "fixed_main_in_norm1", "fixed_res_warmup3000",
          "fixed_lr_main_mult0.1", "anticip", "anticip_res_warmup3000"}


def load_reimpl():
    runs = []
    for f in sorted(glob.glob(f"{RES}/gpu[4-9]/*.json") + glob.glob(f"{RES}/gpu1[0-9]/*.json")):
        d = json.load(open(f))
        if d.get("done"):
            a = d["args"]
            runs.append(dict(arm=a["chunker"] + a.get("tag", ""), seed=a["seed"], impl="ours", **d["final"],
                             sweep=d.get("sweep", []), log=d["log"]))
    return runs


def load_official():
    runs = []
    for f in sorted(glob.glob(f"{RES}/official[a-z0-9]*/*.json")):
        if "smoke" in f:
            continue
        d = json.load(open(f))
        if d.get("done"):
            runs.append(dict(arm=d["args"]["variant"], seed=d["args"]["seed"], impl="official", **d["final"],
                             log=d["log"]))
    return runs


R, O = load_reimpl(), load_official()


def group(arm):
    if arm in ("hnet",):
        return "H-Net"
    if arm == "hnet_anticip":
        return "H-Net + anticipatory feature"
    if arm in STATIC or arm in ("fixed", "random"):
        return "static / non-learned boundaries"
    return None


STYLE = {"H-Net": (BLUE, "o"), "H-Net + anticipatory feature": (ORANGE, "^"),
         "static / non-learned boundaries": (AQUA, "s")}

# ---------------- Fig 1: bpb vs ctx_gain -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=False)
for ax, runs, title in [(axes[0], R, "Re-implementation (~150M params)"),
                        (axes[1], O, "Official H-Net code (~100M params)")]:
    for g, (c, m) in STYLE.items():
        pts = [(r["ctx_gain"], r["bpb"]) for r in runs if group(r["arm"]) == g]
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, y, s=22, c=c, marker=m, edgecolors="white", linewidths=0.6, zorder=3)
    ax.set_xlabel("ctx_gain (bpb)")
    ax.set_title(title, loc="left")
    ax.axvline(0.05, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
axes[0].set_ylabel("validation bits per byte")
for g, (c, m) in STYLE.items():
    nr = sum(group(r["arm"]) == g for r in R); no = sum(group(r["arm"]) == g for r in O)
    axes[0].scatter([], [], s=26, c=c, marker=m, label=g)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, -0.06))
fig.savefig(f"{OUT}/fig_scatter.pdf")
plt.close(fig)

# ---------------- Fig 2: per-seed distributions -----------------------------------------------
cols = [("ours", "hnet", "H-Net\n(ours)", BLUE, "o"), ("ours", "hnet_anticip", "H-Net+antic.\n(ours)", ORANGE, "^"),
        ("official", "hnet", "H-Net\n(official)", BLUE, "o"),
        ("official", "hnet_anticip", "H-Net+antic.\n(official)", ORANGE, "^"),
        ("official", "random", "random\n(official)", AQUA, "s"), ("official", "fixed", "fixed\n(official)", AQUA, "s")]
fig, ax = plt.subplots(figsize=(6.8, 2.5))
rng = np.random.default_rng(0)
xt, xl = [], []
k = 0
for impl, arm, lab, c, m in cols:
    vals = [r["bpb"] for r in (R if impl == "ours" else O) if r["arm"] == arm]
    if not vals:
        continue
    jit = rng.uniform(-0.12, 0.12, len(vals))
    ax.scatter(k + jit, vals, s=16, c=c, marker=m, edgecolors="white", linewidths=0.5, zorder=3)
    ax.plot([k - 0.25, k + 0.25], [np.median(vals)] * 2, color=INK, lw=1.2, zorder=4)
    xt.append(k); xl.append(f"{lab}\nn={len(vals)}")
    k += 1
ax.set_xticks(xt, xl)
ax.grid(axis="x", visible=False)
ax.set_ylabel("validation bits per byte")
ax.set_title("Per-seed outcomes (bar = median); same data, steps and evaluation for every run", loc="left")
fig.savefig(f"{OUT}/fig_seeds.pdf")
plt.close(fig)

# ---------------- Fig 3: inference-time compression sweep -------------------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.5))
pick = [("fixed", 1, "fixed (collapsed)", AQUA, "s"), ("anticip", 1, "anticip (collapsed)", AQUA, "D"),
        ("hnet", 1, "H-Net", BLUE, "o"), ("hnet_anticip", 5, "H-Net+antic.", ORANGE, "^")]
for arm, seed, lab, c, m in pick:
    r = next((r for r in R if r["arm"] == arm and r["seed"] == seed and r["sweep"]), None)
    if r is None:
        continue
    base = r["bpb"] if False else None
    sw = sorted(r["sweep"], key=lambda e: e["ratio"])
    ref = [e for e in sw if abs(e.get("dial", e.get("thr", 0)) - (1.0 if "dial" in e else 0.5)) < 1e-6][0]["bpb"]
    ax.plot([e["ratio"] for e in sw], [e["bpb"] - ref for e in sw], color=c, marker=m, ms=4.5, lw=1.6, label=lab)
ax.set_xscale("log")
ax.set_xticks([2, 3, 4, 6, 9], ["2", "3", "4", "6", "9"])
ax.set_xlabel("compression at inference (bytes / chunk)")
ax.set_ylabel("Δ bpb vs. training setting")
ax.set_title("Changing the chunking after training", loc="left")
ax.legend(frameon=False, loc="upper left")
fig.savefig(f"{OUT}/fig_dial.pdf")
plt.close(fig)

# ---------------- Fig 4: ctx_gain during training (official runs with per-eval ctx_gain) -------
traj = [r for r in O if sum("ctx_gain" in e for e in r["log"]) > 2]
if traj:
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for r in traj:
        g = group(r["arm"])
        c, m = STYLE[g]
        xs = [e["step"] for e in r["log"] if "ctx_gain" in e]
        ys = [e["ctx_gain"] for e in r["log"] if "ctx_gain" in e]
        ax.plot(xs, ys, color=c, lw=1.0, alpha=0.85)
    for g, (c, m) in STYLE.items():
        if any(group(r["arm"]) == g for r in traj):
            ax.plot([], [], color=c, lw=1.6, label=g)
    ax.set_xlabel("training step")
    ax.set_ylabel("ctx_gain (bpb)")
    ax.set_title("Main-network use during training (official code)", loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(f"{OUT}/fig_traj.pdf")
    plt.close(fig)

# ---------------- numbers for the text ----------------------------------------------------------
def summ(runs, arm):
    v = [r for r in runs if r["arm"] == arm]
    if not v:
        return None
    b = [r["bpb"] for r in v]; c = [r["ctx_gain"] for r in v]
    return dict(n=len(v), mean=st.mean(b), sd=st.stdev(b) if len(b) > 1 else 0, min=min(b), max=max(b),
                med=float(np.median(b)), ctx_mean=st.mean(c), ctx_min=min(c), ctx_max=max(c),
                collapsed=sum(x <= 0.05 for x in c), ratio=st.mean(r["ratio"] for r in v),
                time=st.mean(r["time"] for r in v), at_space=st.mean(r.get("at_space", float("nan")) for r in v))


nums = {"ours": {a: summ(R, a) for a in sorted({r["arm"] for r in R})},
        "official": {a: summ(O, a) for a in sorted({r["arm"] for r in O})}}
allb = [r["bpb"] for r in R + O]; allc = [r["ctx_gain"] for r in R + O]
nums["corr_all"] = float(np.corrcoef(allb, allc)[0, 1])
json.dump(nums, open(f"{OUT}/numbers.json", "w"), indent=1)
for impl in ("ours", "official"):
    for a, s in nums[impl].items():
        if s:
            print(f"{impl:8s} {a:24s} n={s['n']:2d} mean={s['mean']:.3f} sd={s['sd']:.3f} [{s['min']:.3f},{s['max']:.3f}] "
                  f"ctx {s['ctx_min']:.3f}-{s['ctx_max']:.3f} collapsed {s['collapsed']}/{s['n']} ratio {s['ratio']:.2f}")
print("corr(bpb, ctx_gain) over all runs:", round(nums["corr_all"], 3))
