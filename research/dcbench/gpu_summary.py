"""Aggregate the valid GPU runs (results/gpu4 onward; waves 1-3 used a config in which the
main network is bypassed and are excluded) into a markdown table, sorted by mean bpb.
A run counts as "alive" if replacing the main network's output by its mean costs > 0.05 bpb."""
import glob, json, collections, statistics as st

ALIVE = 0.05
g = collections.defaultdict(list)
for f in sorted(glob.glob("results/gpu[4-9]/*.json")):
    d = json.load(open(f))
    if not d.get("done"):
        continue
    a, fin = d["args"], d["final"]
    g[a["chunker"] + a.get("tag", "")].append((a["seed"], fin))

print("| arm | seeds | val bpb (per seed) | mean | ctx_gain (per seed) | main alive | compression | boundaries on space/punct | wall s |")
print("|---|---|---|---|---|---|---|---|---|")
for k, v in sorted(g.items(), key=lambda kv: st.mean(x["bpb"] for _, x in kv[1])):
    v = sorted(v, key=lambda sv: sv[0])
    b = [x["bpb"] for _, x in v]
    c = [x.get("ctx_gain", float("nan")) for _, x in v]
    alive = sum(ci > ALIVE for ci in c)
    print(f"| `{k}` | {','.join(str(s) for s, _ in v)} | {', '.join(f'{x:.3f}' for x in b)} | **{st.mean(b):.3f}** | "
          f"{', '.join(f'{x:.3f}' for x in c)} | {alive}/{len(v)} | {st.mean(x['ratio'] for _, x in v):.2f} | "
          f"{st.mean(x.get('at_space', float('nan')) for _, x in v):.2f} | {st.mean(x['time'] for _, x in v):.0f} |")

# alive-only means
print()
for k, v in sorted(g.items()):
    al = [x["bpb"] for _, x in v if x.get("ctx_gain", 0) > ALIVE]
    if al and len(al) < len(v):
        print(f"* `{k}` mean over alive seeds only: {st.mean(al):.3f} ({len(al)}/{len(v)})")
