"""Tabulate results/*/*.json -> markdown table (final eval of each run)."""
import glob, json, os, collections, statistics as st

rows = collections.defaultdict(list)
for f in sorted(glob.glob("results/*/*.json")):
    r = json.load(open(f))
    a, fin = r["args"], r["final"]
    split = "shifted (round1)" if "round1" in f else "interleaved"
    rows[(split, a["chunker"] + a.get("tag", ""))].append((a["seed"], fin))

print("| split | chunker | seeds | val bpb (each) | mean val bpb | compression L/M | word-boundary precision | wall time (s) |")
print("|---|---|---|---|---|---|---|---|")
for (split, c), v in sorted(rows.items()):
    b = [x["bpb"] for _, x in v]
    print(f"| {split} | {c} | {','.join(str(s) for s,_ in v)} | {', '.join(f'{x:.3f}' for x in b)} | "
          f"{st.mean(b):.3f} | {st.mean(x['ratio'] for _,x in v):.2f} | "
          f"{st.mean(x['word_prec'] for _,x in v):.2f} | {st.mean(x['time'] for _,x in v):.0f} |")

# seed-paired table on the interleaved split
arms = ["fixed", "fixed_N3.4", "hnet", "hnet_noema", "if", "if_last", "if_last_frozen"]
by = {c: {s: x["bpb"] for s, x in rows.get(("interleaved", c), [])} for c in arms}
seeds = sorted({s for c in arms for s in by[c]})
print()
print("| seed | " + " | ".join(arms) + " |")
print("|---|" + "---|" * len(arms))
for s in seeds:
    print(f"| {s} | " + " | ".join(f"{by[c][s]:.3f}" if s in by[c] else "–" for c in arms) + " |")
common = [s for s in seeds if s in by["fixed"] and s in by["if_last"]]
if len(common) > 1:
    d = [by["if_last"][s] - by["fixed"][s] for s in common]
    m, sd = st.mean(d), st.stdev(d)
    print(f"\nif_last − fixed over seeds {common}: mean {m:+.4f}, sd {sd:.4f}, t = {m / (sd / len(d) ** .5):.2f}")
common = [s for s in seeds if s in by["hnet"] and s in by["if_last"]]
if len(common) > 1:
    d = [by["if_last"][s] - by["hnet"][s] for s in common]
    m, sd = st.mean(d), st.stdev(d)
    print(f"if_last − hnet  over seeds {common}: mean {m:+.4f}, sd {sd:.4f}, t = {m / (sd / len(d) ** .5):.2f}")
