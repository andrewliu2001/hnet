"""Tabulate results/*/*.json -> markdown table (final eval of each run)."""
import glob, json, os, collections, statistics as st

rows = collections.defaultdict(list)
for f in sorted(glob.glob("results/*/*.json")):
    r = json.load(open(f))
    a, fin = r["args"], r["final"]
    split = "shifted (round1)" if "round1" in f else "interleaved"
    rows[(split, a["chunker"])].append((a["seed"], fin))

print("| split | chunker | seeds | val bpb (each) | mean val bpb | compression L/M | word-boundary precision | wall time (s) |")
print("|---|---|---|---|---|---|---|---|")
for (split, c), v in sorted(rows.items()):
    b = [x["bpb"] for _, x in v]
    print(f"| {split} | {c} | {','.join(str(s) for s,_ in v)} | {', '.join(f'{x:.3f}' for x in b)} | "
          f"{st.mean(b):.3f} | {st.mean(x['ratio'] for _,x in v):.2f} | "
          f"{st.mean(x['word_prec'] for _,x in v):.2f} | {st.mean(x['time'] for _,x in v):.0f} |")
