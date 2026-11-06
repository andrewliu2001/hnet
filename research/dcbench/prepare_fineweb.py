"""Download FineWeb-Edu sample-10BT shards and write raw-byte train/val files.

Documents are UTF-8 bytes separated by a single 0x00 byte. The last 1% of documents
(the sample is already shuffled) is the validation set.

  python prepare_fineweb.py --shards 2 --out data
"""
import argparse, os, urllib.request
import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument("--shards", type=int, default=2)
p.add_argument("--out", default="data")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

base = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{:03d}_00000.parquet"
docs = []
for i in range(args.shards):
    path = os.path.join(args.out, f"shard{i}.parquet")
    if not os.path.exists(path):
        print("downloading", base.format(i), flush=True)
        urllib.request.urlretrieve(base.format(i), path + ".tmp")
        os.rename(path + ".tmp", path)
    t = pq.read_table(path, columns=["text"])
    docs.extend(t.column("text").to_pylist())
    os.remove(path)
n_val = len(docs) // 100
for name, part in [("val", docs[-n_val:]), ("train", docs[:-n_val])]:
    with open(os.path.join(args.out, f"{name}.bin"), "wb") as f:
        for d in part:
            f.write(d.encode("utf-8"))
            f.write(b"\x00")
    print(name, len(part), "docs", os.path.getsize(os.path.join(args.out, f"{name}.bin")) / 1e9, "GB", flush=True)
