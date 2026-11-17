import sys
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
import pandas as pd

REPO = "fesevu/ethereum_fraud_dataset_by_activity"
api = HfApi()
print("START: connecting to Hugging Face...", flush=True)

# 1. Top-level folders
print("\n== TOP LEVEL ==", flush=True)
top = list(api.list_repo_tree(REPO, repo_type="dataset", recursive=False))
for e in top:
    print(type(e).__name__, e.path, flush=True)

# 2. All files, grouped by folder (size and count)
print("\n== LISTING ALL FILES (may take 1-3 min) ==", flush=True)
folders = defaultdict(lambda: [0, 0])
parquets = []
n = 0
for e in api.list_repo_tree(REPO, repo_type="dataset", recursive=True):
    if type(e).__name__ == "RepoFile":
        d = e.path.rsplit("/", 1)[0] if "/" in e.path else "."
        folders[d][0] += 1
        folders[d][1] += e.size or 0
        if e.path.endswith(".parquet"):
            parquets.append((e.path, e.size or 0))
        n += 1
        if n % 2000 == 0:
            print(f"  ...{n} files scanned", flush=True)

print(f"\nTotal files: {n}", flush=True)
for d, (c, s) in sorted(folders.items()):
    print(f"{d}: {c} files, {s/1e9:.3f} GB", flush=True)

# 3. Labels file (3.5 MB, safe to download)
print("\n== LABELS FILE ==", flush=True)
p = hf_hub_download(REPO, "gnn_dataset/labels/targets_global.parquet", repo_type="dataset")
lab = pd.read_parquet(p)
print("shape:", lab.shape)
print(lab.dtypes)
print(lab.head(10))
for c in lab.columns:
    if lab[c].nunique() < 20:
        print(f"value counts for {c}:\n{lab[c].value_counts()}")

# 4. Schema of one file from each other top-level folder (footer only, no big download)
print("\n== SCHEMA OF ONE FILE PER FOLDER ==", flush=True)
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
seen = set()
for path, size in parquets:
    d = path.rsplit("/", 1)[0]
    if d in seen or "labels" in d:
        continue
    seen.add(d)
    try:
        with fs.open(f"datasets/{REPO}/{path}", "rb") as f:
            pf = pq.ParquetFile(f)
            print(f"\n{path} ({size/1e6:.1f} MB, {pf.metadata.num_rows} rows, {pf.metadata.num_row_groups} row groups)")
            print(pf.schema_arrow)
    except Exception as ex:
        print(f"could not read {path}: {ex}")
    if len(seen) >= 8:
        break

print("\nDONE", flush=True)