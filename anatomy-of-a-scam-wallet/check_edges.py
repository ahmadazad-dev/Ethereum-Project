import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "fesevu/ethereum_fraud_dataset_by_activity"
api = HfApi()
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")

print("== README (first 150 lines) ==", flush=True)
try:
    for line in open(dl("README.md"), encoding="utf-8", errors="replace").read().splitlines()[:150]:
        print(line[:200])
except Exception as e:
    print("README error:", e)

for folder in ["gnn_dataset", "gnn_dataset/mapping", "gnn_dataset/meta", "gnn_dataset/targets"]:
    print("\n== FOLDER", folder, flush=True)
    for e in api.list_repo_tree(REPO, path_in_repo=folder, repo_type="dataset"):
        if type(e).__name__ != "RepoFile":
            continue
        print(e.path, e.size)
        if e.path.endswith(".parquet"):
            t = pd.read_parquet(dl(e.path))
            print(t.shape); print(t.dtypes); print(t.head(3).to_string())
        elif (e.size or 0) < 20000:
            print(open(dl(e.path), encoding="utf-8", errors="replace").read()[:1500])

print("\n== MONTH PROBE: 2017-06 ==", flush=True)
lab = pd.read_parquet(dl("gnn_dataset/labels/targets_global.parquet"))
print(pd.crosstab(lab.is_scam, lab.is_contract))
ed = pd.read_parquet(dl("gnn_dataset/edges_by_month/month=2017-06/edges.parquet"))
print(ed.shape); print(ed.head(5).to_string())
print("ts min/max:", ed.ts.min(), ed.ts.max())
print("as seconds:", pd.to_datetime(ed.ts.min(), unit="s"), "| as ms:", pd.to_datetime(ed.ts.min(), unit="ms"))
print("duplicate tx_hash:", int(ed.tx_hash.duplicated().sum()), "| contract_creation rows:", int(ed.contract_creation.sum()))
ids = lab.node_id.values
s = ed.src_id.isin(ids); d = ed.dst_id.isin(ids)
print("src labeled %.3f | dst labeled %.3f | both %.3f | either %.3f" % (s.mean(), d.mean(), (s & d).mean(), (s | d).mean()))
scam = lab.node_id[lab.is_scam == 1].values
norm = lab.node_id[lab.is_scam == 0].values
act = pd.concat([ed.src_id, ed.dst_id]); act = act[act.isin(ids)]
cnt = act.value_counts()
big = cnt[cnt >= 5]
print("labeled wallets active this month: fraud %d, normal %d" % (cnt.index.isin(scam).sum(), cnt.index.isin(norm).sum()))
print("... with >=5 txs: fraud %d, normal %d" % (big.index.isin(scam).sum(), big.index.isin(norm).sum()))
print("value_wei sample:", ed.value_wei.head(5).tolist(), "| tx_fee_wei sample:", ed.tx_fee_wei.head(3).tolist())
print("DONE")
