import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
REPO = "fesevu/ethereum_fraud_dataset_by_activity"
api = HfApi()
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")
lab = pd.read_parquet(dl("gnn_dataset/labels/targets_global.parquet"))
mt = pd.read_parquet(dl("gnn_dataset/targets/month_targets.parquet"))
sizes = {}
for e in api.list_repo_tree(REPO, path_in_repo="gnn_dataset/edges_by_month", recursive=True, repo_type="dataset"):
    if type(e).__name__ == "RepoFile":
        sizes[e.path.split("month=")[1].split("/")[0]] = e.size / 1e6
mt = mt.merge(lab[["node_id", "is_scam"]], on="node_id", how="inner")
g = mt.groupby("month").is_scam.agg(scam="sum", total="size").reset_index()
g["normal"] = g.total - g.scam
g["mb"] = g.month.map(sizes)
g["scam_per_gb"] = (g.scam / (g.mb / 1000)).round(0)
print(g[["month", "scam", "normal", "mb", "scam_per_gb"]].to_string(index=False))
print("\nSANITY CHECK 2017-06 (probe saw 49 fraud / 3352 normal):")
print(g[g.month == "2017-06"].to_string(index=False))
sets = {m: set(d.node_id[d.is_scam == 1]) for m, d in mt.groupby("month")}
cov, tot, chosen = set(), 0.0, []
marks = [2000, 4000, 6000, 8000, 10000]
print("\nGREEDY: best scam-wallets-per-GB months first")
for _, r in g.dropna(subset=["mb"]).sort_values("scam_per_gb", ascending=False).iterrows():
    cov |= sets.get(r.month, set()); tot += r.mb; chosen.append(r.month)
    while marks and tot >= marks[0]:
        print(f"~{marks[0]/1000:.0f} GB: {len(chosen)} months, {len(cov)} unique scam wallets")
        if marks[0] == 4000:
            print("   months:", sorted(chosen))
        marks.pop(0)
print("DONE")
