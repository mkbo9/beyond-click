"""M2 Label: compare each video with its channel's usual views → top / middle / bottom.

ratio = views / median views of the 20 uploads closest in time (same channel)
top = channel's highest 20% ratios · bottom = lowest 20% · middle = the rest
"""
import numpy as np
import pandas as pd


def _usual_views(views: np.ndarray, k: int) -> np.ndarray:
    out = np.empty(len(views))
    idx = np.arange(len(views))
    for i in idx:
        nearest = idx[np.argsort(np.abs(idx - i))][1:k + 1] # closest uploads in time, not itself
        out[i] = np.median(views[nearest])
    return out


def run(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg["interim"] / "videos.csv", parse_dates=["published_at"])
    parts = []
    for _, g in df.sort_values("published_at").groupby("channel"):
        g = g.copy()
        g["usual_views"] = _usual_views(g["views"].to_numpy(float), cfg["neighbours"]) # same channel only
        g["ratio"] = (g["views"] + 1) / (g["usual_views"] + 1)
        lo, hi = g["ratio"].quantile([cfg["bottom_quantile"], cfg["top_quantile"]]) # thresholds per channel
        g["label"] = np.where(g["ratio"] >= hi, "top", np.where(g["ratio"] <= lo, "bottom", "middle"))
        parts.append(g)
    out = pd.concat(parts).reset_index(drop=True)
    out.to_csv(cfg["interim"] / "labeled.csv", index=False)
    print("  labels:", out["label"].value_counts().to_dict())
    return out
