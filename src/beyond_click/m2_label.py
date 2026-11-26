"""M2 Label: compare each video with its channel's usual views → top / middle / bottom.

ratio = views / median views of the 20 uploads closest in time (same channel)
top = channel's highest 20% ratios · bottom = lowest 20% · middle = the rest
Writes (docs/INTERFACES.md): data/interim/labeled.csv
"""
import numpy as np
import pandas as pd

from .store import warn_if_old


def _usual_views(views: np.ndarray, k: int) -> np.ndarray:
    out = np.empty(len(views))
    idx = np.arange(len(views))
    for i in idx:
        nearest = idx[np.argsort(np.abs(idx - i), kind="stable")][1:k + 1]  # closest uploads in time, not itself
        out[i] = np.median(views[nearest])
    return out


def run(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg["interim"] / "videos.csv", parse_dates=["published_at"])
    warn_if_old(df)
    parts = []
    for handle, g in df.sort_values("published_at").groupby("channel_handle"):
        if len(g) < cfg["min_videos_per_channel"]:
            # too few videos: "top 20%" would be 1–2 videos and "usual views" unstable
            print(f"  dropped {handle}: only {len(g)} eligible videos (minimum {cfg['min_videos_per_channel']})")
            continue
        g = g.copy()
        g["usual_views"] = _usual_views(g["views"].to_numpy(float), cfg["neighbours"])  # same channel only
        g["ratio"] = (g["views"] + 1) / (g["usual_views"] + 1)
        lo, hi = g["ratio"].quantile([cfg["bottom_quantile"], cfg["top_quantile"]])  # thresholds per channel
        g["label"] = np.where(g["ratio"] >= hi, "top", np.where(g["ratio"] <= lo, "bottom", "middle"))
        parts.append(g)
    if not parts:
        raise SystemExit("No channel has enough eligible videos. Check channels.local.yaml (channels) and config.yaml (age window).")
    out = pd.concat(parts).reset_index(drop=True)
    out.to_csv(cfg["interim"] / "labeled.csv", index=False)
    print(f"  {out.channel_handle.nunique()} channels · labels:", out["label"].value_counts().to_dict())
    return out
