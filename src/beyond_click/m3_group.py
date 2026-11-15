"""M3 Group: CLIP fingerprints (512 numbers per thumbnail) → PCA → HDBSCAN groups + 2D map.

Writes (docs/INTERFACES.md): data/processed/fingerprints.npz (ids + fingerprints)
                             data/processed/groups.csv (video_id, group, membership, map_x, map_y)
                             data/processed/grouping.json (which grouping rule was used)
Groups are the evidence; the 2D map is for browsing only (never used for statistics).
This step never looks at top/bottom labels.
"""
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .config import abs_path, pick_device
from .store import cached_fingerprints, centre_by_channel, load_labeled, save_fingerprints


def pad_square(img: Image.Image) -> Image.Image:
    """Add black bars instead of cropping, so text near the edges is kept."""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def clip_fingerprints(paths, cfg) -> np.ndarray:
    import open_clip
    import torch
    device = pick_device(cfg)
    # first run downloads the CLIP weights (~600 MB) to ~/.cache; later runs reuse them
    model, _, preprocess = open_clip.create_model_and_transforms(cfg["clip_model"], pretrained=cfg["clip_pretrained"])
    model = model.to(device).eval()                # inference only: CLIP is never trained here
    feats = []
    with torch.no_grad():                          # no gradients → faster, less memory
        for i in range(0, len(paths), cfg["batch_size"]):
            # pad to square (no crop) → CLIP's own resize to 224×224 + colour normalisation
            x = torch.stack([preprocess(pad_square(Image.open(p).convert("RGB"))) for p in paths[i:i + cfg["batch_size"]]])
            f = model.encode_image(x.to(device))   # 49 patches → 12 layers → 512 numbers per image
            f = f / f.norm(dim=-1, keepdim=True)   # scale each fingerprint to length 1
            feats.append(f.float().cpu().numpy())
    return np.vstack(feats).astype("float32")


def fake_fingerprints(paths) -> np.ndarray:
    """Offline stand-in for CLIP: tiny downscaled pixels, scaled to length 1."""
    X = np.stack([np.asarray(Image.open(p).convert("RGB").resize((16, 9)), dtype="float32").ravel() / 255 for p in paths])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def fingerprints(df, cfg) -> np.ndarray:
    """Compute fingerprints only for thumbnails that have none yet (cache keyed by video id + model)."""
    tag = "fake-pixels" if cfg["mode"] == "fake" else f"{cfg['clip_model']}/{cfg['clip_pretrained']}"
    cache = cached_fingerprints(cfg, tag)
    todo = df[~df.video_id.isin(cache.keys())]
    if len(todo):
        paths = [abs_path(cfg, p) for p in todo.thumb_path]   # CSV stores relative paths
        new = fake_fingerprints(paths) if cfg["mode"] == "fake" else clip_fingerprints(paths, cfg)
        cache.update(zip(todo.video_id, new))
    X = np.stack([cache[v] for v in df.video_id])
    save_fingerprints(cfg, df.video_id, X, tag)                # saved WITH ids (see store.py)
    print(f"  fingerprints {X.shape} · {len(todo)} new, {len(df) - len(todo)} from cache")
    return X


def cluster(Z, cfg):
    """HDBSCAN finds the number of groups itself; KMeans fallback if too many stay ungrouped."""
    hdb = HDBSCAN(min_cluster_size=cfg["min_cluster_size"], copy=True).fit(Z)
    labels, membership = hdb.labels_, hdb.probabilities_     # −1 = ungrouped; membership 0–1
    ungrouped = float((labels == -1).mean())
    if ungrouped <= cfg["max_ungrouped_share"] and len(set(labels) - {-1}) >= 2:
        return labels, membership, "HDBSCAN"
    # fallback: KMeans, k chosen by the best silhouette score; every thumbnail gets a group
    best = None
    for k in range(cfg["kmeans_k_range"][0], cfg["kmeans_k_range"][1] + 1):
        if k >= len(Z):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg["seed"]).fit(Z)
        score = silhouette_score(Z, km.labels_)
        if best is None or score > best[0]:
            best = (score, k, km)
    _, k, km = best
    dist = km.transform(Z).min(1)                              # distance to own group centre
    membership = 1 - (dist - dist.min()) / (dist.max() - dist.min() + 1e-9)   # 1 = most typical
    return km.labels_, membership, f"KMeans fallback (k={k}; HDBSCAN left {ungrouped:.0%} ungrouped)"


def dominated_share(labels, channels, limit) -> float:
    """Share of groups where one channel supplies more than `limit` of the thumbnails."""
    groups = [g for g in set(labels) if g != -1]
    if not groups:
        return 0.0
    top = [pd.Series(channels[labels == g]).value_counts(normalize=True).iloc[0] for g in groups]
    return float(np.mean(np.array(top) > limit))


def reduce(X, cfg):
    # PCA: keep the 50 directions where thumbnails differ most (distances get more reliable)
    return PCA(min(cfg["pca_dims"], X.shape[0] - 1, X.shape[1]), random_state=cfg["seed"]).fit_transform(X)


def map_2d(X, seed):
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(X)
    except ImportError:
        print("  (umap not installed: using PCA for the map)")
        return PCA(2, random_state=seed).fit_transform(X)


def run(cfg) -> pd.DataFrame:
    df = load_labeled(cfg)
    channels = df.channel_handle.to_numpy()
    X = fingerprints(df, cfg)

    labels, membership, how = cluster(reduce(X, cfg), cfg)
    dom_raw = dominated_share(labels, channels, cfg["channel_dominated_share"])
    used, centred = X, False
    # Rule fixed in advance (docs/ANALYSIS_PLAN.md): if most groups are just "one channel's look",
    # remove each channel's average fingerprint and group again.
    if dom_raw > 0.5 and len(set(channels)) >= 2:
        used, centred = centre_by_channel(X, channels), True
        labels, membership, how = cluster(reduce(used, cfg), cfg)
    dom = dominated_share(labels, channels, cfg["channel_dominated_share"])

    xy = map_2d(used, cfg["seed"])                 # display only
    out = pd.DataFrame(dict(video_id=df.video_id, group=labels, membership=np.round(membership, 3),
                            map_x=xy[:, 0], map_y=xy[:, 1]))
    out.to_csv(cfg["processed"] / "groups.csv", index=False)
    info = dict(method=how, channel_centred=centred, n_groups=len(set(labels) - {-1}),
                ungrouped_share=round(float((labels == -1).mean()), 3),
                one_channel_groups_before=round(dom_raw, 2), one_channel_groups_after=round(dom, 2))
    (cfg["processed"] / "grouping.json").write_text(json.dumps(info, indent=2))
    print(f"  {info['n_groups']} groups · {how} · channel-centred: {centred} "
          f"· one-channel groups: {dom_raw:.0%} → {dom:.0%} · ungrouped {info['ungrouped_share']:.0%}")
    return out
