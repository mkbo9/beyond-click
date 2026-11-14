"""M3 Group: CLIP fingerprints (512 numbers per thumbnail) → PCA → HDBSCAN groups + 2D map.

Writes (INTERFACES.md): data/processed/fingerprints.npy (rows = labeled.csv order)
                        data/processed/groups.csv (video_id, group, membership, map_x, map_y)
Groups are the evidence; the 2D map is for browsing only (never used for statistics).
"""
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .config import abs_path


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
    # first run downloads the CLIP weights (~600 MB) to ~/.cache; later runs reuse them
    model, _, preprocess = open_clip.create_model_and_transforms(cfg["clip_model"], pretrained=cfg["clip_pretrained"])
    model.eval()                                   # inference only: CLIP is never trained here
    feats = []
    with torch.no_grad():                          # no gradients → faster, less memory
        for i in range(0, len(paths), cfg["batch_size"]):
            # pad to square (no crop) → CLIP's own resize to 224×224 + colour normalisation
            x = torch.stack([preprocess(pad_square(Image.open(p).convert("RGB"))) for p in paths[i:i + cfg["batch_size"]]])
            f = model.encode_image(x)              # 49 patches → 12 layers → 512 numbers per image
            feats.append((f / f.norm(dim=-1, keepdim=True)).numpy())   # scale each fingerprint to length 1
    return np.vstack(feats).astype("float32")


def fake_fingerprints(paths) -> np.ndarray:
    """Offline stand-in for CLIP: tiny downscaled pixels, normalised to length 1."""
    X = np.stack([np.asarray(Image.open(p).convert("RGB").resize((16, 9)), dtype="float32").ravel() / 255 for p in paths])
    X -= X.mean(0)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def group(Z, cfg):
    """HDBSCAN finds the number of groups itself; KMeans fallback if too many stay ungrouped."""
    hdb = HDBSCAN(min_cluster_size=cfg["min_cluster_size"], copy=True).fit(Z)
    labels, membership = hdb.labels_, hdb.probabilities_     # −1 = ungrouped; membership 0–1
    ungrouped = (labels == -1).mean()
    if ungrouped <= cfg["max_ungrouped_share"]:
        return labels, membership, f"HDBSCAN (ungrouped {ungrouped:.0%})"
    # fallback (plan): KMeans, k chosen by the best silhouette score; every thumbnail gets a group
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
    return km.labels_, membership, f"KMeans fallback, k={k} (HDBSCAN left {ungrouped:.0%} ungrouped)"


def map_2d(X, seed):
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(X)
    except ImportError:
        print("  (umap not installed: using PCA for the map)")
        return PCA(2, random_state=seed).fit_transform(X)


def run(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg["interim"] / "labeled.csv")
    paths = [abs_path(cfg, p) for p in df["thumb_path"]]  # CSV stores repo-relative paths
    X = fake_fingerprints(paths) if cfg["mode"] == "fake" else clip_fingerprints(paths, cfg)
    np.save(cfg["processed"] / "fingerprints.npy", X)

    # PCA: keep the 50 directions where thumbnails differ most (distances get more reliable)
    Z = PCA(min(cfg["pca_dims"], X.shape[0] - 1, X.shape[1]), random_state=cfg["seed"]).fit_transform(X)
    labels, membership, how = group(Z, cfg)
    xy = map_2d(X, cfg["seed"])                    # display only
    out = pd.DataFrame(dict(video_id=df["video_id"], group=labels, membership=np.round(membership, 3),
                            map_x=xy[:, 0], map_y=xy[:, 1]))
    out.to_csv(cfg["processed"] / "groups.csv", index=False)
    print(f"  fingerprints {X.shape} · {len(set(labels) - {-1})} groups · {how}")
    return out
