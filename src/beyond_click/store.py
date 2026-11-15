"""Shared reading/writing of pipeline files, with checks that the files belong together.

Why: fingerprints are a plain number table. If the video list changes (new collection)
but old fingerprints are reused, every later result is silently wrong. So fingerprints
are saved WITH their video ids, and every reader checks the ids.
"""
import numpy as np
import pandas as pd


class PipelineMismatch(RuntimeError):
    """Raised when files from different runs are mixed. Fix: re-run from the step named in the message."""


def load_labeled(cfg) -> pd.DataFrame:
    return pd.read_csv(cfg["interim"] / "labeled.csv", parse_dates=["published_at"])


def save_fingerprints(cfg, ids, X: np.ndarray, tag: str) -> None:
    # ids = video ids in row order · tag = which model made them (so a model change forces a recompute)
    np.savez(cfg["processed"] / "fingerprints.npz", ids=np.asarray(ids, dtype=str), X=X.astype("float32"), tag=tag)


def load_fingerprints(cfg, ids) -> np.ndarray:
    """Return fingerprints in exactly the order of `ids`. Stops if any video has no fingerprint."""
    path = cfg["processed"] / "fingerprints.npz"
    if not path.exists():
        raise PipelineMismatch("No fingerprints found: run from m3.")
    f = np.load(path, allow_pickle=False)
    row = {v: i for i, v in enumerate(f["ids"])}
    missing = [v for v in ids if v not in row]
    if missing:
        raise PipelineMismatch(f"{len(missing)} videos have no fingerprint (e.g. {missing[0]}): "
                               "the video list changed, so re-run from m3.")
    return f["X"][[row[v] for v in ids]]            # reorder by id, never trust row order


def cached_fingerprints(cfg, tag: str) -> dict:
    """Fingerprints already computed with the same model: {video_id: vector}."""
    path = cfg["processed"] / "fingerprints.npz"
    if not path.exists():
        return {}
    f = np.load(path, allow_pickle=False)
    return dict(zip(f["ids"], f["X"])) if str(f["tag"]) == tag else {}


def merge_checked(df: pd.DataFrame, cfg, name: str, step: str) -> pd.DataFrame:
    """Join a per-video table onto df by video_id; stop if a video is missing or doubled."""
    other = pd.read_csv(cfg["processed"] / name)
    if other.video_id.duplicated().any() or not set(df.video_id) <= set(other.video_id):
        raise PipelineMismatch(f"{name} does not match the video list: re-run from {step}.")
    return df.merge(other, on="video_id", how="left", validate="one_to_one")


def centre_by_channel(X: np.ndarray, channels) -> np.ndarray:
    """Subtract each channel's average fingerprint: removes the channel's own 'house style'.

    Uses no top/bottom labels, so it is safe to apply to test channels too.
    """
    channels = np.asarray(channels)
    out = X.astype("float64").copy()
    for c in np.unique(channels):
        m = channels == c
        out[m] -= out[m].mean(0)
    return out


def warn_if_old(df: pd.DataFrame, days: int = 30) -> None:
    """YouTube API rule: raw API data must be refreshed or deleted after 30 days."""
    if "collected_at" not in df:
        return
    age = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(df["collected_at"], utc=True).min()).days
    if age > days:
        print(f"  WARNING: data was collected {age} days ago. API rule: refresh (re-run m1) or delete it.")
