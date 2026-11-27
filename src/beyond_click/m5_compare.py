"""M5 Compare: is a group / element more common among a channel's top or bottom thumbnails?

effect    = average over channels of (share among top − share among bottom)
error bar = 95% t-interval over the per-channel differences (honest with few channels)
p         = shuffle test: top/bottom labels shuffled within each channel
q         = p corrected for testing many groups and elements (Benjamini–Hochberg)
Writes (docs/INTERFACES.md): outputs/compare.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from .m4_describe import group_labels, load_cuts
from .store import load_labeled, merge_checked


def effects(F, labels, channels, rng, n_perm):
    """F = table [thumbnails × features] of 0/1 values. All features are handled together.

    Returns per feature: effect, ci_low, ci_high, p, and the per-channel differences.
    """
    diffs, perm_sum = [], 0.0
    for c in np.unique(channels):
        m = channels == c
        Fc, y = F[m], labels[m]
        n_top, n_bot = int(y.sum()), int((1 - y).sum())
        if n_top == 0 or n_bot == 0:
            continue                                         # channel has no top or no bottom thumbnails
        diffs.append(Fc[y == 1].mean(0) - Fc[y == 0].mean(0))   # share among top − share among bottom
        # shuffle the labels WITHIN this channel, n_perm times at once (one row = one shuffle)
        Y = rng.permuted(np.tile(y, (n_perm, 1)), axis=1).astype(float)
        perm_sum = perm_sum + (Y @ Fc) / n_top - ((1 - Y) @ Fc) / n_bot
    D = np.array(diffs)                                      # [channels × features]
    n = len(D)
    effect = D.mean(0)
    perm = perm_sum / n                                      # [shuffles × features]
    p = (1 + (np.abs(perm) >= np.abs(effect) - 1e-12).sum(0)) / (1 + n_perm)
    if n >= 2:
        half = stats.t.ppf(0.975, n - 1) * D.std(0, ddof=1) / np.sqrt(n)   # t-interval over channels
    else:
        half = np.full(F.shape[1], np.nan)                   # one channel: no error bar possible
    return effect, effect - half, effect + half, p, D


def bh(p):
    """Benjamini–Hochberg: adjust p-values for many tests."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def run(cfg) -> pd.DataFrame:
    d = load_labeled(cfg)
    d = merge_checked(d, cfg, "groups.csv", "m3")           # stops if files come from different runs
    d = merge_checked(d, cfg, "elements.csv", "m4")
    labels_by_group = group_labels(cfg)                      # unique display names, e.g. "g3: face + text"
    d = d[d.label.isin(["top", "bottom"])].reset_index(drop=True)
    y = (d.label == "top").astype(int).to_numpy()
    channels = d.channel_handle.to_numpy()
    rng = np.random.default_rng(cfg["seed"])

    kinds = ["group"] * len(labels_by_group) + ["element"] * len(cfg["elements"])
    names = list(labels_by_group.values()) + list(cfg["elements"])
    F = np.column_stack([(d.group == g).to_numpy(float) for g in labels_by_group]
                        + [d[f"has_{e}"].to_numpy(float) for e in cfg["elements"]])
    effect, lo, hi, p, D = effects(F, y, channels, rng, cfg["permutation_n"])

    share = F.mean(0)
    largest = [pd.Series(channels[F[:, j] > 0]).value_counts(normalize=True).iloc[0] if F[:, j].any() else np.nan
               for j in range(F.shape[1])]
    out = pd.DataFrame(dict(kind=kinds, name=names, n_thumbnails=F.sum(0).astype(int), effect=effect,
                            ci_low=lo, ci_high=hi, p=p, channels=len(D), channels_positive=(D > 0).sum(0),
                            largest_channel_share=np.round(largest, 2)))
    informative = (share >= cfg["element_min_share"]) & (share <= cfg["element_max_share"])
    # elements whose cut agrees too little with the hand marks: their yes/no is mostly CLIP's error → not compared
    cuts = load_cuts(cfg)
    unreliable = np.array([k == "element" and cuts[nm]["status"] == "unreliable" for k, nm in zip(kinds, names)])
    tested = informative & ~unreliable
    out["q"] = np.nan
    out.loc[tested, "q"] = bh(out.p[tested])                 # only compared features count as tests
    sig = out.q < cfg["fdr_alpha"]
    out["verdict"] = np.select([unreliable, ~informative, sig & (out.ci_low > 0), sig & (out.ci_high < 0)],
                               ["unreliable", "uninformative", "more in top", "more in bottom"], "unclear")
    out.round(4).to_csv(cfg["out"] / "compare.csv", index=False)
    print(out[["kind", "name", "effect", "ci_low", "ci_high", "q", "verdict"]].round(3).to_string(index=False))
    return out
