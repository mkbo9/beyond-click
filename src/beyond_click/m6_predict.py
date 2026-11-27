"""M6 Predict: can a model tell top from bottom thumbnails on channels it never saw?

Model : logistic regression on channel-centred fingerprints (each channel's average look removed).
Test  : leave channels out (GroupKFold): the test channels are never seen in training.
Score : AUC per test channel, then averaged (0.5 = coin flip, 1.0 = perfect). This is the headline:
        "top" is defined inside a channel, so ranking is judged inside each channel.
Checks: shuffled labels, 20 times (must average ~0.5) · a model using only the element scores
        · the same model on raw fingerprints (shows how much the channel look disturbs).
Writes (docs/INTERFACES.md): outputs/predict_scores.csv · data/processed/predictions.csv
At the end it trains the checker on all top/bottom thumbnails → models/checker.joblib (see checker.py).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .store import centre_by_channel, load_fingerprints, load_labeled, merge_checked


def predict_unseen_channels(X, y, channels):
    """For every thumbnail: the model's top-likelihood, from a model that never saw its channel."""
    p = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=min(5, len(np.unique(channels)))).split(X, y, channels):
        # C=0.1: strong regularisation, because there are few thumbnails and many numbers per thumbnail
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        model.fit(X[train], y[train])
        p[test] = model.predict_proba(X[test])[:, 1]
    return p


def scores(y, p, channels) -> dict:
    per_channel = [roc_auc_score(y[channels == c], p[channels == c])
                   for c in np.unique(channels) if len(set(y[channels == c])) == 2]
    return dict(auc=float(np.mean(per_channel)),            # headline: average of per-channel AUCs
                auc_pooled=float(roc_auc_score(y, p)),      # all channels thrown together (for reference)
                auc_min_channel=float(min(per_channel)), auc_max_channel=float(max(per_channel)))


def run(cfg) -> pd.DataFrame:
    df = load_labeled(cfg)
    X_raw = load_fingerprints(cfg, df.video_id)              # matched by video id, never by row order
    X_centred = centre_by_channel(X_raw, df.channel_handle)  # uses all thumbnails, no labels
    el = merge_checked(df[["video_id"]], cfg, "elements.csv", "m4")
    keep = df.label.isin(["top", "bottom"]).to_numpy()       # the middle 60% is not used
    d = df[keep].reset_index(drop=True)
    y = (d.label == "top").astype(int).to_numpy()
    ch = d.channel_handle.to_numpy()
    if len(np.unique(ch)) < 2:
        print("  M6 needs at least 2 channels (better 3+): skipped")
        pd.DataFrame(columns=["model", "auc", "auc_pooled", "auc_min_channel", "auc_max_channel"]).to_csv(
            cfg["out"] / "predict_scores.csv", index=False)
        return pd.DataFrame()

    X_c, X_r = X_centred[keep], X_raw[keep]
    X_el = el[[c for c in el.columns if c.startswith("p_")]].to_numpy()[keep]
    results = []

    p = predict_unseen_channels(X_c, y, ch)
    results.append(dict(model="fingerprint model", **scores(y, p, ch)))
    pd.DataFrame(dict(video_id=d.video_id, label=d.label, p_top=p.round(3))).to_csv(
        cfg["processed"] / "predictions.csv", index=False)
    results.append(dict(model="element-only model", **scores(y, predict_unseen_channels(X_el, y, ch), ch)))
    results.append(dict(model="fingerprint model, raw (not centred)", **scores(y, predict_unseen_channels(X_r, y, ch), ch)))

    # sanity check: shuffle top/bottom WITHIN each channel, 20 times; the average must be ~0.5
    rng = np.random.default_rng(cfg["seed"])
    shuffled = []
    for _ in range(20):
        y_s = y.copy()
        for c in np.unique(ch):
            m = ch == c
            y_s[m] = rng.permutation(y_s[m])
        shuffled.append(scores(y_s, predict_unseen_channels(X_c, y_s, ch), ch))
    s = pd.DataFrame(shuffled)
    results.append(dict(model="shuffled labels (sanity check, mean of 20)", auc=s.auc.mean(),
                        auc_pooled=s.auc_pooled.mean(), auc_min_channel=s.auc.min(), auc_max_channel=s.auc.max()))

    out = pd.DataFrame(results).round(3)
    out.to_csv(cfg["out"] / "predict_scores.csv", index=False)
    print(out.to_string(index=False))
    from .checker import train                  # final model for scoring new thumbnails (every run retrains it)
    train(cfg)
    return out
