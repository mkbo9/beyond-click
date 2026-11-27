"""Checker: score a new thumbnail with a model trained on all our top/bottom thumbnails.

  uv run python -m beyond_click.checker path/to/thumbnail.jpg                     # any channel
  uv run python -m beyond_click.checker path/to/thumbnail.jpg --channel @handle   # a channel in our data
  uv run python -m beyond_click.checker path/to/thumbnail.jpg --fake              # model from the fake run

Prints: top-likelihood, nearest group (with its name), elements found (with their cuts).
A hint, not a verdict: it shows what appeared more often in top thumbnails of the channels we
collected; it does not say what will get views.
Trained at the end of M6 (so every run retrains it) → models/checker.joblib (docs/INTERFACES.md).
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .m4_describe import group_labels, load_cuts
from .store import centre_by_channel, load_fingerprints, load_labeled, merge_checked

DISCLAIMER = "A hint, not a verdict; consumer tech review niche only."


def train(cfg) -> Path:
    """Fit the final model on ALL top/bottom thumbnails and save everything needed to score a new one."""
    df = load_labeled(cfg)
    X = load_fingerprints(cfg, df.video_id).astype("float64")    # raw fingerprints, matched by video id
    ch = df.channel_handle.to_numpy()
    X_c = centre_by_channel(X, ch)                               # same centring as the M6 model
    keep = df.label.isin(["top", "bottom"]).to_numpy()
    y = (df.label[keep] == "top").astype(int).to_numpy()
    # same settings as M6 (docs/ANALYSIS_PLAN.md): standardised, C = 0.1
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X_c[keep], y)

    # group centres in the space M3 grouped in (channel-centred or raw, see grouping.json)
    grouping = json.loads((cfg["processed"] / "grouping.json").read_text())
    groups = merge_checked(df[["video_id"]], cfg, "groups.csv", "m3").group.to_numpy()
    G = X_c if grouping["channel_centred"] else X
    names = group_labels(cfg)                                    # "g3: phones on a desk"
    centres = np.stack([G[groups == g].mean(0) for g in names])

    tag = str(np.load(cfg["processed"] / "fingerprints.npz")["tag"])   # which model made the fingerprints
    bundle = dict(
        model=model, tag=tag, mode=cfg["mode"],
        channel_means={c: X[ch == c].mean(0) for c in np.unique(ch)},   # for centring a known channel
        overall_mean=X.mean(0),                                        # for any other channel
        group_centred=bool(grouping["channel_centred"]), group_ids=list(names),
        group_names=list(names.values()), group_centres=centres,
        elements=cfg["elements"], cuts=load_cuts(cfg))
    if cfg["mode"] == "fake":   # fake mode has no CLIP text model: element scores come from the nearest thumbnails
        el = merge_checked(df[["video_id"]], cfg, "elements.csv", "m4")
        bundle.update(ref_X=X, ref_P=el[[f"p_{e}" for e in cfg["elements"]]].to_numpy())
    path = cfg["models"] / "checker.joblib"
    joblib.dump(bundle, path)
    print(f"  checker trained on {keep.sum()} top/bottom thumbnails → {path.relative_to(cfg['root'])}")
    return path


def _fingerprint(image: Path, cfg, bundle):
    """→ (fingerprint, clip) — clip is the loaded CLIP model (None in fake mode), reused for the elements."""
    from .m3_group import clip_fingerprints, fake_fingerprints, load_clip
    if bundle["mode"] == "fake":
        return fake_fingerprints([image])[0].astype("float64"), None
    tag = f"{cfg['clip_model']}/{cfg['clip_pretrained']}"
    if tag != bundle["tag"]:
        raise SystemExit(f"The checker was trained on {bundle['tag']} fingerprints, config.yaml says {tag}: re-run from m3.")
    clip = load_clip(cfg)
    return clip_fingerprints([image], cfg, clip)[0].astype("float64"), clip


def _element_probs(x, cfg, bundle, clip) -> dict:
    els = list(bundle["elements"])
    if clip is None:   # fake: average scores of the 5 most similar thumbnails in our data
        sim = bundle["ref_X"] @ x / (np.linalg.norm(bundle["ref_X"], axis=1) * np.linalg.norm(x))
        return dict(zip(els, bundle["ref_P"][np.argsort(-sim)[:5]].mean(0)))
    from .m4_describe import _clip_element_probs
    model, _, device = clip
    P = _clip_element_probs(x[None].astype("float32"), {**cfg, "elements": bundle["elements"]}, model=model, device=device)
    return {e: float(P[e].iloc[0]) for e in els}


def score(image, cfg, channel: str | None = None) -> dict:
    path = cfg["models"] / "checker.joblib"
    if not path.exists():
        raise SystemExit(f"No checker found at {path.relative_to(cfg['root'])}: run the pipeline up to m6 first.")
    b = joblib.load(path)
    image = Path(image).expanduser()
    if not image.exists():
        raise SystemExit(f"Image not found: {image}")
    x, clip = _fingerprint(image, cfg, b)

    known = {c.lower(): c for c in b["channel_means"]}
    handle = None if channel is None else ("@" + channel.lstrip("@")).lower()
    if handle in known:   # channel in our data: remove that channel's own average look (as in training)
        mean, centring = b["channel_means"][known[handle]], "the channel's average (channel is in our data)"
    else:                 # unknown channel: the best we have is the average over all our channels
        mean = b["overall_mean"]
        centring = "the overall average (" + ("channel not in our data" if channel else "no channel given") + ")"
        if channel:   # tell which channels we do know (local output only)
            centring += "; channels in our data: " + ", ".join(sorted(b["channel_means"]))
    x_c = x - mean
    p_top = float(b["model"].predict_proba(x_c[None])[0, 1])

    g = x_c if b["group_centred"] else x
    C = b["group_centres"]
    sim = C @ g / (np.linalg.norm(C, axis=1) * np.linalg.norm(g) + 1e-12)   # cosine similarity to each group centre
    i = int(np.argmax(sim))

    probs = _element_probs(x, cfg, b, clip)
    elements = {e: dict(score=round(float(p), 3), cut=b["cuts"][e]["cut"], found=bool(round(float(p), 3) > b["cuts"][e]["cut"]),
                        status=b["cuts"][e]["status"]) for e, p in probs.items()}
    return dict(image=str(image), fake=b["mode"] == "fake", p_top=round(p_top, 3), centring=centring,
                group=b["group_names"][i], group_similarity=round(float(sim[i]), 2), elements=elements)


def report(r: dict) -> str:
    note = {"default_0.5": "cut not calibrated yet (0.5)", "unreliable": "unreliable: not compared in the analysis"}
    lines = [f"Checker: {Path(r['image']).name}" + ("   [FAKE DATA model]" if r["fake"] else ""),
             f"  top-likelihood : {r['p_top']:.2f}   (0.5 = no lean towards top or bottom)",
             f"  centred on     : {r['centring']}",
             f"  nearest group  : {r['group']}   (similarity {r['group_similarity']:.2f})",
             "  elements (score / cut → found):"]
    for e, v in r["elements"].items():
        extra = f"   ({note[v['status']]})" if v["status"] in note else ""
        lines.append(f"    {e:16s} {v['score']:.3f} / {v['cut']:.3f} → {'yes' if v['found'] else 'no'}{extra}")
    lines.append(f"  {DISCLAIMER}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score one thumbnail. " + DISCLAIMER)
    ap.add_argument("image", help="path to a thumbnail image (jpg / png)")
    ap.add_argument("--channel", help="@handle of the channel, if it is one of ours (better centring)")
    ap.add_argument("--fake", action="store_true", help="use the model from the fake run (.fake_run/)")
    args = ap.parse_args(argv)
    cfg = config.load(fake=args.fake)
    print(report(score(args.image, cfg, args.channel)))


if __name__ == "__main__":
    main()
