"""M4 Describe: name each group (SmolVLM2) + score every thumbnail on each element (CLIP).

Writes (docs/INTERFACES.md): data/processed/group_names.csv (group, auto_name, final_name ← you may edit, grouping_id)
                             data/processed/elements.csv (video_id, p_<element> 0–1, has_<element> true/false)
                             data/processed/element_cuts.json (per element: cut, agreement, n_labeled, status)
                             data/processed/element_labels.csv (empty template; you fill it, it is never overwritten)
                             outputs/label_elements.html (thumbnails to mark by hand: video id only)
Fallback: if SmolVLM2 fails or use_vlm is false, groups get placeholder names → fill final_name by hand.
Element cut: the score above which an element counts as present. It is calibrated on your yes/no
marks for a random sample of thumbnails (best agreement); without marks it is 0.5 (test runs only).
Also writes blind check pages (outputs/check_groups.html, check_elements.html): top/bottom hidden.
This step never looks at top/bottom labels.
"""
import hashlib
import html
import json
import os
import numpy as np
import pandas as pd
from PIL import Image

from .config import abs_path, pick_device
from .store import load_fingerprints, load_labeled


# ---------- group names ----------
def group_labels(cfg) -> dict:
    """group id → display label "g3: close-up face + big text" (your final_name wins; ids keep labels unique)."""
    names = pd.read_csv(cfg["processed"] / "group_names.csv", keep_default_na=False)
    chosen = np.where(names.final_name != "", names.final_name, names.auto_name)
    return {int(g): f"g{int(g)}: {n}" for g, n in zip(names.group, chosen)}


def _grouping_id(groups: pd.DataFrame) -> str:
    """Short code that changes whenever the grouping changes (so hand-written names are never reused wrongly)."""
    text = ",".join(f"{v}:{g}" for v, g in zip(groups.video_id, groups.group))
    return hashlib.md5(text.encode()).hexdigest()[:10]


def _vlm_names(groups: pd.DataFrame, df: pd.DataFrame, cfg) -> dict:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    device = pick_device(cfg)
    # Thumbnails are small. By default the processor enlarges each image and cuts it into many
    # patches (thousands of tokens, very slow). One patch per image is enough here.
    processor = AutoProcessor.from_pretrained(cfg["vlm_model"], size={"longest_edge": cfg["vlm_image_edge"]},
                                              do_image_splitting=cfg["vlm_image_splitting"])
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(cfg["vlm_model"], torch_dtype=dtype).to(device).eval()
    names = {}
    for g, members in groups[groups.group >= 0].groupby("group"):
        typical = members.sort_values("membership", ascending=False).head(cfg["images_per_group"])
        paths = df.set_index("video_id").loc[typical.video_id, "thumb_path"]
        images = [Image.open(abs_path(cfg, p)).convert("RGB") for p in paths]
        messages = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": cfg["vlm_prompt"]}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)  # one <image> slot per thumbnail
        inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False)  # deterministic, short answer
        n_in = inputs["input_ids"].shape[1]
        text = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        names[g] = text.strip().split("\n")[0]
        print(f"  group {g} ({n_in} input tokens): {names[g]}")
    return names


def group_names(cfg) -> pd.DataFrame:
    groups = pd.read_csv(cfg["processed"] / "groups.csv")
    df = load_labeled(cfg)
    ids = sorted(set(groups.group) - {-1})
    names = {g: f"group {g} (unnamed)" for g in ids}
    if cfg["mode"] == "fake":
        style = df.set_index("video_id")["fake_style"]
        for g in ids:  # fake: name = most common drawn style in the group
            names[g] = style.loc[groups[groups.group == g].video_id].mode()[0].replace("_", " + ")
    elif cfg["use_vlm"]:
        try:
            names.update(_vlm_names(groups, df, cfg))
        except Exception as e:  # fallback: you name the groups by hand in group_names.csv
            print(f"  SmolVLM2 failed ({type(e).__name__}: {e})\n  → name the groups by hand: "
                  "fill the final_name column in data/processed/group_names.csv, then re-run from m5")
    gid = _grouping_id(groups)
    out = pd.DataFrame(dict(group=ids, auto_name=[names[g] for g in ids], final_name="", grouping_id=gid))
    path = cfg["processed"] / "group_names.csv"
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        if "grouping_id" in old and (old.grouping_id == gid).all():
            out["final_name"] = out.group.map(old.set_index("group")["final_name"]).fillna("")  # keep your names
        elif (old.get("final_name", pd.Series(dtype=str)) != "").any():
            old.to_csv(path.with_suffix(".old.csv"), index=False)       # the groups changed: old names no longer fit
            print("  NOTE: the grouping changed, so your hand-written names were set aside in group_names.old.csv")
    out.to_csv(path, index=False)
    return out


# ---------- element scores ----------
def _clip_element_probs(X: np.ndarray, cfg, model=None, device="cpu") -> pd.DataFrame:
    """model = an already loaded CLIP model (the checker passes its own); None = load it here on the CPU."""
    import open_clip
    import torch
    if model is None:
        model, _, _ = open_clip.create_model_and_transforms(cfg["clip_model"], pretrained=cfg["clip_pretrained"])
        model.eval()
    tok = open_clip.get_tokenizer(cfg["clip_model"])
    probs = {}
    with torch.no_grad():
        for name, (yes, no) in cfg["elements"].items():
            t = model.encode_text(tok([yes, no]).to(device))     # text fingerprints of the two phrases
            t = (t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy()
            logits = 100 * X @ t.T                    # similarity image↔phrase, ×100 = CLIP's usual scale
            e = np.exp(logits - logits.max(1, keepdims=True))
            probs[name] = e[:, 0] / e.sum(1)          # softmax: probability "yes" beats "no"
    return pd.DataFrame(probs)


def _fake_element_probs(df, cfg, rng) -> pd.DataFrame:
    truth = {"face": ["face_text"], "surprised_face": ["face_text"], "big_text": ["face_text"],
             "product": ["product"], "product_in_hand": ["product"], "open_box": ["open_box"]}
    return pd.DataFrame({name: np.clip(df.fake_style.isin(truth.get(name, [])).astype(float) * 0.8
                                       + rng.uniform(0, 0.3, len(df)), 0, 1) for name in cfg["elements"]})


# ---------- element cuts, calibrated on your hand labels ----------
YES, NO = {"yes", "y", "1", "true"}, {"no", "n", "0", "false"}


def label_sample(df, cfg) -> list:
    """The thumbnails you mark by hand: random but fixed (seed), so a re-run shows the same ones."""
    ids = np.sort(df.video_id.astype(str).to_numpy())          # sorted first → same sample whatever the row order
    rng = np.random.default_rng(cfg["seed"])
    return list(rng.choice(ids, size=min(cfg["label_sample_n"], len(ids)), replace=False))


def label_page(cfg, df, sample) -> None:
    """outputs/label_elements.html: the sample thumbnails with their number and video id ONLY
    (no titles, scores, channels or top/bottom), so your marks cannot be biased."""
    paths = df.set_index("video_id").thumb_path
    meaning = "".join(f"<li><b>{html.escape(e)}</b>: {html.escape(yes)}</li>" for e, (yes, _) in cfg["elements"].items())
    figs = "".join(
        f"<figure style='display:inline-block;margin:6px;text-align:center'>"
        f"<img src='{os.path.relpath(abs_path(cfg, paths[v]), cfg['out'])}' width='320'>"
        f"<figcaption style='font-size:14px'>#{i} · {html.escape(v)}</figcaption></figure>"
        for i, v in enumerate(sample, 1))
    banner = "<p style='background:#fde68a;padding:8px'><b>FAKE DATA: test run</b></p>" if cfg["mode"] == "fake" else ""
    page = (f"<html><body style='font-family:sans-serif'>{banner}<h1>Mark the elements by eye</h1>"
            f"<p>Open <code>data/processed/element_labels.csv</code>. It has one row per thumbnail below, in the same "
            f"order (#1 = first row). For every element write <b>yes</b> or <b>no</b>; leave a cell empty if you "
            f"cannot tell. Then re-run M4.</p><ul>{meaning}</ul>{figs}</body></html>")
    (cfg["out"] / "label_elements.html").write_text(page)


def _parse(value: str, video_id: str, element: str):
    v = str(value).strip().lower()
    if v == "":
        return np.nan                                           # empty = not marked
    if v in YES or v in NO:
        return 1.0 if v in YES else 0.0
    raise SystemExit(f"element_labels.csv: '{value}' for video {video_id}, column {element}. "
                     "Use yes, no or leave it empty, then re-run --from m4 --until m4.")


def hand_labels(cfg, df) -> pd.DataFrame | None:
    """Write the label page; write an empty template if needed; return your marks (0/1/NaN) or None."""
    sample = label_sample(df, cfg)
    label_page(cfg, df, sample)
    path = cfg["processed"] / "element_labels.csv"
    elements = list(cfg["elements"])
    if path.exists():
        lab = pd.read_csv(path, dtype=str, keep_default_na=False)
        cols = [c for c in lab.columns if c != "video_id"]
        filled = any((lab[c].str.strip() != "").any() for c in cols)
        if filled and set(lab.video_id) == set(sample):
            missing = [e for e in elements if e not in lab]
            if missing:                                         # new element in config: add an empty column, keep your marks
                for e in missing:
                    lab[e] = ""
                lab.to_csv(path, index=False)
                print(f"  NOTE: added empty columns {missing} to element_labels.csv (your marks are kept)")
            return pd.DataFrame({"video_id": lab.video_id,
                                 **{e: [_parse(x, v, e) for x, v in zip(lab[e], lab.video_id)] for e in elements}})
        if filled:                                              # marks for a different video list (e.g. new data)
            k, old = 1, path.with_suffix(".old.csv")
            while old.exists():
                k += 1
                old = path.with_name(f"element_labels.old{k}.csv")
            path.rename(old)                                    # set aside, never deleted
            print(f"  NOTE: element_labels.csv belongs to a different video list → moved to {old.name}")
    pd.DataFrame({"video_id": sample, **{e: "" for e in elements}}).to_csv(path, index=False)   # empty template
    return None


def best_cut(scores, truth):
    """Cut with the best agreement between 'score > cut' and your marks. Ties → the cut closest to 0.5."""
    s = np.unique(scores)
    candidates = np.concatenate([[max(s[0] - 1e-3, 0)], (s[:-1] + s[1:]) / 2, [s[-1]]])  # all yes … all no
    agreement = np.array([((scores > c) == truth).mean() for c in candidates])
    best = np.flatnonzero(agreement == agreement.max())
    i = best[np.argmin(np.abs(candidates[best] - 0.5))]
    return round(float(candidates[i]), 4), round(float(agreement[i]), 3)


def calibrate_cuts(P: pd.DataFrame, ids, labels, cfg) -> dict:
    cuts = {}
    for e in cfg["elements"]:
        if labels is None or labels[e].notna().sum() == 0:
            cuts[e] = dict(cut=0.5, agreement=None, n_labeled=0, status="default_0.5")
            continue
        lab = labels.dropna(subset=[e])
        scores = P[e].to_numpy()[pd.Index(ids).get_indexer(lab.video_id)]   # matched by video id
        cut, agr = best_cut(scores, lab[e].to_numpy().astype(bool))
        status = "ok" if agr >= cfg["element_min_agreement"] else "unreliable"
        cuts[e] = dict(cut=cut, agreement=agr, n_labeled=int(len(lab)), status=status)
    return cuts


def load_cuts(cfg) -> dict:
    """element → {cut, agreement, n_labeled, status}. Missing file → 0.5 for every element."""
    path = cfg["processed"] / "element_cuts.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    return {e: saved.get(e, dict(cut=0.5, agreement=None, n_labeled=0, status="default_0.5")) for e in cfg["elements"]}


def element_scores(cfg) -> pd.DataFrame:
    df = load_labeled(cfg)
    X = load_fingerprints(cfg, df.video_id)           # raw fingerprints, matched by video id
    rng = np.random.default_rng(cfg["seed"])
    P = _fake_element_probs(df, cfg, rng) if cfg["mode"] == "fake" else _clip_element_probs(X, cfg)
    P = P.round(3)                                    # the same numbers that are saved in elements.csv
    labels = hand_labels(cfg, df)
    cuts = calibrate_cuts(P, df.video_id, labels, cfg)
    (cfg["processed"] / "element_cuts.json").write_text(json.dumps(cuts, indent=2))
    has = pd.DataFrame({e: P[e] > cuts[e]["cut"] for e in cfg["elements"]})   # present = score above its cut
    out = pd.concat([df[["video_id"]], P.add_prefix("p_"), has.add_prefix("has_")], axis=1)
    out.to_csv(cfg["processed"] / "elements.csv", index=False)

    share = has.mean()
    table = pd.DataFrame(cuts).T.assign(share=share.round(2))
    print("  element cuts (agreement = share of your marks the cut gets right):")
    print("  " + table.to_string().replace("\n", "\n  "))
    if labels is None:
        print("  WARNING: no hand labels yet → cut 0.5 for every element (fine for test runs, not for results).\n"
              "  → open outputs/label_elements.html, fill data/processed/element_labels.csv, "
              "re-run --from m4 --until m4")
    bad = [e for e, c in cuts.items() if c["status"] == "unreliable"]
    if bad:
        print(f"  unreliable (agreement < {cfg['element_min_agreement']:.0%}, not compared in M5): {bad}")
    odd = share[(share < cfg["element_min_share"]) | (share > cfg["element_max_share"])]
    if len(odd):
        print(f"  uninformative (almost never / almost always present): {list(odd.index)}")
    return out


def run(cfg):
    group_names(cfg)
    element_scores(cfg)
    from .m7_show import check_pages          # imported here to avoid a circular import
    check_pages(cfg)
    print("  label page (video ids only): outputs/label_elements.html · your marks: data/processed/element_labels.csv")
