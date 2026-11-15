"""M4 Describe: name each group (SmolVLM2) + score every thumbnail on each element (CLIP).

Writes (docs/INTERFACES.md): data/processed/group_names.csv (group, auto_name, final_name ← you may edit, grouping_id)
                             data/processed/elements.csv (video_id, p_<element> 0–1, has_<element> true/false)
Fallback: if SmolVLM2 fails or use_vlm is false, groups get placeholder names → fill final_name by hand.
Also writes blind check pages (outputs/check_groups.html, check_elements.html): top/bottom hidden.
This step never looks at top/bottom labels.
"""
import hashlib
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
def _clip_element_probs(X: np.ndarray, cfg) -> pd.DataFrame:
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms(cfg["clip_model"], pretrained=cfg["clip_pretrained"])
    model.eval()
    tok = open_clip.get_tokenizer(cfg["clip_model"])
    probs = {}
    with torch.no_grad():
        for name, (yes, no) in cfg["elements"].items():
            t = model.encode_text(tok([yes, no]))     # text fingerprints of the two phrases
            t = (t / t.norm(dim=-1, keepdim=True)).float().numpy()
            logits = 100 * X @ t.T                    # similarity image↔phrase, ×100 = CLIP's usual scale
            e = np.exp(logits - logits.max(1, keepdims=True))
            probs[name] = e[:, 0] / e.sum(1)          # softmax: probability "yes" beats "no"
    return pd.DataFrame(probs)


def _fake_element_probs(df, cfg, rng) -> pd.DataFrame:
    truth = {"face": ["face_text"], "surprised_face": ["face_text"], "big_text": ["face_text"],
             "product": ["product"], "product_in_hand": ["product"], "open_box": ["open_box"]}
    return pd.DataFrame({name: np.clip(df.fake_style.isin(truth.get(name, [])).astype(float) * 0.8
                                       + rng.uniform(0, 0.3, len(df)), 0, 1) for name in cfg["elements"]})


def element_scores(cfg) -> pd.DataFrame:
    df = load_labeled(cfg)
    X = load_fingerprints(cfg, df.video_id)           # raw fingerprints, matched by video id
    rng = np.random.default_rng(cfg["seed"])
    P = _fake_element_probs(df, cfg, rng) if cfg["mode"] == "fake" else _clip_element_probs(X, cfg)
    has = P > 0.5
    out = pd.concat([df[["video_id"]], P.add_prefix("p_").round(3), has.add_prefix("has_")], axis=1)
    out.to_csv(cfg["processed"] / "elements.csv", index=False)
    share = has.mean()
    print("  share of thumbnails with each element:", share.round(2).to_dict())
    odd = share[(share < cfg["element_min_share"]) | (share > cfg["element_max_share"])]
    if len(odd):
        print(f"  uninformative (almost never / almost always present): {list(odd.index)}")
    return out


def run(cfg):
    group_names(cfg)
    element_scores(cfg)
    from .m7_show import check_pages          # imported here to avoid a circular import
    check_pages(cfg)
