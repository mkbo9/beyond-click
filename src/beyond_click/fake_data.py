"""Offline test data: drawn unboxing-style thumbnails + made-up views. Used only with --fake.

Lets you test the chain without an API key or model downloads. Files go to .fake_run/.
Planted signal: 'face_text' thumbnails get ~40% more views than usual,
'product' thumbnails ~30% fewer. A working pipeline should find this.
Each fake channel also has its own background tint (its 'house style'), like real channels.
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .config import rel_path

STYLES = ["face_text", "product", "open_box"]
VIEW_EFFECT = {"face_text": 1.4, "product": 0.7, "open_box": 1.0}   # the planted signal
TINTS = [(205, 225, 250), (250, 225, 205), (215, 245, 215)]          # one background tint per channel


def draw(style: str, tint, rng) -> Image.Image:
    bg = tuple(int(np.clip(t + rng.integers(-12, 13), 0, 255)) for t in tint)
    img = Image.new("RGB", (320, 180), bg)
    d = ImageDraw.Draw(img)
    if style == "face_text":                      # big face on the right, bold text on the left
        cx, cy, r = 220 + rng.integers(-15, 15), 90, 60
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(244, 165, 130))
        d.rectangle([20, 40, 150, 70], fill=(17, 24, 39))
        d.rectangle([20, 85, 110, 115], fill=(220, 38, 38))
    elif style == "product":                      # a dark gadget in the middle
        w = int(rng.integers(60, 90))
        d.rounded_rectangle([160 - w, 30, 160 + w, 150], radius=14, fill=(30, 30, 35))
    else:                                         # an open brown box
        d.polygon([(90, 80), (230, 80), (215, 165), (105, 165)], fill=(181, 131, 90))
        d.polygon([(90, 80), (60, 45), (150, 45), (160, 80)], fill=(160, 112, 75))
    return img


def make(cfg) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for c in range(3):
        base = 10 ** rng.uniform(4, 5.3)          # channel size
        for i in range(40):
            style = rng.choice(STYLES)
            vid = f"c{c}v{i:03d}"
            path = cfg["thumbs"] / f"{vid}.jpg"
            draw(style, TINTS[c], rng).save(path)
            rows.append(dict(
                video_id=vid, channel_handle=f"@fake_channel_{c}", channel=f"Fake channel {c}",
                title=f"Fake video {i} ({style})",
                published_at=pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=int(i * 7)),
                duration_s=600, views=int(base * VIEW_EFFECT[style] * rng.lognormal(0, 0.35)),
                thumb_path=rel_path(cfg, path), collected_at=pd.Timestamp.now(tz="UTC"),
                fake_style=style))       # extra column, fake mode only (used to check results)
    return pd.DataFrame(rows)
