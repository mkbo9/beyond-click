"""Offline test data: drawn unboxing-style thumbnails + made-up views. Used only with --fake.

Lets you test the chain without an API key or model downloads.
Planted signal: 'face_text' thumbnails get ~40% more views than usual,
'product' thumbnails ~30% fewer. A working pipeline should find this.
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .config import rel_path

STYLES = ["face_text", "product", "open_box"]
VIEW_EFFECT = {"face_text": 1.4, "product": 0.7, "open_box": 1.0} # the planted signal


def draw(style: str, rng) -> Image.Image:
    img = Image.new("RGB", (320, 180), tuple(int(v) for v in rng.integers(150, 255, 3)))
    d = ImageDraw.Draw(img)
    if style == "face_text":
        # face + text
        cx, cy, r = 220 + rng.integers(-15, 15), 90, 60
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(244, 165, 130))
        d.rectangle([20, 40, 150, 70], fill=(17, 24, 39))
        d.rectangle([20, 85, 110, 115], fill=(220, 38, 38))
    elif style == "product":
        # plain bg + doodad
        d.rectangle([0, 0, 320, 180], fill=(245, 245, 245))
        w = int(rng.integers(60, 90))
        d.rounded_rectangle([160 - w, 30, 160 + w, 150], radius=14, fill=(30, 30, 35))
    else:
        # box
        d.polygon([(90, 80), (230, 80), (215, 165), (105, 165)], fill=(181, 131, 90))
        d.polygon([(90, 80), (60, 45), (150, 45), (160, 80)], fill=(160, 112, 75))
    return img


def make(cfg) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for c in range(3):
        base = 10 ** rng.uniform(4, 5.3) # channel size
        for i in range(40):
            style = rng.choice(STYLES)
            vid = f"c{c}v{i:03d}"
            path = cfg["thumbs"] / f"{vid}.jpg"
            draw(style, rng).save(path)
            rows.append(dict(
                video_id=vid, channel_handle=f"@fake_channel_{c}", channel=f"Fake channel {c}",
                title=f"Fake video {i} ({style})",
                published_at=pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=int(i * 7)),
                duration_s=600, views=int(base * VIEW_EFFECT[style] * rng.lognormal(0, 0.35)),
                thumb_path=rel_path(cfg, path), collected_at=pd.Timestamp.now(tz="UTC"),
                fake_style=style)) # extra column, fake mode only (used to check results)
    return pd.DataFrame(rows)
