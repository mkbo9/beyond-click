"""Checks of the wiring: fake runs stay in .fake_run/, and mixed-up files are caught. Run: uv run pytest"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))  # works even if the package is not installed
from beyond_click import config, m1_collect, m2_label, m3_group  # noqa: E402
from beyond_click.store import PipelineMismatch, load_fingerprints, load_labeled, save_fingerprints  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    c = config.load(fake=True)
    for step in (m1_collect, m2_label, m3_group):
        step.run(c)
    return c


def test_fake_run_never_uses_real_folders(cfg):
    assert ".fake_run" in str(cfg["interim"]) and ".fake_run" in str(cfg["out"])


def test_fingerprints_follow_video_ids_not_row_order(cfg):
    df = load_labeled(cfg)
    X = load_fingerprints(cfg, df.video_id)
    shuffled = df.sample(frac=1, random_state=0)             # same videos, different row order
    assert np.allclose(load_fingerprints(cfg, shuffled.video_id), X[shuffled.index])


def test_missing_fingerprint_is_caught(cfg):
    df = load_labeled(cfg)
    X = load_fingerprints(cfg, df.video_id)
    save_fingerprints(cfg, df.video_id[:-5], X[:-5], "fake-pixels")   # pretend 5 videos were never fingerprinted
    with pytest.raises(PipelineMismatch):
        load_fingerprints(cfg, df.video_id)
    save_fingerprints(cfg, df.video_id, X, "fake-pixels")             # put it back


def test_labels_are_balanced_per_channel(cfg):
    counts = load_labeled(cfg).groupby("channel_handle").label.value_counts().unstack()
    assert (counts["top"] == counts["bottom"]).all()
