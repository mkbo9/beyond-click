"""Sanity checks for the maths in M2, M5 and the channel-centring. Run: uv run pytest"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))  # works even if the package is not installed
from beyond_click.m2_label import _usual_views  # noqa: E402
from beyond_click.m5_compare import bh, effects  # noqa: E402
from beyond_click.store import centre_by_channel  # noqa: E402


def _data(p_top, p_bottom, seed):
    rng = np.random.default_rng(seed)
    channels = np.repeat(np.arange(5), 40)
    labels = np.tile([1, 0], 100)
    F = (rng.random(200) < np.where(labels == 1, p_top, p_bottom)).astype(float)[:, None]
    return F, labels, channels, rng


def test_usual_views_is_median_of_neighbours():
    views = np.array([10, 10, 10, 1000, 10, 10, 10], float)
    assert _usual_views(views, 4)[3] == 10  # one viral video doesn't change its neighbours' median


def test_no_effect_on_unrelated_feature():
    F, labels, channels, rng = _data(0.3, 0.3, seed=0)      # feature unrelated to top/bottom
    effect, lo, hi, p, _ = effects(F, labels, channels, rng, 500)
    assert lo[0] < 0 < hi[0] and p[0] > 0.05


def test_finds_planted_effect():
    F, labels, channels, rng = _data(0.6, 0.2, seed=1)      # feature much more common in top
    effect, lo, hi, p, _ = effects(F, labels, channels, rng, 500)
    assert lo[0] > 0 and p[0] < 0.01


def test_bh_never_lowers_p():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    assert (bh(p) >= p).all()


def test_centring_removes_channel_average():
    rng = np.random.default_rng(2)
    channels = np.repeat(["a", "b"], 10)
    X = rng.normal(0, 1, (20, 4)) + np.where(channels[:, None] == "a", 5.0, -5.0)   # two very different channels
    Xc = centre_by_channel(X, channels)
    assert np.allclose(Xc[channels == "a"].mean(0), 0) and np.allclose(Xc[channels == "b"].mean(0), 0)
