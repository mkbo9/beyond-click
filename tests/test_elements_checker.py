"""Checks of the calibrated element cut (label page, labels file, cuts, M5 verdict) and the checker.

Runs the fake chain in a temporary folder, so the .fake_run/ folder is not changed. Run: uv run pytest
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))  # works even if the package is not installed
from beyond_click import checker, config, m1_collect, m2_label, m3_group, m4_describe, m5_compare, m6_predict  # noqa: E402


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    c = config.load(fake=True)
    base = tmp_path_factory.mktemp("fake")              # own folder: the tests never touch .fake_run/
    c.update(base=base, thumbs=base / "data/raw", interim=base / "data/interim", processed=base / "data/processed",
             models=base / "models", out=base / "outputs", logs=base / "logs", root=base)
    for key in ("thumbs", "interim", "processed", "models", "out", "logs"):
        c[key].mkdir(parents=True, exist_ok=True)
    for step in (m1_collect, m2_label, m3_group, m4_describe):
        step.run(c)
    return c


def _labels_path(cfg):
    return cfg["processed"] / "element_labels.csv"


def test_label_page_shows_ids_only_and_template_is_empty(cfg):
    page = (cfg["out"] / "label_elements.html").read_text()
    assert page.count("<figure") == cfg["label_sample_n"]
    assert "title=" not in page and "Fake video" not in page        # no titles
    from beyond_click.m7_show import BORDER
    assert "@fake_channel" not in page and not any(c in page for c in BORDER.values())   # no channels, no top/bottom
    lab = pd.read_csv(_labels_path(cfg), dtype=str, keep_default_na=False)
    assert list(lab.columns) == ["video_id", *cfg["elements"]] and len(lab) == cfg["label_sample_n"]
    assert (lab.drop(columns="video_id") == "").all().all()
    cuts = json.loads((cfg["processed"] / "element_cuts.json").read_text())
    assert all(c["status"] == "default_0.5" and c["cut"] == 0.5 for c in cuts.values())


def test_best_cut_finds_the_separating_score():
    cut, agreement = m4_describe.best_cut(np.array([0.1, 0.2, 0.3, 0.6, 0.7]), np.array([0, 0, 1, 1, 1], bool))
    assert agreement == 1.0 and 0.2 < cut < 0.3


def test_filled_labels_set_cuts_are_kept_and_unreliable_is_not_compared(cfg):
    df = pd.read_csv(cfg["interim"] / "labeled.csv")
    lab = pd.read_csv(_labels_path(cfg), dtype=str, keep_default_na=False)
    style = df.set_index("video_id").fake_style.loc[lab.video_id]
    lab["face"] = np.where(style == "face_text", "yes", "no")       # correct marks → calibrated, reliable
    # bright_colours: marks alternate along the score order → no cut can agree well → unreliable
    p = pd.read_csv(cfg["processed"] / "elements.csv").set_index("video_id").p_bright_colours.loc[lab.video_id]
    order = np.argsort(p.to_numpy(), kind="stable")
    marks = np.empty(len(lab), dtype=object)
    marks[order] = np.where(np.arange(len(lab)) % 2 == 0, "yes", "no")
    lab["bright_colours"] = marks
    lab.to_csv(_labels_path(cfg), index=False)
    before = _labels_path(cfg).read_text()

    m4_describe.run(cfg)
    assert _labels_path(cfg).read_text() == before                  # your marks are never overwritten
    cuts = json.loads((cfg["processed"] / "element_cuts.json").read_text())
    assert cuts["face"]["status"] == "ok" and cuts["face"]["agreement"] >= 0.75 and cuts["face"]["n_labeled"] == 40
    assert cuts["bright_colours"]["status"] == "unreliable"
    assert cuts["big_text"]["status"] == "default_0.5"              # column left empty → default
    el = pd.read_csv(cfg["processed"] / "elements.csv")
    assert (el.has_face == (el.p_face > cuts["face"]["cut"])).all()  # has_ uses the calibrated cut

    comp = m5_compare.run(cfg).set_index("name")
    assert comp.loc["bright_colours", "verdict"] == "unreliable" and np.isnan(comp.loc["bright_colours", "q"])


def test_wrong_label_value_stops_with_a_clear_message(cfg):
    good = _labels_path(cfg).read_text()
    lab = pd.read_csv(_labels_path(cfg), dtype=str, keep_default_na=False)
    lab.loc[0, "face"] = "maybe"
    lab.to_csv(_labels_path(cfg), index=False)
    with pytest.raises(SystemExit, match="maybe"):
        m4_describe.run(cfg)
    _labels_path(cfg).write_text(good)


def test_labels_for_another_video_list_are_set_aside(cfg):
    good = _labels_path(cfg).read_text()
    pd.DataFrame({"video_id": ["other1", "other2"], **{e: "yes" for e in cfg["elements"]}}).to_csv(
        _labels_path(cfg), index=False)
    m4_describe.run(cfg)
    assert (cfg["processed"] / "element_labels.old.csv").exists()   # moved, not deleted
    fresh = pd.read_csv(_labels_path(cfg), dtype=str, keep_default_na=False)
    assert len(fresh) == cfg["label_sample_n"] and (fresh.drop(columns="video_id") == "").all().all()
    _labels_path(cfg).write_text(good)
    m4_describe.run(cfg)


def test_checker_is_trained_in_m6_and_scores_a_thumbnail(cfg):
    m6_predict.run(cfg)
    assert (cfg["models"] / "checker.joblib").exists()
    image = cfg["thumbs"] / "c0v001.jpg"
    known = checker.score(image, cfg, channel="@fake_channel_0")
    other = checker.score(image, cfg, channel="@not_in_data")
    assert 0 <= known["p_top"] <= 1 and "channel's average" in known["centring"]
    assert "overall average" in other["centring"]
    assert known["group"] in m4_describe.group_labels(cfg).values()
    assert set(known["elements"]) == set(cfg["elements"])
    text = checker.report(known)
    assert checker.DISCLAIMER in text and "FAKE DATA" in text
