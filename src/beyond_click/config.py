"""Load config.yaml and give every step the same paths and settings.

All file locations follow claude/INTERFACES.md. Paths stored inside CSV files are
relative to the repo root (e.g. data/raw/abc.jpg), so the repo can move.
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]  # repo root: src/beyond_click/config.py → ../../


def load(path: str | Path = ROOT / "config.yaml", fake: bool = False) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if fake:
        cfg["mode"] = "fake"
    cfg["root"] = ROOT
    cfg["thumbs"] = ROOT / "data" / "raw"                  # M1 thumbnail images (<video_id>.jpg)
    cfg["interim"] = ROOT / "data" / "interim"             # videos.csv, labeled.csv
    cfg["processed"] = ROOT / "data" / "processed"         # fingerprints, groups, names, elements, predictions
    cfg["models"] = ROOT / "models"                        # saved model for the checker (chat 03)
    cfg["out"] = ROOT / "outputs"                          # charts, compare.csv, summary.md
    cfg["logs"] = ROOT / "logs"                            # collect log
    for key in ("thumbs", "interim", "processed", "models", "out", "logs"):
        cfg[key].mkdir(parents=True, exist_ok=True)
    return cfg


def abs_path(cfg: dict, rel: str) -> Path:
    """Turn a repo-relative path from a CSV into a full path."""
    return cfg["root"] / rel


def rel_path(cfg: dict, p: Path) -> str:
    """Turn a full path into a repo-relative path for storing in a CSV."""
    return str(Path(p).resolve().relative_to(cfg["root"]))
