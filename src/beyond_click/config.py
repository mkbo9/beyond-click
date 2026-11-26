"""Load config.yaml and give every step the same paths and settings.

File locations follow docs/INTERFACES.md. Paths stored inside CSV files are relative
(e.g. data/raw/abc.jpg), so the repo can move.

Safety rule: a FAKE run never touches real data. In fake mode every data/output folder
lives under the hidden folder .fake_run/ (git-ignored); real runs use the repo folders.

Channel handles: config.yaml (in git, public) holds placeholders only. Real @handles live in
the git-ignored file channels.local.yaml; if it exists, its `channels` list replaces the one
in config.yaml (real mode only; fake mode ignores it).
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]       # repo root: src/beyond_click/config.py → ../../
LOCAL_CHANNELS = ROOT / "channels.local.yaml"    # real handles, git-ignored, never pushed
PLACEHOLDER = "@REPLACE_ME"                      # config.yaml keeps only @REPLACE_ME_1..3


def load(path: str | Path = ROOT / "config.yaml", fake: bool = False,
         local_path: str | Path = LOCAL_CHANNELS) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if fake:
        cfg["mode"] = "fake"
    cfg["channels_source"] = "config.yaml"
    local = Path(local_path)
    if cfg["mode"] != "fake" and local.exists():          # fake mode never reads real handles
        data = yaml.safe_load(local.read_text()) or {}
        channels = data.get("channels") if isinstance(data, dict) else None
        if not isinstance(channels, list) or not channels:
            raise SystemExit(f"{local.name} has no 'channels:' list. See channels.local.example.yaml.")
        cfg["channels"] = [str(h).strip() for h in channels]   # replaces the placeholder list
        cfg["channels_source"] = local.name
    cfg["root"] = ROOT
    # fake runs get their own base folder, so they can never overwrite or mix with real data
    base = ROOT / ".fake_run" if cfg["mode"] == "fake" else ROOT
    cfg["base"] = base
    cfg["thumbs"] = base / "data" / "raw"            # M1 thumbnail images (<video_id>.jpg)
    cfg["interim"] = base / "data" / "interim"       # videos.csv, labeled.csv
    cfg["processed"] = base / "data" / "processed"   # fingerprints, groups, names, elements, predictions
    cfg["models"] = base / "models"                  # saved model for the checker
    cfg["out"] = base / "outputs"                    # charts, compare.csv, summary.md
    cfg["logs"] = base / "logs"                      # collect log
    for key in ("thumbs", "interim", "processed", "models", "out", "logs"):
        cfg[key].mkdir(parents=True, exist_ok=True)
    return cfg


def require_real_channels(cfg: dict) -> list[str]:
    """Stop a real run that still has placeholder channels (used by M1 and smoke/t1_api.py)."""
    chans = [str(h) for h in (cfg.get("channels") or [])]
    if not chans or any(PLACEHOLDER in h for h in chans):
        raise SystemExit("No real channels set. Run: cp channels.local.example.yaml channels.local.yaml\n"
                         "and put your @handles into channels.local.yaml (git-ignored). "
                         "Do NOT put real handles into config.yaml: it is in the public repo.")
    wrong = [h for h in chans if not h.startswith("@")]
    if wrong:
        raise SystemExit(f"Channel handles must start with @ ({cfg.get('channels_source')}): {wrong}")
    return chans


def abs_path(cfg: dict, rel: str) -> Path:
    """Turn a relative path from a CSV into a full path."""
    return cfg["base"] / rel


def rel_path(cfg: dict, p: Path) -> str:
    """Turn a full path into a relative path for storing in a CSV."""
    return str(Path(p).resolve().relative_to(cfg["base"].resolve()))


def pick_device(cfg: dict) -> str:
    """config 'device': cpu (default) | xpu (Intel GPU) | cuda (NVIDIA GPU) | auto (best available)."""
    want = cfg.get("device", "cpu")
    if want != "auto":
        return want
    import torch
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cuda" if torch.cuda.is_available() else "cpu"