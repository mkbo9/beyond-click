

"""Channel list rules: real handles come only from the git-ignored channels.local.yaml. Run: uv run pytest"""
import sys
from pathlib import Path
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))  # works even if the package is not installed
from beyond_click import config  # noqa: E402

PLACEHOLDERS = ["@REPLACE_ME_1", "@REPLACE_ME_2", "@REPLACE_ME_3"]


@pytest.fixture
def files(tmp_path):
    """A temporary config.yaml (real mode, placeholders) and a path for a local channel file."""
    cfg = yaml.safe_load((config.ROOT / "config.yaml").read_text())
    cfg.update(mode="real", channels=PLACEHOLDERS)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, tmp_path / "channels.local.yaml"     # temp files only: never reads your real local file


def test_local_file_replaces_channels(files):
    path, local = files
    local.write_text('channels: ["@test_a", "@test_b"]\n')
    cfg = config.load(path, local_path=local)
    assert cfg["channels"] == ["@test_a", "@test_b"]
    assert cfg["channels_source"] == "channels.local.yaml"
    config.require_real_channels(cfg)                  # real handles → no stop


def test_fake_mode_ignores_local_file(files):
    path, local = files
    local.write_text('channels: ["@test_a"]\n')
    assert config.load(path, fake=True, local_path=local)["channels"] == PLACEHOLDERS


def test_placeholders_without_local_file_stop(files):
    path, local = files                                # local file does not exist
    cfg = config.load(path, local_path=local)
    with pytest.raises(SystemExit, match="channels.local.yaml"):
        config.require_real_channels(cfg)


@pytest.mark.parametrize("text", ["", "channels: []\n", "other: 1\n"])
def test_broken_local_file_stops(files, text):
    path, local = files
    local.write_text(text)
    with pytest.raises(SystemExit, match="no 'channels:' list"):
        config.load(path, local_path=local)


def test_handle_without_at_stops(files):
    path, local = files
    local.write_text('channels: ["test_a"]\n')
    with pytest.raises(SystemExit, match="must start with @"):
        config.require_real_channels(config.load(path, local_path=local))


def test_tracked_config_has_only_placeholders():
    # config.yaml is public: this test fails if a real handle ever lands in it
    chans = yaml.safe_load((config.ROOT / "config.yaml").read_text())["channels"]
    assert chans and all(h.startswith(config.PLACEHOLDER) for h in chans)










def test_cpu_device_needs_no_torch(monkeypatch):
    """device: cpu must work without importing torch (fake runs and tests have no torch)."""
    monkeypatch.setitem(sys.modules, "torch", None)   # any 'import torch' would now fail
    assert config.pick_device({"device": "cpu"}) == "cpu"
