"""Run the chain M1 → M7 and time each step.

  uv run python run_pipeline.py --fake --until m2   # offline test of M1–M2 (no API key, no models)
  uv run python run_pipeline.py --fake              # offline test of the whole chain
  uv run python run_pipeline.py                     # real run: config.yaml + channels.local.yaml
  uv run python run_pipeline.py --from m4           # restart from a step (earlier outputs are reused)

Fake runs write only into the hidden folder .fake_run/ and never touch real data.
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))  # works even if the package isn't installed
from beyond_click import config  # noqa: E402
from beyond_click.store import PipelineMismatch  # noqa: E402

STEPS = [("m1", "Collect", "m1_collect"), ("m2", "Label", "m2_label"), ("m3", "Group", "m3_group"),
         ("m4", "Describe", "m4_describe"), ("m5", "Compare", "m5_compare"), ("m6", "Predict", "m6_predict"),
         ("m7", "Show", "m7_show")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="offline test data, written to .fake_run/")
    ap.add_argument("--from", dest="start", default="m1", help="first step to run, e.g. m4")
    ap.add_argument("--until", default="m7", help="last step to run, e.g. m2")
    args = ap.parse_args()
    cfg = config.load(fake=args.fake)
    if cfg["mode"] == "fake":
        print(f"FAKE RUN: test data only, everything goes to {cfg['base']}")
        
    else:
        print(f"REAL RUN: {len(cfg['channels'])} channels from {cfg['channels_source']}")  # count only, no handles
        
    keys = [k for k, _, _ in STEPS]
    total = time.time()
    for key, name, module_name in STEPS[keys.index(args.start):keys.index(args.until) + 1]:
        # import a step only when it runs, so each step needs only its own libraries
        module = importlib.import_module(f"beyond_click.{module_name}")
        t = time.time()
        print(f"\n== {key.upper()} {name} ==")
        try:
            module.run(cfg)
        except PipelineMismatch as e:               # files from different runs were mixed
            raise SystemExit(f"STOPPED: {e}")
        print(f"  done in {time.time() - t:.1f} s")
    print(f"\nDone in {time.time() - total:.1f} s · outputs in {cfg['out']}")
