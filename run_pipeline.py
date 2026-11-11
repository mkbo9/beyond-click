

#print("Hello from the Beyond Click pipeline!")



"""
the chain 

  uv run python run_pipeline.py --fake --until m2   # test
  uv run python run_pipeline.py --fake              # test
  uv run python run_pipeline.py                     # realrun
  uv run python run_pipeline.py --from m4           # restart
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))  # works even if the package isn't installed
from beyond_click import config  # noqa: E402

STEPS = [("m1", "Collect", "m1_collect"), ("m2", "Label", "m2_label"), ("m3", "Group", "m3_group"),
         ("m4", "Describe", "m4_describe"), ("m5", "Compare", "m5_compare"), ("m6", "Predict", "m6_predict"),
         ("m7", "Show", "m7_show")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="offline test data")
    ap.add_argument("--from", dest="start", default="m1", help="first step to run, e.g. m4")
    ap.add_argument("--until", default="m7", help="last step to run, e.g. m2")
    args = ap.parse_args()
    cfg = config.load(fake=args.fake)
    keys = [k for k, _, _ in STEPS]
    total = time.time()
    for key, name, module_name in STEPS[keys.index(args.start):keys.index(args.until) + 1]:
        # import a step only when it runs: chat 01 needs only its own libraries for m1–m2
        module = importlib.import_module(f"beyond_click.{module_name}")
        t = time.time()
        print(f"\n== {key.upper()} {name} ==")
        module.run(cfg)
        print(f"  done in {time.time() - t:.1f} s")
    print(f"\nDone in {time.time() - total:.1f} s · outputs in {cfg['out']}")






