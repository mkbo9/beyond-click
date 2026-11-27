
"""T0 environment check: Python version, libraries, API key present (never printed).

Run in the repo root:  uv run python smoke/t0_env.py
Changes nothing. Exit code 0 = all fine.
"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ok = True

print(f"Python {sys.version.split()[0]} ({sys.executable})")
if sys.version_info[:2] < (3, 12):
    print("  FAIL: need Python 3.12 or newer")
    ok = False

# libraries needed by M1, M2 and the fake chain
for name in ["pandas", "numpy", "yaml", "dotenv", "requests", "isodate", "googleapiclient",
             "PIL", "sklearn", "scipy"]:
    try:
        mod = importlib.import_module(name)
        print(f"  ok   {name:16s} {getattr(mod, '__version__', '')}")
    except ImportError as e:
        print(f"  FAIL {name:16s} {e}")
        ok = False

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")
key = os.environ.get("YOUTUBE_API_KEY", "")
if not (ROOT / ".env").exists():
    print("  FAIL .env not found in the repo root (cp .env.example .env)")
    ok = False
elif not key or key.startswith("paste_"):
    print("  FAIL YOUTUBE_API_KEY is empty or still the placeholder in .env")
    ok = False
elif key != key.strip():
    print("  FAIL YOUTUBE_API_KEY has spaces or a Windows line ending (set LF in VS Code)")
    ok = False
else:
    print(f"  ok   API key present ({len(key)} characters, value not shown)")   # length only, never the key

local = ROOT / "channels.local.yaml"
print(f"  info channels.local.yaml {'exists' if local.exists() else 'not created yet'}")

print("T0 PASSED" if ok else "T0 FAILED")
sys.exit(0 if ok else 1)

