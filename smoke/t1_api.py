"""T1 API smoke test: one channel → about 50 eligible videos + thumbnails in a temp folder.

Run in the repo root:
  uv run python smoke/t1_api.py                 # first channel in channels.local.yaml
  uv run python smoke/t1_api.py --index 2       # third channel (0, 1, 2, ...)
  uv run python smoke/t1_api.py --handle @name  # any channel, typed on the command line
Uses the real M1 code (m1_collect.py). Writes nothing into the repo: thumbnails go to /tmp.
Quota: 1 unit per API call (about 1 + 2 per 50 uploads looked at). Never calls search.list.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from beyond_click import config, m1_collect  # noqa: E402


class QuotaCounter:
    """Wraps the API client and counts list() calls (each costs 1 quota unit here)."""
    def __init__(self, yt):
        self._yt, self.units = yt, 0

    def __getattr__(self, resource):            # e.g. yt.videos → counted version
        make = getattr(self._yt, resource)
        counter = self

        class _Resource:
            def list(self, **kw):
                counter.units += 1
                return make().list(**kw)
        return lambda: _Resource()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handle", help="channel @handle (default: from channels.local.yaml)")
    ap.add_argument("--index", type=int, default=0, help="which channel of channels.local.yaml")
    ap.add_argument("--n", type=int, default=50, help="eligible videos to collect")
    args = ap.parse_args()

    cfg = config.load()                                  # real mode: reads channels.local.yaml if present
    if args.handle:
        handle = args.handle.strip()
        if not handle.startswith("@"):
            raise SystemExit("The handle must start with @")
    else:
        handle = config.require_real_channels(cfg)[args.index]   # never hard-coded in this script

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key or key.startswith("paste_"):
        raise SystemExit("No API key in .env (run smoke/t0_env.py)")

    import isodate
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from PIL import Image
    import pandas as pd

    tmp = Path(tempfile.mkdtemp(prefix="t1_thumbs_"))   # outside the repo
    yt = QuotaCounter(build("youtube", "v3", developerKey=key, cache_discovery=False))
    now = pd.Timestamp.now(tz="UTC")
    rows, reasons, looked = [], {}, 0
    try:
        title, uploads = m1_collect._channel_uploads(yt, handle)
        for v in m1_collect._videos_newest_first(yt, uploads, cfg, now):
            if len(rows) >= args.n:
                break
            looked += 1
            reason = m1_collect._skip_reason(v, cfg, now, isodate)
            path = tmp / f"{v['id']}.jpg"
            if reason is None:
                url = v["snippet"]["thumbnails"].get("medium", {}).get("url") \
                    or f"https://i.ytimg.com/vi/{v['id']}/mqdefault.jpg"
                if not m1_collect._download(url, path):
                    reason = "thumbnail download failed"
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            rows.append(dict(video_id=v["id"], published_at=v["snippet"]["publishedAt"][:10],
                             duration_s=int(isodate.parse_duration(v["contentDetails"]["duration"]).total_seconds()),
                             views=int(v["statistics"]["viewCount"]),
                             size="x".join(map(str, Image.open(path).size))))
    except HttpError as e:
        raise SystemExit(m1_collect._explain_api_error(e))

    df = pd.DataFrame(rows)
    print(f"channel: {title}")
    print(f"uploads looked at: {looked} · kept: {len(df)} · skipped: {sum(reasons.values())}")
    print(f"skip reasons: {reasons}")
    live = reasons.get("live stream or premiere", 0)
    print(f"live/premiere share of looked-at uploads: {live / max(looked, 1):.0%}")
    if len(df):
        print(f"kept dates: {df.published_at.min()} … {df.published_at.max()} · "
              f"median views {int(df.views.median()):,} · thumbnail sizes {df['size'].value_counts().to_dict()}")
        print(df.head(5).to_string(index=False))
    print(f"thumbnails in {tmp} ({len(list(tmp.glob('*.jpg')))} files)")
    print(f"quota used: about {yt.units} units (daily limit 10,000)")
    enough = len(df) >= cfg["min_videos_per_channel"]
    print("T1 PASSED" if enough else f"T1 WARNING: fewer than {cfg['min_videos_per_channel']} eligible videos "
          "on this channel (it would be dropped in M2)")


if __name__ == "__main__":
    main()

    