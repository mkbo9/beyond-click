"""M1 Collect: YouTube Data API → data/interim/videos.csv + data/raw/<video_id>.jpg

Writes (docs/INTERFACES.md): videos.csv with video_id, channel_handle, channel, title,
published_at, duration_s, views, thumb_path, collected_at · logs/collect_<date>.csv
API cost: about 2 quota units per 50 videos looked at. Never uses search.list (100 units per call).
"""
import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

from .config import rel_path


def _channel_uploads(yt, handle):
    # 1 unit: channel handle (@name) → channel title + its "uploads" playlist id
    r = yt.channels().list(part="contentDetails,snippet", forHandle=handle).execute()
    if not r.get("items"):
        raise SystemExit(f"Channel not found: {handle}. Check the @handle in config.yaml.")
    item = r["items"][0]
    return item["snippet"]["title"], item["contentDetails"]["relatedPlaylists"]["uploads"]


def _videos_newest_first(yt, uploads, cfg, now):
    """Yield video details page by page (50 per page), newest first.

    Stops when the uploads are older than the age window or max_ids_per_channel is reached,
    so channels with many Shorts or daily uploads still give enough eligible videos.
    """
    token, seen = None, 0
    while seen < cfg["max_ids_per_channel"]:
        page = yt.playlistItems().list(part="contentDetails", playlistId=uploads,
                                       maxResults=50, pageToken=token).execute()          # 1 unit
        ids = [it["contentDetails"]["videoId"] for it in page["items"]]
        if not ids:
            return
        seen += len(ids)
        details = yt.videos().list(part="snippet,contentDetails,statistics,liveStreamingDetails",
                                   id=",".join(ids)).execute()["items"]                    # 1 unit for 50 videos
        details.sort(key=lambda v: v["snippet"]["publishedAt"], reverse=True)
        yield from details
        token = page.get("nextPageToken")
        oldest = min(pd.Timestamp(v["snippet"]["publishedAt"]) for v in details) if details else now
        if not token or (now - oldest).days / 30.4 > cfg["max_age_months"]:
            return                                    # everything further back is too old


def _download(url, path, tries=3):
    for k in range(tries):                      # retry: thumbnail servers sometimes time out
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            return True
        except requests.RequestException:
            time.sleep(1 + k)
    return False


def _skip_reason(v, cfg, now, isodate):
    """Eligibility rules from the plan. Returns None if the video is kept."""
    sn, cd, st = v["snippet"], v["contentDetails"], v.get("statistics", {})
    age_months = (now - pd.Timestamp(sn["publishedAt"])).days / 30.4
    if "liveStreamingDetails" in v:
        return "live stream or premiere"        # stream archives look different; premieres are dropped too
    if isodate.parse_duration(cd["duration"]).total_seconds() < cfg["min_duration_s"]:
        return "short video"                    # Shorts can be up to 180 s
    if "#shorts" in sn["title"].lower():
        return "#shorts in title"
    if age_months < cfg["min_age_months"]:
        return "too new"                        # views still growing
    if age_months > cfg["max_age_months"]:
        return "too old"
    if "viewCount" not in st:
        return "views hidden"
    return None


def _explain_api_error(e) -> str:
    text = str(e)
    if "API key not valid" in text or "API_KEY_INVALID" in text:
        return "The API key is not valid. Copy it again from Google Cloud → Credentials into .env."
    if "accessNotConfigured" in text or "has not been used" in text or "is disabled" in text:
        return "YouTube Data API v3 is not enabled for your Google Cloud project. Enable it in APIs & Services → Library."
    if "quotaExceeded" in text:
        return "Daily API quota is used up. It resets at midnight Pacific Time."
    return f"YouTube API error: {text}"


def run(cfg) -> pd.DataFrame:
    if cfg["mode"] == "fake":
        from . import fake_data
        df = fake_data.make(cfg)
        df.to_csv(cfg["interim"] / "videos.csv", index=False)
        print(f"  fake data: {len(df)} thumbnails from {df.channel.nunique()} channels")
        return df

    load_dotenv(cfg["root"] / ".env")
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key or key.startswith("paste_"):
        raise SystemExit("No API key: put YOUTUBE_API_KEY=<your key> into the .env file in the repo root.")
    if any("REPLACE_ME" in h for h in cfg["channels"]):
        raise SystemExit("Put your channel @handles into config.yaml → channels.")
    import isodate
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    yt = build("youtube", "v3", developerKey=key)      # the key is only ever read from .env
    now = pd.Timestamp.now(tz="UTC")
    rows, skipped = [], []
    try:
        for handle in cfg["channels"]:
            title, uploads = _channel_uploads(yt, handle)
            kept, reasons = 0, {}
            for v in _videos_newest_first(yt, uploads, cfg, now):
                if kept >= cfg["max_videos_per_channel"]:
                    break
                reason = _skip_reason(v, cfg, now, isodate)
                path = cfg["thumbs"] / f"{v['id']}.jpg"
                if reason is None and not path.exists():
                    # medium = 320×180, no black bars; fall back to the standard URL pattern if missing
                    url = v["snippet"]["thumbnails"].get("medium", {}).get("url") \
                        or f"https://i.ytimg.com/vi/{v['id']}/mqdefault.jpg"
                    if not _download(url, path):
                        reason = "thumbnail download failed"
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                sn = v["snippet"]
                rows.append(dict(video_id=v["id"], channel_handle=handle, channel=title, title=sn["title"],
                                 published_at=pd.Timestamp(sn["publishedAt"]),
                                 duration_s=int(isodate.parse_duration(v["contentDetails"]["duration"]).total_seconds()),
                                 views=int(v["statistics"]["viewCount"]), thumb_path=rel_path(cfg, path),
                                 collected_at=now))   # raw API data: refresh or delete after 30 days
                kept += 1
            skipped += [dict(channel_handle=handle, reason=r, count=n) for r, n in reasons.items()]
            print(f"  {handle}: kept {kept} · skipped {reasons}")
            live = reasons.get("live stream or premiere", 0)
            if live > 0.15 * (kept + sum(reasons.values())):
                print(f"    note: {live} live/premiere videos dropped on this channel; "
                      "if these are mostly normal premieres, tell 00 Master.")
    except HttpError as e:
        raise SystemExit(_explain_api_error(e))
    df = pd.DataFrame(rows)
    df.to_csv(cfg["interim"] / "videos.csv", index=False)
    pd.DataFrame(skipped, columns=["channel_handle", "reason", "count"]).to_csv(
        cfg["logs"] / f"collect_{now:%Y-%m-%d}.csv", index=False)
    print(f"  total {len(df)} thumbnails")
    return df
