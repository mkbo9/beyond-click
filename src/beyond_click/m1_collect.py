#draft

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
        raise ValueError(f"Channel not found: {handle} (check the @handle)")
    item = r["items"][0]
    return item["snippet"]["title"], item["contentDetails"]["relatedPlaylists"]["uploads"]


def _video_ids(yt, playlist, max_ids):
    # 
    ids, token = [], None
    while len(ids) < max_ids:
        r = yt.playlistItems().list(part="contentDetails", playlistId=playlist,
                                    maxResults=50, pageToken=token).execute()
        ids += [it["contentDetails"]["videoId"] for it in r["items"]]
        token = r.get("nextPageToken")
        if not token:
            break
    return ids[:max_ids]


def _details(yt, ids):
    # 
    out = []
    for i in range(0, len(ids), 50):
        r = yt.videos().list(part="snippet,contentDetails,statistics,liveStreamingDetails",
                             id=",".join(ids[i:i + 50])).execute()
        out += r["items"]
    return out


def _download(url, path, tries=3):
    for k in range(tries):                      # retry
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
        return "live stream"
    if isodate.parse_duration(cd["duration"]).total_seconds() < cfg["min_duration_s"]:
        return "short video"                    # 
    if "#shorts" in sn["title"].lower():
        return "#shorts in title"
    if not (cfg["min_age_months"] <= age_months <= cfg["max_age_months"]):
        return "outside age window"             # 
    if "viewCount" not in st:
        return "views hidden"
    return None


def run(cfg) -> pd.DataFrame:
    if cfg["mode"] == "fake":
        from . import fake_data
        df = fake_data.make(cfg)
        df.to_csv(cfg["interim"] / "videos.csv", index=False)
        print(f"  fake data: {len(df)} thumbnails from {df.channel.nunique()} channels")
        return df

    import isodate
    from googleapiclient.discovery import build
    load_dotenv(cfg["root"] / ".env")
    yt = build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])  # key only from .env
    now = pd.Timestamp.now(tz="UTC")
    rows, skipped = [], []
    for handle in cfg["channels"]:
        title, uploads = _channel_uploads(yt, handle)
        kept, reasons = 0, {}
        for v in _details(yt, _video_ids(yt, uploads, cfg["max_ids_per_channel"])):
            if kept >= cfg["max_videos_per_channel"]:
                break
            reason = _skip_reason(v, cfg, now, isodate)
            path = cfg["thumbs"] / f"{v['id']}.jpg"
            if reason is None and not path.exists() and not _download(v["snippet"]["thumbnails"]["medium"]["url"], path):
                reason = "thumbnail download failed"
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            sn = v["snippet"]
            rows.append(dict(video_id=v["id"], channel_handle=handle, channel=title, title=sn["title"],
                             published_at=pd.Timestamp(sn["publishedAt"]),
                             duration_s=int(isodate.parse_duration(v["contentDetails"]["duration"]).total_seconds()),
                             views=int(v["statistics"]["viewCount"]), thumb_path=rel_path(cfg, path),
                             collected_at=now))   # raw API data
            kept += 1
        skipped += [dict(channel_handle=handle, reason=r, count=n) for r, n in reasons.items()]
        print(f"  {handle}: kept {kept} · skipped {reasons}")
    df = pd.DataFrame(rows)
    df.to_csv(cfg["interim"] / "videos.csv", index=False)
    pd.DataFrame(skipped, columns=["channel_handle", "reason", "count"]).to_csv(
        cfg["logs"] / f"collect_{now:%Y-%m-%d}.csv", index=False)
    print(f"  total {len(df)} thumbnails")
    return df






