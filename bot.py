"""
Best Lecture Finder — Telegram Bot
-----------------------------------
/best <topic>  ->  Searches YouTube for the topic, pulls 20-50 candidate
videos, scores them on relevance + views + likes + freshness + duration +
channel quality, and returns the top 1-3 matches.

Setup:
  1. pip install -r requirements.txt
  2. Set two environment variables (or edit the constants below):
       TELEGRAM_BOT_TOKEN   - from @BotFather
       YOUTUBE_API_KEY      - from Google Cloud Console (YouTube Data API v3)
  3. python bot.py

Notes:
  - Free YouTube Data API quota is 10,000 units/day.
    One /best call costs roughly: 100 (search) + ~1 (videos.list, batched)
    + ~1 (channels.list, batched) = ~102 units. That's about 95 searches/day
    on the free quota. Adjust MAX_RESULTS below if you need to conserve it.
"""

import os
import re
import math
import logging
from datetime import datetime, timezone

import requests
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PUT_YOUR_TELEGRAM_TOKEN_HERE")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "PUT_YOUR_YOUTUBE_API_KEY_HERE")

MAX_RESULTS = 40          # how many candidate videos to pull (20-50 recommended)
TOP_N = 3                 # how many best videos to return
IDEAL_MIN_MINUTES = 8      # duration sweet spot (lectures)
IDEAL_MAX_MINUTES = 45
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("best-lecture-bot")


# ---------------------------------------------------------------------------
# YouTube helpers
# ---------------------------------------------------------------------------

def search_videos(query: str, max_results: int = MAX_RESULTS) -> list[str]:
    """Return a list of video IDs for the given query, most relevant first."""
    video_ids = []
    page_token = None
    while len(video_ids) < max_results:
        params = {
            "part": "id",
            "q": query,
            "type": "video",
            "maxResults": min(50, max_results - len(video_ids)),
            "relevanceLanguage": "hi",
            "safeSearch": "moderate",
            "key": YOUTUBE_API_KEY,
        }
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid:
                video_ids.append(vid)
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return video_ids


def fetch_video_details(video_ids: list[str]) -> list[dict]:
    """Batch-fetch stats/snippet/contentDetails for up to 50 IDs at a time."""
    details = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(chunk),
            "key": YOUTUBE_API_KEY,
        }
        resp = requests.get(YOUTUBE_VIDEOS_URL, params=params, timeout=15)
        resp.raise_for_status()
        details.extend(resp.json().get("items", []))
    return details


def fetch_channel_subscribers(channel_ids: list[str]) -> dict[str, int]:
    """Return {channel_id: subscriber_count} for up to 50 channels at a time."""
    subs = {}
    unique_ids = list(dict.fromkeys(channel_ids))
    for i in range(0, len(unique_ids), 50):
        chunk = unique_ids[i:i + 50]
        params = {
            "part": "statistics",
            "id": ",".join(chunk),
            "key": YOUTUBE_API_KEY,
        }
        resp = requests.get(YOUTUBE_CHANNELS_URL, params=params, timeout=15)
        resp.raise_for_status()
        for item in resp.json().get("items", []):
            stats = item.get("statistics", {})
            if not stats.get("hiddenSubscriberCount"):
                subs[item["id"]] = int(stats.get("subscriberCount", 0))
            else:
                subs[item["id"]] = 0
    return subs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def parse_iso8601_duration(duration: str) -> int:
    """Convert ISO 8601 duration (e.g. PT15M33S) to seconds."""
    match = re.match(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration or ""
    )
    if not match:
        return 0
    h, m, s = (int(x) if x else 0 for x in match.groups())
    return h * 3600 + m * 60 + s


def relevance_score(query: str, title: str, description: str) -> float:
    """Very light keyword-overlap relevance score (0-1)."""
    q_words = set(re.findall(r"\w+", query.lower()))
    if not q_words:
        return 0.0
    text = f"{title} {description}".lower()
    hits = sum(1 for w in q_words if w in text)
    title_bonus = 0.3 if query.lower().strip() in title.lower() else 0.0
    return min(1.0, hits / len(q_words) + title_bonus)


def duration_score(seconds: int) -> float:
    """1.0 inside the ideal lecture-length window, decaying outside it."""
    minutes = seconds / 60
    if IDEAL_MIN_MINUTES <= minutes <= IDEAL_MAX_MINUTES:
        return 1.0
    if minutes < IDEAL_MIN_MINUTES:
        return max(0.0, minutes / IDEAL_MIN_MINUTES)
    # longer than ideal: gentle decay, floor at 0.3
    overflow = minutes - IDEAL_MAX_MINUTES
    return max(0.3, 1.0 - overflow / 120)


def freshness_score(published_at: str) -> float:
    """Newer videos score higher; decays over ~5 years."""
    try:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.5
    age_days = (datetime.now(timezone.utc) - published).days
    return max(0.0, 1.0 - age_days / (365 * 5))


def log_scale(value: int, cap: int) -> float:
    """0-1 normalized log scale, capped at `cap` for the top score."""
    if value <= 0:
        return 0.0
    return min(1.0, math.log10(value + 1) / math.log10(cap + 1))


def score_video(query: str, item: dict, subscriber_count: int) -> dict:
    snippet = item["snippet"]
    stats = item.get("statistics", {})
    content = item.get("contentDetails", {})

    views = int(stats.get("viewCount", 0))
    likes = int(stats.get("likeCount", 0)) if "likeCount" in stats else 0
    duration_sec = parse_iso8601_duration(content.get("duration", ""))

    rel = relevance_score(query, snippet.get("title", ""), snippet.get("description", ""))
    views_s = log_scale(views, cap=5_000_000)
    like_ratio = (likes / views) if views > 0 else 0
    likes_s = min(1.0, like_ratio * 40)  # ~2.5% like ratio maxes this out
    fresh_s = freshness_score(snippet.get("publishedAt", ""))
    dur_s = duration_score(duration_sec)
    channel_s = log_scale(subscriber_count, cap=2_000_000)

    weights = {
        "relevance": 0.30,
        "views": 0.20,
        "likes": 0.15,
        "freshness": 0.15,
        "duration": 0.10,
        "channel": 0.10,
    }
    total = (
        rel * weights["relevance"]
        + views_s * weights["views"]
        + likes_s * weights["likes"]
        + fresh_s * weights["freshness"]
        + dur_s * weights["duration"]
        + channel_s * weights["channel"]
    )

    return {
        "id": item["id"],
        "title": snippet.get("title", ""),
        "channel": snippet.get("channelTitle", ""),
        "views": views,
        "published": snippet.get("publishedAt", "")[:10],
        "duration_sec": duration_sec,
        "match_pct": round(total * 100),
        "url": f"https://www.youtube.com/watch?v={item['id']}",
    }


def find_best_lectures(query: str, top_n: int = TOP_N) -> list[dict]:
    video_ids = search_videos(query)
    if not video_ids:
        return []
    details = fetch_video_details(video_ids)
    channel_ids = [d["snippet"]["channelId"] for d in details if "snippet" in d]
    sub_counts = fetch_channel_subscribers(channel_ids)

    scored = [
        score_video(query, item, sub_counts.get(item["snippet"]["channelId"], 0))
        for item in details
    ]
    scored.sort(key=lambda v: v["match_pct"], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

def format_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def format_views(views: int) -> str:
    if views >= 1_000_000:
        return f"{views / 1_000_000:.1f}M"
    if views >= 1_000:
        return f"{views / 1_000:.1f}K"
    return str(views)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 नमस्ते! मुझे कोई topic भेजिए इस format में:\n\n"
        "/best संविधान की प्रस्तावना\n\n"
        "मैं YouTube पर 20-50 videos check करके आपको best 1-3 lectures दूँगा।"
    )


async def best(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text("कृपया topic भी लिखें, जैसे:\n/best संविधान की प्रस्तावना")
        return

    status_msg = await update.message.reply_text("🔎 Searching...")

    try:
        results = find_best_lectures(query)
    except requests.HTTPError as e:
        logger.exception("YouTube API error")
        await status_msg.edit_text(
            "⚠️ YouTube API से जवाब नहीं मिला। API key / quota चेक करें।\n"
            f"Detail: {e}"
        )
        return
    except Exception:
        logger.exception("Unexpected error")
        await status_msg.edit_text("⚠️ कुछ गड़बड़ हो गई, दोबारा कोशिश करें।")
        return

    if not results:
        await status_msg.edit_text(f"❌ '{query}' के लिए कोई result नहीं मिला।")
        return

    lines = ["🏆 *Best Match(es)*\n"]
    medals = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(results):
        lines.append(
            f"{medals[i] if i < len(medals) else '▪️'} 📚 *{r['title']}*\n"
            f"👤 {r['channel']}\n"
            f"👁️ {format_views(r['views'])} views  ⏱ {format_duration(r['duration_sec'])}  "
            f"📅 {r['published']}\n"
            f"🎯 Match: {r['match_pct']}%\n"
            f"▶️ [Watch on YouTube]({r['url']})\n"
        )

    await status_msg.edit_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=False,
    )


def main() -> None:
    if "PUT_YOUR" in TELEGRAM_BOT_TOKEN or "PUT_YOUR" in YOUTUBE_API_KEY:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN और YOUTUBE_API_KEY set करें "
            "(environment variables के रूप में, या bot.py में सीधे भरें)."
        )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("best", best))

    logger.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
