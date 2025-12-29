import json
import logging
import os
import tempfile
import re
import asyncio
import hashlib
from pathlib import Path
from collections import Counter
from typing import Annotated, Dict

import mlx_whisper
import yt_dlp
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("subtitle-server")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BLACKLIST = [
    "明镜与点点",
    "订阅明镜",
    "字幕志愿者",
    "优优独播剧场",
    "汇聚精英",
    "字幕提供者",
]

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
CACHE_DIR = Path(os.getenv("MCP_SUBTITLE_CACHE_DIR", str(Path.home() / ".cache" / "mcp-subtitle")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global dictionary to track running tasks: (url, language) -> asyncio.Task
TASKS: Dict[tuple[str, str], asyncio.Task] = {}
# Global semaphore to limit concurrent background tasks
SEM = asyncio.Semaphore(2)


def run_sync_in_executor(func, *args, **kwargs):
    import asyncio

    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


def find_audio_files(path, extensions):
    return [f for f in os.listdir(path) if any(f.endswith(ext) for ext in extensions)]


def find_audio_format_id(info):
    return "bestaudio[abr<144]/bestaudio"  # Limit bitrate to 144kbps


def _get_cache_path(url: str, language: str = "auto") -> Path:
    url_hash = hashlib.md5(f"{url}|{language}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{url_hash}.json"


async def _fetch_subtitles_impl(url: str, language: str = "auto") -> str:
    """
    Internal implementation of subtitle extraction.
    """
    logger.info(f"Processing URL: {url}")

    sub_preferences_en = ["en", "en-US", "en-GB", "en-AU", "en-CA", "en-IN", "en-IE"]
    sub_preferences_zh = ["zh-CN", "zh-Hans", "zh", "zh-Hant", "zh-TW", "zh-HK", "zh-SG"]
    autosub_preferences = ["en"]

    # Common options to allow JS challenge solving
    common_options = {
        "remote_components": {"ejs:github"},
    }

    with yt_dlp.YoutubeDL(common_options.copy()) as ydl:
        # Extract info without downloading
        info = await run_sync_in_executor(ydl.extract_info, url, download=False, process=False)

    # Determine language preference
    if language != "auto":
        guessed_language = language
        if language in sub_preferences_zh:
            sub_preferences = sub_preferences_zh + sub_preferences_en
        else:
            sub_preferences = sub_preferences_en + sub_preferences_zh
        logger.info(f"Using provided language: {language}")
    else:
        # Determine language preference based on title characters
        if "title" in info and len([c for c in info["title"] if ord(c) in range(0x3400, 0xA000)]) >= 5:
            sub_preferences = sub_preferences_zh + sub_preferences_en
            guessed_language = "zh"
        else:
            sub_preferences = sub_preferences_en + sub_preferences_zh
            guessed_language = "en"
        logger.info(f"Guessed language: {guessed_language}")

    # For Whisper transcription, we force "zh" if the language is Chinese-like.
    # Otherwise, we leave it as None for automatic detection per requirements.
    whisper_language = "zh" if guessed_language in sub_preferences_zh else None

    subtitle = None

    # helper to check available subs
    available_subs = info.get("subtitles", {})
    available_autosubs = info.get("automatic_captions", {})

    # 1. Check manual subtitles
    for lang in sub_preferences:
        if lang in available_subs:
            subtitle = "sub", lang
            break

    # 2. Check other manual subtitles if preferred not found
    if subtitle is None:
        for lang in available_subs:
            if lang != "live_chat":
                subtitle = "sub", lang
                break

    # 3. Check auto subtitles
    if subtitle is None:
        for lang in available_autosubs:  # Check available auto subs against checking list? Or just take any?
            # Legacy code checked against autosub_preferences (en)
            if lang in autosub_preferences:
                subtitle = "autosub", lang
                break
        # If preferred auto sub not found, maybe just take the first one?
        if subtitle is None and available_autosubs:
            first_lang = next(iter(available_autosubs))
            subtitle = "autosub", first_lang

    if subtitle:
        logger.info(f"Found subtitle: {subtitle}")
        with tempfile.TemporaryDirectory() as tmpdir:
            options = {
                **common_options,
                "outtmpl": f"{tmpdir}/output.%(ext)s",
                "skip_download": True,
                "subtitleslangs": [subtitle[1]],
                "subtitlesformat": "json3",
            }
            if subtitle[0] == "sub":
                options["writesubtitles"] = True
            elif subtitle[0] == "autosub":
                options["writeautomaticsub"] = True

            with yt_dlp.YoutubeDL(options) as ydl:
                await run_sync_in_executor(ydl.download, [url])

            # Read the downloaded subtitle file
            # yt-dlp might name it output.lang.json3
            expected_filename = f"{tmpdir}/output.{subtitle[1]}.json3"
            if not os.path.exists(expected_filename):
                # fallback search
                files = os.listdir(tmpdir)
                for f in files:
                    if f.endswith(".json3"):
                        expected_filename = os.path.join(tmpdir, f)
                        break

            if os.path.exists(expected_filename):
                with open(expected_filename) as f:
                    json3 = json.load(f)
                    subtitle_lines = []
                    # Check if 'events' key exists and is iterable
                    if json3 and "events" in json3:
                        for event in json3["events"]:
                            if "segs" in event:
                                line = "".join([seg["utf8"] for seg in event["segs"] if "utf8" in seg]).strip()
                                if line:
                                    subtitle_lines.append(line)
                    subtitle_content = "\n".join(subtitle_lines)
                    return f"Title: {info.get('title', 'Unknown')}\nSubtitles:\n{subtitle_content}"
            else:
                logger.warning("Subtitle download failed or file not found. Falling back to transcription.")

    # 4. Transcribe
    logger.info("Transcribing audio...")
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_options = {
            **common_options,
            "format": find_audio_format_id(info),
            "outtmpl": f"{tmpdir}/audio.%(ext)s",
            "concurrent_fragment_downloads": 4,
        }
        with yt_dlp.YoutubeDL(audio_options) as ydl:
            await run_sync_in_executor(ydl.download, [url])

        audio_files = find_audio_files(tmpdir, [".m4a", ".webm", ".mp3", ".mp4"])
        if not audio_files:
            raise RuntimeError("Failed to download audio")

        audio_path = f"{tmpdir}/{audio_files[0]}"

        # Use mlx_whisper
        # path_or_uri: str, path_or_uri to the audio file
        # model: str = "mlx-community/whisper-large-v3-mlx"
        try:
            duration = info.get("duration") or 0
            if duration > 1800:
                logger.info(f"Video duration {duration}s > 30m, disabling condition_on_previous_text")
                enable_previous_text = False
            else:
                enable_previous_text = True

            # Run transcription in executor to avoid blocking event loop
            result = await run_sync_in_executor(
                mlx_whisper.transcribe,
                audio_path,
                path_or_hf_repo=WHISPER_MODEL,
                condition_on_previous_text=enable_previous_text,
                language=whisper_language
            )
            # Use segments for better line break handling
            segments = result.get("segments", [])

            # Check quality
            segment_texts = [s.get("text", "").strip() for s in segments]

            # Check 1: Repetitions
            text_counts = Counter(segment_texts)
            has_repetition = any(count > 10 for count in text_counts.values())

            # Check 2: Blacklist
            blacklist_hits = 0
            for text in segment_texts:
                for bad_word in BLACKLIST:
                    if bad_word in text:
                        blacklist_hits += 1

            # Check 3: Character Repetition
            has_char_repetition = False
            for text in segment_texts:
                if re.search(r"(.)\1{19,}", text):
                    has_char_repetition = True
                    break

            if (has_repetition or blacklist_hits > 3 or has_char_repetition) and enable_previous_text:
                logger.warning(
                    f"Bad transcription detected (repetition={has_repetition}, blacklist_hits={blacklist_hits}, char_repetition={has_char_repetition}). Retrying with condition_on_previous_text=False"
                )
                # Run retry in executor as well
                result = await run_sync_in_executor(
                    mlx_whisper.transcribe,
                    audio_path,
                    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                    condition_on_previous_text=False,
                    language=whisper_language
                )
                segments = result.get("segments", [])

            if not segments and "text" in result:
                subtitle_content = result["text"]
            else:
                subtitle_content = "\n".join(seg["text"].strip() for seg in segments)

            return f"Title: {info.get('title', 'Unknown')}\nSubtitles:\n{subtitle_content}"
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {str(e)}")


async def _background_worker(url: str, language: str = "auto"):
    """
    Background task wrapper that handles caching and cleanup.
    Limit concurrent tasks using SEM.
    """
    try:
        async with SEM:
            result = await _fetch_subtitles_impl(url, language)
            # Json cache if successful
            cache_path = _get_cache_path(url, language)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
            logger.info(f"Cached result for {url} to {cache_path}")
            return result
    except Exception as e:
        logger.error(f"Background task failed for {url}: {e}")
        raise e
    finally:
        # Cleanup task
        if (url, language) in TASKS:
            del TASKS[(url, language)]


@mcp.tool()
async def get_subtitles(
    url: Annotated[str, "URL of the YouTube video."],
    language: Annotated[str, "Language for subtitles (default: auto)."] = "auto",
    use_cache: Annotated[bool, "Whether to use cached results if available."] = True,
) -> str:
    """
    Extracts subtitles from a video.
    First attempts to download existing subtitles (manual, then auto-generated).
    If none are available, transcribes the audio using mlx-whisper.
    """
    # 1. Check cache
    cache_path = _get_cache_path(url, language)
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                logger.info(f"Serving from cache: {url} (lang: {language})")
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache for {url}: {e}")

    # 2. Check for running task
    if (url, language) in TASKS:
        logger.info(f"Joining existing task for {url} (lang: {language})")
        return await TASKS[(url, language)]

    # 3. Start new task
    logger.info(f"Starting new background task for {url} (lang: {language})")
    # Create the task but don't await it immediately in a way that cancellation kills the task
    # We want the task to survive cancellation.
    # We use await asyncio.shield(task) to protect the task.
    task = asyncio.create_task(_background_worker(url, language))
    TASKS[(url, language)] = task

    try:
        return await task
    except asyncio.CancelledError:
        logger.info(f"Client disconnected for {url}, task continues in background")
        # Ensure the task continues running even if this await is cancelled
        # If we cancel here, we MUST not cancel the task.
        # However, await task will propagate cancellation to the task if we don't shield it.
        # Wait, if I await task, and I get cancelled, the expected behavior is that the task gets cancelled.
        # To prevent this, requests handlers MUST use shield.
        return await asyncio.shield(task)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MCP Subtitle Server")
    parser.add_argument("--sse", action="store_true", help="Run in SSE mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (SSE mode only)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (SSE mode only)")

    args = parser.parse_args()

    if args.sse:
        logger.info(f"Starting SSE server on {args.host}:{args.port}")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run()


# This allows running the server directly
if __name__ == "__main__":
    main()
