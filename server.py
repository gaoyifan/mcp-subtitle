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
import concurrent.futures
import multiprocessing

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
# Global dictionary to track running tasks: (url, language) -> asyncio.Task
TASKS: Dict[tuple[str, str], asyncio.Task] = {}

# Global process pool for sequential execution
_PROCESS_POOL = None


def find_audio_files(path, extensions):
    return [f for f in os.listdir(path) if any(f.endswith(ext) for ext in extensions)]


def find_audio_format_id(info):
    return "bestaudio[abr<144]/bestaudio"  # Limit bitrate to 144kbps


def _get_cache_path(url: str, language: str = "auto") -> Path:
    url_hash = hashlib.md5(f"{url}|{language}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{url_hash}.json"


def _worker_process(url: str, language: str) -> str:
    """
    Synchronous worker function that runs in a separate process.
    Handles downloading, transcribing, and caching.
    """
    # Re-configure logging for the child process
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"worker-{os.getpid()}")
    logger.info(f"Processing URL: {url} (lang={language})")

    try:
        sub_preferences_en = ["en", "en-US", "en-GB", "en-AU", "en-CA", "en-IN", "en-IE"]
        sub_preferences_zh = ["zh-CN", "zh-Hans", "zh", "zh-Hant", "zh-TW", "zh-HK", "zh-SG"]
        autosub_preferences = ["en"]
        
        # Common options
        common_options = {
            "remote_components": {"ejs:github"},
        }
        
        # 1. Extract Info
        with yt_dlp.YoutubeDL(common_options.copy()) as ydl:
            info = ydl.extract_info(url, download=False, process=False)

        # 2. Determine Language
        if language != "auto":
            guessed_language = language
            if language in sub_preferences_zh:
                sub_preferences = sub_preferences_zh + sub_preferences_en
            else:
                sub_preferences = sub_preferences_en + sub_preferences_zh
        else:
            if "title" in info and len([c for c in info["title"] if ord(c) in range(0x3400, 0xA000)]) >= 5:
                sub_preferences = sub_preferences_zh + sub_preferences_en
                guessed_language = "zh"
            else:
                sub_preferences = sub_preferences_en + sub_preferences_zh
                guessed_language = "en"

        whisper_language = "zh" if guessed_language in sub_preferences_zh else None
        
        # 3. Check Existing Subtitles
        subtitle = None
        available_subs = info.get("subtitles", {})
        available_autosubs = info.get("automatic_captions", {})

        # Manual subs
        for lang in sub_preferences:
            if lang in available_subs:
                subtitle = "sub", lang
                break
        if subtitle is None:
            for lang in available_subs:
                if lang != "live_chat":
                    subtitle = "sub", lang
                    break
        
        # Auto subs
        if subtitle is None:
            for lang in available_autosubs:
                if lang in autosub_preferences:
                    subtitle = "autosub", lang
                    break
            if subtitle is None and available_autosubs:
                first_lang = next(iter(available_autosubs))
                subtitle = "autosub", first_lang

        # 4. Download or Transcribe
        subtitle_content = ""
        
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
                    ydl.download([url])

                expected_filename = f"{tmpdir}/output.{subtitle[1]}.json3"
                if not os.path.exists(expected_filename):
                    for f in os.listdir(tmpdir):
                        if f.endswith(".json3"):
                            expected_filename = os.path.join(tmpdir, f)
                            break
                
                if os.path.exists(expected_filename):
                    with open(expected_filename) as f:
                        json3 = json.load(f)
                        subtitle_lines = []
                        if json3 and "events" in json3:
                            for event in json3["events"]:
                                if "segs" in event:
                                    line = "".join([seg["utf8"] for seg in event["segs"] if "utf8" in seg]).strip()
                                    if line:
                                        subtitle_lines.append(line)
                        subtitle_content = "\n".join(subtitle_lines)
                else:
                    logger.warning("Subtitle download failed, falling back to transcription")
                    subtitle = None # Force transcription

        if not subtitle:
            logger.info("Transcribing audio...")
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_options = {
                    **common_options,
                    "format": find_audio_format_id(info),
                    "outtmpl": f"{tmpdir}/audio.%(ext)s",
                    "concurrent_fragment_downloads": 4,
                }
                with yt_dlp.YoutubeDL(audio_options) as ydl:
                    ydl.download([url])

                audio_files = find_audio_files(tmpdir, [".m4a", ".webm", ".mp3", ".mp4"])
                if not audio_files:
                    raise RuntimeError("Failed to download audio")

                audio_path = f"{tmpdir}/{audio_files[0]}"

                duration = info.get("duration") or 0
                condition_on_previous_text = duration <= 1800
                if not condition_on_previous_text:
                    logger.info("Video too long, disabling condition_on_previous_text")

                result = mlx_whisper.transcribe(
                    audio_path,
                    path_or_hf_repo=WHISPER_MODEL,
                    condition_on_previous_text=condition_on_previous_text,
                    language=whisper_language
                )
                
                segments = result.get("segments", [])
                segment_texts = [s.get("text", "").strip() for s in segments]

                # Quality Checks
                has_repetition = any(count > 10 for count in Counter(segment_texts).values())
                blacklist_hits = sum(1 for text in segment_texts for bad_word in BLACKLIST if bad_word in text)
                has_char_repetition = any(re.search(r"(.)\1{19,}", text) for text in segment_texts)

                if (has_repetition or blacklist_hits > 3 or has_char_repetition) and condition_on_previous_text:
                    logger.warning("Bad transcription detected. Retrying with condition_on_previous_text=False")
                    result = mlx_whisper.transcribe(
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

        final_result = f"Title: {info.get('title', 'Unknown')}\nSubtitles:\n{subtitle_content}"
        
        # Cache Result
        cache_path = _get_cache_path(url, language)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False)
        logger.info(f"Cached result to {cache_path}")
        
        return final_result

    except Exception as e:
        logger.error(f"Worker process failed: {e}")
        raise


async def _background_task_wrapper(url: str, language: str):
    """
    Wraps the process pool execution to behave like an asyncio Task.
    Handles cleanup of the TASKS dictionary.
    """
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        # Initialize pool with max_workers=2
        # Use 'spawn' for safe Metal usage
        ctx = multiprocessing.get_context("spawn")
        _PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=ctx)

    loop = asyncio.get_running_loop()
    try:
        # Submit to process pool
        result = await loop.run_in_executor(_PROCESS_POOL, _worker_process, url, language)
        return result
    finally:
        # Cleanup task tracking
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
    
    # Create wrapper task and track it
    task = asyncio.create_task(_background_task_wrapper(url, language))
    TASKS[(url, language)] = task

    try:
        # Shield to allow background completion even if request cancels
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        logger.info(f"Client disconnected for {url}, task continues in background")
        # Ensure we wait on the shielded task so we don't accidentally cancel logic
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
