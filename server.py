import json
import logging
import os
import tempfile
import asyncio
from typing import Annotated

from fastmcp import FastMCP
import yt_dlp
import mlx_whisper

# Initialize FastMCP server
mcp = FastMCP("subtitle-server")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_sync_in_executor(func, *args, **kwargs):
    import asyncio
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))

def find_audio_files(path, extensions):
    return [f for f in os.listdir(path) if any(f.endswith(ext) for ext in extensions)]

def find_audio_format_id(info):
    return "bestaudio[filesize<20M]" # Limit size for faster processing

@mcp.tool()
async def get_subtitles(url: Annotated[str, "URL of the YouTube video."]) -> str:
    """
    Extracts subtitles from a video.
    First attempts to download existing subtitles (manual, then auto-generated).
    If none are available, transcribes the audio using mlx-whisper.
    """
    logger.info(f"Processing URL: {url}")
    
    sub_preferences_en = ["en", "en-US", "en-GB", "en-AU", "en-CA", "en-IN", "en-IE"]
    sub_preferences_zh = ["zh-CN", "zh-Hans", "zh", "zh-Hant", "zh-TW", "zh-HK", "zh-SG"]
    autosub_preferences = ["en"]

    with yt_dlp.YoutubeDL() as ydl:
        # Extract info without downloading
        info = await run_sync_in_executor(ydl.extract_info, url, download=False, process=False)

    # Determine language preference based on title characters
    if "title" in info and len([c for c in info["title"] if ord(c) in range(0x3400, 0xA000)]) >= 5:
        sub_preferences = sub_preferences_zh + sub_preferences_en
        logger.info("Guessed language: zh")
    else:
        sub_preferences = sub_preferences_en + sub_preferences_zh
        logger.info("Guessed language: en")

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
        for lang in available_autosubs: # Check available auto subs against checking list? Or just take any? 
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
            "format": find_audio_format_id(info), 
            "outtmpl": f"{tmpdir}/audio.%(ext)s",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }]
        }
        with yt_dlp.YoutubeDL(audio_options) as ydl:
            await run_sync_in_executor(ydl.download, [url])
            
        audio_files = find_audio_files(tmpdir, [".m4a", ".mp3", ".webm", ".mp4"])
        if not audio_files:
             raise RuntimeError("Failed to download audio")
             
        audio_path = f"{tmpdir}/{audio_files[0]}"
        
        # Use mlx_whisper
        # path_or_uri: str, path_or_uri to the audio file
        # model: str = "mlx-community/whisper-large-v3-mlx"
        try:
             result = mlx_whisper.transcribe(
                 audio_path, 
                 path_or_hf_repo="mlx-community/whisper-large-v3-mlx"
             )
             # Use segments for better line break handling
             segments = result.get("segments", [])
             if not segments and "text" in result:
                 subtitle_content = result["text"]
             else:
                 subtitle_content = "\n".join(seg["text"].strip() for seg in segments)
             
             return f"Title: {info.get('title', 'Unknown')}\nSubtitles:\n{subtitle_content}"
        except Exception as e:
             logger.error(f"Transcription failed: {e}")
             raise RuntimeError(f"Transcription failed: {str(e)}")

def main():
    mcp.run()

# This allows running the server directly
if __name__ == "__main__":
    main()
