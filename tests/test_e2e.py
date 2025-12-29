import os
import sys
import asyncio
import hashlib
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastmcp import Client

from server import mcp, CACHE_DIR

def get_cache_path(url: str) -> Path:
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{url_hash}.json"

@pytest.fixture
def clean_cache():
    # Setup: ensure cache dir exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # No teardown cleanup strictly required, but maybe good practice for specific test files
    # For now, we leave cache for manual inspection if needed, or we can clean specific items.

@pytest.mark.asyncio
async def test_get_subtitles_existing(clean_cache):
    # Test with a video that likely has subtitles (manual or auto)
    url = "https://www.youtube.com/watch?v=VQRLujxTm3c"
    cache_path = get_cache_path(url)
    
    # Ensure starting fresh
    if cache_path.exists():
        cache_path.unlink()

    async with Client(mcp) as client:
        result = await client.call_tool("get_subtitles", arguments={"url": url})
        assert result is not None
        content = result.content
        text = content[0].text
        assert "What you doing up there?" in text

        # Verify cache creation
        assert cache_path.exists()
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        assert cached_data == text

        # Verify second fetch (should ideally be fast/cached)
        result2 = await client.call_tool("get_subtitles", arguments={"url": url})
        assert result2.content[0].text == text

@pytest.mark.asyncio
async def test_get_subtitles_transcription(clean_cache):
    # Test with a video that requires transcription/auto-generated
    url = "https://www.youtube.com/watch?v=5cqaHCQ4pi4"
    cache_path = get_cache_path(url)
    
    # Ensure starting fresh
    if cache_path.exists():
        cache_path.unlink()

    async with Client(mcp) as client:
        result = await client.call_tool("get_subtitles", arguments={"url": url})
        assert result is not None
        text = result.content[0].text
        assert "大家好" in text

        # Verify cache
        assert cache_path.exists()

@pytest.mark.asyncio
async def test_concurrency_limit():
    # Test concurrency limit with multiple requests
    # We use 3 URLs. Since limit is 2, one should wait.
    urls = [
        "https://www.youtube.com/watch?v=VQRLujxTm3c", # Short
        "https://www.youtube.com/watch?v=jNQXAC9IVRw", # Me at the zoo (Short)
        "https://www.youtube.com/watch?v=5cqaHCQ4pi4"  # Transcription one
    ]
    
    async with Client(mcp) as client:
        tasks = [client.call_tool("get_subtitles", arguments={"url": url}) for url in urls]
        results = await asyncio.gather(*tasks)
        
        for res in results:
            assert res is not None
            assert len(res.content) > 0
            assert len(res.content[0].text) > 0
