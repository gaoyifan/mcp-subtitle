import os
import sys
import asyncio
import hashlib
import json
import shutil
from pathlib import Path

# Set test cache directory BEFORE importing server to ensure it takes effect
TEST_CACHE_DIR = Path("/tmp/mcp-subtitle-test")
os.environ["MCP_SUBTITLE_CACHE_DIR"] = str(TEST_CACHE_DIR)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastmcp import Client
from server import mcp, CACHE_DIR

def get_cache_path(url: str, language: str = "auto") -> Path:
    url_hash = hashlib.md5(f"{url}|{language}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{url_hash}.json"

@pytest.fixture
def clean_cache():
    # Setup: ensure cache dir exists and is empty
    if TEST_CACHE_DIR.exists():
        shutil.rmtree(TEST_CACHE_DIR)
    TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Optional Teardown: could clean up here too, but keeping it for inspection is often useful
    # shutil.rmtree(TEST_CACHE_DIR)

@pytest.mark.asyncio
async def test_get_subtitles_existing(clean_cache):
    # Test with a video that likely has subtitles (manual or auto)
    url = "https://www.youtube.com/watch?v=VQRLujxTm3c"
    cache_path = get_cache_path(url)
    
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
    
    async with Client(mcp) as client:
        result = await client.call_tool("get_subtitles", arguments={"url": url})
        assert result is not None
        text = result.content[0].text
        assert "大家好" in text

        # Verify cache
        assert cache_path.exists()

@pytest.mark.asyncio
async def test_get_subtitles_with_language(clean_cache):
    # Test setting language explicitly
    url = "https://www.youtube.com/watch?v=VQRLujxTm3c"
    language = "zh" 
    cache_path = get_cache_path(url, language)
    
    async with Client(mcp) as client:
        result = await client.call_tool("get_subtitles", arguments={"url": url, "language": language})
        assert result is not None
        text = result.content[0].text
        assert len(text) > 0

        # Verify cache creation with language in key
        assert cache_path.exists()
        
        # Verify default language request still works and has different cache
        default_cache = get_cache_path(url, "auto")
        await client.call_tool("get_subtitles", arguments={"url": url})
        assert default_cache.exists()
        assert default_cache != cache_path

@pytest.mark.asyncio
async def test_get_subtitles_bypass_cache(clean_cache):
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    cache_path = get_cache_path(url)
    
    async with Client(mcp) as client:
        # First call to populate cache
        await client.call_tool("get_subtitles", arguments={"url": url})
        assert cache_path.exists()
        
        # Modify cache file to check if it's being used
        with open(cache_path, "w") as f:
            json.dump("Modified Cache Content", f)
            
        # Call with use_cache=True (default)
        result_cached = await client.call_tool("get_subtitles", arguments={"url": url})
        assert result_cached.content[0].text == "Modified Cache Content"
        
        # Call with use_cache=False
        result_fresh = await client.call_tool("get_subtitles", arguments={"url": url, "use_cache": False})
        assert result_fresh.content[0].text != "Modified Cache Content"
        assert "Me at the zoo" in result_fresh.content[0].text

@pytest.mark.asyncio
async def test_concurrency_limit(clean_cache):
    # Test concurrency limit with multiple requests
    urls = [
        "https://www.youtube.com/watch?v=VQRLujxTm3c", 
        "https://www.youtube.com/watch?v=jNQXAC9IVRw", 
        "https://www.youtube.com/watch?v=5cqaHCQ4pi4"  
    ]
    
    async with Client(mcp) as client:
        tasks = [client.call_tool("get_subtitles", arguments={"url": url}) for url in urls]
        results = await asyncio.gather(*tasks)
        
        for res in results:
            assert res is not None
            assert len(res.content) > 0
            assert len(res.content[0].text) > 0
