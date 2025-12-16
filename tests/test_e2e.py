import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastmcp import Client
from server import mcp

@pytest.mark.asyncio
async def test_get_subtitles_existing():
    # Test with a video that likely has subtitles (manual or auto)
    # URL 1: https://www.youtube.com/watch?v=VQRLujxTm3c
    # In-memory testing: pass the mcp object directly
    
    async with Client(mcp) as client:
        result = await client.call_tool("get_subtitles", arguments={"url": "https://www.youtube.com/watch?v=VQRLujxTm3c"})
        assert result is not None
        # result is likely CallToolResult, access content
        content = result.content
        assert isinstance(content, list) 
        assert len(content) > 0
        text = content[0].text
        print(f"Result length: {len(text)}")
        print(text)
        
        # Proper assertion based on video content
        expected_start = "What you doing up there?"
        assert expected_start in text, f"Expected '{expected_start}' in subtitle text"

@pytest.mark.asyncio
async def test_get_subtitles_transcription():
    # Test with a video that requires transcription/auto-generated
    # URL 2: https://www.youtube.com/watch?v=5cqaHCQ4pi4
    
    async with Client(mcp) as client:
        result = await client.call_tool("get_subtitles", arguments={"url": "https://www.youtube.com/watch?v=5cqaHCQ4pi4"})
        assert result is not None
        content = result.content
        text = content[0].text
        assert len(text) > 0
        print(f"Result length: {len(text)}")
        print(text)
        
        # Proper assertion based on video content (transcription)
        expected_start = "大家好"
        assert expected_start in text, f"Expected '{expected_start}' to be present (start of transcription)"
