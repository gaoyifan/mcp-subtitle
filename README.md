# MCP Subtitle Server

An MCP server that extracts subtitles from videos.

## Features

- Downloads existing subtitles from YouTube videos involved (prioritizes manual over auto-generated).
- Falls back to local transcription using `mlx-whisper` if no subtitles are available.

## Usage

### Prerequisites

- `uv` installed ([Introduction to uv](https://github.com/astral-sh/uv)).
- `ffmpeg` installed (required for audio extraction).
- macOS (Apple Silicon recommended for `mlx-whisper`).

### Running the Server

You can run the server directly using `uv`:

```bash
# Run from source
uv run server.py
```

### Client Usage

To use this server with an MCP client (like Claude Desktop or another MCP-compliant tool), you can configure it as follows:

```json
{
  "mcpServers": {
    "subtitle": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/gaoyifan/mcp-subtitle",
        "mcp-subtitle"
      ]
    }
  }
}
```

### MCP JSON Config

If your client reads an `mcp.json` file, you can drop in the same config:

```json
{
  "mcpServers": {
    "subtitle": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/gaoyifan/mcp-subtitle",
        "mcp-subtitle"
      ]
    }
  }
}
```

Or run it ephemerally:

```bash
uvx --from git+https://github.com/gaoyifan/mcp-subtitle mcp-subtitle
```

### SSE Mode

To run the server in SSE mode (e.g. for inspection or remote access):

```bash
uv run server.py --sse --port 8000
```

### Remote SSE MCP Config

To connect to a remote SSE server, point your client at its URL:

```json
{
  "mcpServers": {
    "subtitle-remote": {
      "transport": "sse",
      "url": "https://your-host.example.com:8000/sse"
    }
  }
}
```


### Development & Testing

```bash
# Run tests
uv run pytest
```
