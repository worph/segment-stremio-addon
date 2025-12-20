# SegmentPlayer Stremio Addon

A standalone Stremio addon that catalogs your local media library and streams via SegmentPlayer.

## Features

- Scans media directory for video files (mp4, mkv, mov, avi, webm, m4v, ts, m2ts)
- Provides Stremio-compatible catalog with search and genre filtering
- Multiple stream options per video:
  - Direct Play (original codec)
  - HLS Original (source resolution, transcoded)
  - HLS Auto (adaptive bitrate)
  - Open in SegmentPlayer web player
- Series detection (S##E## pattern)
- Subtitle extraction (WebVTT)
- Multi-audio track support

## Quick Start

### With Docker

```bash
docker run -d --name stremio-addon \
  -p 7000:7000 \
  -v /path/to/videos:/data/media:ro \
  ghcr.io/your-username/segment-stremio-addon
```

### Local Development

```bash
MEDIA_DIR="/path/to/videos" python3 server.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDIA_DIR` | `/data/media` | Directory containing video files |
| `PORT` | `7000` | HTTP server port |

Stream URLs are auto-detected from request headers (Host, X-Forwarded-Host, Referer).

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│ Stremio Client  │         │ Stremio Addon   │
│                 │ ◄────── │ (This service)  │
│ Browse/Search   │         │ - Catalog       │
│ Select video    │         │ - Metadata      │
└────────┬────────┘         │ - Stream URLs   │
         │                  └─────────────────┘
         │                          │
         │ Stream URL               │ Points to
         │                          ▼
         │                  ┌─────────────────┐
         └─────────────────►│ SegmentPlayer   │
                            │ - HLS Transcode │
                            │ - Direct serve  │
                            └─────────────────┘
```

The addon generates stream URLs that point to your SegmentPlayer instance for actual media streaming. Both services share the same media directory.

## Installing in Stremio

1. Open the addon page: `http://your-addon-host:7000/`
2. Copy the manifest URL or click "Install in Stremio"
3. The addon will appear in your Stremio catalog

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Setup page |
| `GET /manifest.json` | Addon manifest |
| `GET /catalog/:type/:id.json` | Video catalog |
| `GET /meta/:type/:id.json` | Video metadata |
| `GET /stream/:type/:id.json` | Stream URLs |
| `GET /health` | Health check |

## Docker Compose Example

```yaml
services:
  segment-player:
    image: ghcr.io/your-username/segment-player
    ports:
      - "8080:80"
    volumes:
      - /path/to/videos:/data/media:ro
      - segment-cache:/data/cache

  stremio-addon:
    image: ghcr.io/your-username/segment-stremio-addon
    ports:
      - "7000:7000"
    volumes:
      - /path/to/videos:/data/media:ro

volumes:
  segment-cache:
```

Note: When both services are behind a reverse proxy at the same domain, the addon automatically uses the correct URL for stream links.

## License

MIT
