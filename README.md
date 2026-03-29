# Segment Stremio Addon

A standalone Stremio addon that catalogs your local media library and streams with on-the-fly HLS transcoding.

## Features

- Scans media directory for video files (mp4, mkv, mov, avi, webm, m4v, ts, m2ts)
- Stremio-compatible catalog with search
- Multiple stream options per video:
  - Direct Play (original codec pass-through with range request support)
  - HLS Original (transcoded at source resolution)
  - HLS Auto (adaptive bitrate, player selects quality)
  - Quality-specific streams (720p, 480p, 360p)
- Adaptive quality encoding that adjusts x264 preset/CRF based on CPU performance
- Series detection (S##E## pattern)
- Subtitle extraction (WebVTT)
- Multi-audio track support (separate stream per audio language)
- Optional API key protection via URL prefix
- Live transcoding metrics dashboard

## Quick Start

### With Docker

```bash
docker run -d --name stremio-addon \
  -p 7000:7000 \
  -v /path/to/videos:/data/media:ro \
  -v segment-cache:/data/cache \
  segment-stremio-addon
```

With API key protection:

```bash
docker run -d --name stremio-addon \
  -p 7000:7000 \
  -e HASH_API_SEED=your-secret-seed \
  -v /path/to/videos:/data/media:ro \
  -v segment-cache:/data/cache \
  segment-stremio-addon
```

### Local Development

```bash
# Requires ffmpeg installed
MEDIA_DIR="/path/to/videos" CACHE_DIR="/tmp/cache" python3 server.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDIA_DIR` | `/data/media` | Directory containing video files |
| `CACHE_DIR` | `/data/cache` | Directory for transcoded segment cache |
| `PORT` | `7000` | HTTP server port |
| `SCHEME` | (auto) | URL scheme (`http` or `https`). Auto-detects: localhost = http, otherwise https |
| `SEGMENT_DURATION` | `4` | HLS segment length in seconds |
| `PREFETCH_SEGMENTS` | `4` | Number of segments to prefetch ahead |
| `HASH_API_SEED` | (empty) | When set, enables API key protection. All API routes require a key prefix derived from this seed |

## API Key Protection

When `HASH_API_SEED` is set, a 16-character API key is derived via SHA-256 hash of the seed. All routes (except `/`, `/health`) require the prefix `/api/{key}/`:

```
# Without protection
http://host:7000/manifest.json

# With protection (HASH_API_SEED=mysecret)
http://host:7000/api/652c7dc687d98c98/manifest.json
```

The setup page at `/` remains accessible without the key and displays the full manifest URL. This makes reverse proxy configuration easy — match `/api/*` to bypass proxy-level auth since the URL key handles it.

## Installing in Stremio

You can install this addon from either the **local Stremio app** (Windows, macOS, Linux, Android, iOS) or from **Stremio Web** at https://web.strem.io/:

1. Open the addon setup page: `http://your-host:7000/`
2. Copy the manifest URL or click "Install in Stremio"
3. Alternatively, in Stremio go to **Settings** > **Addons** > **Install addon from URL** and paste the manifest URL
4. The addon will appear as "SSA" in your Stremio catalog

## Architecture

```
Stremio Client → This Addon (catalog/meta/stream + HLS transcoding)
```

Everything runs in a single service — no external dependencies. The addon scans the media directory, serves the Stremio API, and transcodes on-the-fly using FFmpeg.

### Transcoding Design

- Single FFmpeg process gets 100% CPU for fastest per-segment time
- After serving segment N, prefetch N+1 to N+4 in background
- Adaptive quality adjusts encoding preset and CRF based on transcode ratio (target 60-80%)
- Segments are cached to disk for instant replay on seek

## API Endpoints

All endpoints below are prefixed with `/api/{key}` when API key protection is enabled.

| Endpoint | Description |
|----------|-------------|
| `GET /` | Setup page (always accessible) |
| `GET /health` | Health check (always accessible) |
| `GET /manifest.json` | Addon manifest |
| `GET /catalog/:type/:id.json` | Video catalog |
| `GET /meta/:type/:id.json` | Video metadata |
| `GET /stream/:type/:id.json` | Stream URLs |
| `GET /transcode/:path/master.m3u8` | ABR master playlist |
| `GET /transcode/:path/master_:quality.m3u8` | Quality-specific master playlist |
| `GET /transcode/:path/stream_a:n_:quality.m3u8` | Stream playlist |
| `GET /transcode/:path/seg_a:n_:quality_:seg.ts` | Muxed video+audio segment |
| `GET /transcode/:path/subtitle_:n.m3u8` | Subtitle playlist |
| `GET /transcode/:path/subtitle_:n.vtt` | Extracted subtitle (WebVTT) |
| `GET /direct/:path` | Direct file serving with range support |
| `GET /transcode/metrics` | Transcoding metrics (JSON) |
| `GET /transcode/reset-metrics` | Reset metrics counters |

## Docker Compose Example

```yaml
services:
  stremio-addon:
    build: .
    ports:
      - "7000:7000"
    environment:
      - HASH_API_SEED=your-secret-seed  # optional
    volumes:
      - /path/to/videos:/data/media:ro
      - segment-cache:/data/cache

volumes:
  segment-cache:
```

## License

MIT
