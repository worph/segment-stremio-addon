#!/usr/bin/env python3
"""
Stremio Addon Server with Integrated HLS Transcoder

A standalone Stremio addon that:
1. Catalogs media files and provides Stremio-compatible API
2. Transcodes video on-the-fly using FFmpeg with adaptive quality
3. Serves HLS streams with muxed video+audio segments
4. Supports direct file serving with range requests

Environment Variables:
- MEDIA_DIR: Directory containing video files (default: /data/media)
- CACHE_DIR: Directory for transcoded segments (default: /data/cache)
- PORT: HTTP server port (default: 7000)
- SEGMENT_DURATION: HLS segment length in seconds (default: 4)
- PREFETCH_SEGMENTS: How many segments to prefetch ahead (default: 4)
"""
from __future__ import annotations

import os
import re
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote
import threading

from stremio import StremioHandler, get_media_file_count
import transcoder

# Configuration
PORT = int(os.environ.get('PORT', '7000'))

# Global Stremio handler
stremio_handler = StremioHandler()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Log requests for debugging
        print(f"[{self.address_string()}] {fmt % args}")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def send_data(self, data: bytes, content_type: str):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', len(data))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Range')
        self.end_headers()

    def do_HEAD(self):
        """Handle HEAD requests for health checks and streaming."""
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path.startswith('/stremio/') or path == '/' or path.startswith('/manifest.json'):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            if path.endswith('.json'):
                self.send_header('Content-Type', 'application/json')
            else:
                self.send_header('Content-Type', 'text/html')
            self.end_headers()
        elif path.startswith('/transcode/') or path.startswith('/direct/'):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            if path.endswith('.m3u8'):
                self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
            elif path.endswith('.ts'):
                self.send_header('Content-Type', 'video/mp2t')
            elif path.endswith('.vtt'):
                self.send_header('Content-Type', 'text/vtt')
            else:
                self.send_header('Content-Type', 'application/octet-stream')
            self.end_headers()
        else:
            self.send_error(404)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # Root - serve setup page
        if path == '/' or path == '/index.html':
            return self.serve_setup_page()

        # Health check
        if path == '/health':
            return self.send_json({'status': 'ok'})

        # Stremio manifest at root level (for cleaner URLs)
        if path == '/manifest.json':
            return self.handle_stremio_manifest()

        # Stremio addon endpoints (also support /stremio/ prefix)
        if path == '/stremio/manifest.json':
            return self.handle_stremio_manifest()

        m = re.match(r'^(?:/stremio)?/catalog/(\w+)/([^/]+)(?:/([^.]+))?\.json$', path)
        if m:
            return self.handle_stremio_catalog(m.group(1), m.group(2), m.group(3))

        m = re.match(r'^(?:/stremio)?/meta/(\w+)/([^/]+)\.json$', path)
        if m:
            return self.handle_stremio_meta(m.group(1), m.group(2))

        m = re.match(r'^(?:/stremio)?/stream/(\w+)/([^/]+)\.json$', path)
        if m:
            return self.handle_stremio_stream(m.group(1), m.group(2))

        # === Transcoder endpoints ===

        # Metrics
        if path == '/transcode/metrics':
            metrics = transcoder.get_metrics()
            metrics['total_files'] = get_media_file_count()
            return self.send_json(metrics)

        if path == '/transcode/reset-metrics':
            transcoder.reset_metrics()
            return self.send_json({'status': 'ok'})

        # Direct file serving with range support
        m = re.match(r'^/direct/(.+)$', path)
        if m:
            return self.handle_direct_file(m.group(1))

        # Master playlist (all resolutions)
        m = re.match(r'^/transcode/(.+?)/master\.m3u8$', path)
        if m:
            return self.handle_master_playlist(m.group(1))

        # Quality-specific master playlist (e.g., master_720p.m3u8, master_original.m3u8)
        m = re.match(r'^/transcode/(.+?)/master_(\w+)\.m3u8$', path)
        if m:
            return self.handle_master_playlist(m.group(1), m.group(2))

        # Stream playlist (muxed video+audio)
        m = re.match(r'^/transcode/(.+?)/stream_a(\d+)_(\w+)\.m3u8$', path)
        if m:
            return self.handle_stream_playlist(m.group(1), int(m.group(2)), m.group(3))

        # Muxed video+audio segment
        m = re.match(r'^/transcode/(.+?)/seg_a(\d+)_(\w+)_(\d+)\.ts$', path)
        if m:
            return self.handle_segment(m.group(1), int(m.group(2)), m.group(3), int(m.group(4)))

        # Subtitle playlist
        m = re.match(r'^/transcode/(.+?)/subtitle_(\d+)\.m3u8$', path)
        if m:
            return self.handle_subtitle_playlist(m.group(1), int(m.group(2)))

        # Subtitle VTT
        m = re.match(r'^/transcode/(.+?)/subtitle_(\d+)\.vtt$', path)
        if m:
            return self.handle_subtitle_vtt(m.group(1), int(m.group(2)))

        self.send_error(404)

    def serve_setup_page(self):
        """Serve the Stremio setup page."""
        html_path = os.path.join(os.path.dirname(__file__), 'www', 'index.html')
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content.encode('utf-8')))
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        else:
            # Fallback minimal page
            content = f"""<!DOCTYPE html>
<html>
<head><title>SegmentPlayer Stremio Addon</title></head>
<body style="font-family: sans-serif; background: #1a1a2e; color: #fff; padding: 2rem;">
<h1>SegmentPlayer Stremio Addon</h1>
<p>Install URL: <code>{self.get_base_url()}/manifest.json</code></p>
<p><a href="stremio://{self.headers.get('Host', 'localhost')}/manifest.json" style="color: #4dabf7;">Install in Stremio</a></p>
</body>
</html>"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))

    def handle_stremio_manifest(self):
        host = self.get_host()
        data, content_type = stremio_handler.handle_manifest(host)
        self.send_data(data, content_type)

    def get_host(self) -> str:
        """Extract host from request headers (without protocol or port)."""
        host = self.headers.get('X-Forwarded-Host') or self.headers.get('Host', 'localhost')
        # Remove port if present
        if ':' in host and not host.startswith('['):  # Handle IPv6
            host = host.rsplit(':', 1)[0]
        return host

    def handle_stremio_catalog(self, catalog_type: str, catalog_id: str, extra: str = None):
        extra_dict = {}
        if extra:
            for param in extra.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    extra_dict[key] = unquote(value)

        data, content_type = stremio_handler.handle_catalog(catalog_type, catalog_id, extra_dict)
        self.send_data(data, content_type)

    def handle_stremio_meta(self, meta_type: str, meta_id: str):
        data, content_type = stremio_handler.handle_meta(meta_type, meta_id)
        if data:
            self.send_data(data, content_type)
        else:
            self.send_error(404, "Video not found")

    def handle_stremio_stream(self, stream_type: str, stream_id: str):
        # Auto-detect base URL from request headers
        base_url = self.get_base_url()

        data, content_type = stremio_handler.handle_stream(stream_type, stream_id, base_url)
        if data:
            self.send_data(data, content_type)
        else:
            self.send_error(404, "Video not found")

    # === Transcoder handlers ===

    def get_file_info(self, filepath: str):
        """Get file path and info, or send error."""
        full_path = os.path.join(transcoder.MEDIA_DIR, filepath)
        if not os.path.exists(full_path):
            self.send_error(404, f"File not found: {filepath}")
            return None, None, None

        info = transcoder.get_video_info(full_path)
        if not info:
            self.send_error(500, "Could not probe file")
            return None, None, None

        return full_path, transcoder.get_file_hash(filepath), info

    def handle_master_playlist(self, filepath: str, resolution: str = None):
        """Generate and serve master HLS playlist."""
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return

        playlist = transcoder.generate_master_playlist(info, resolution)
        self.send_data(playlist.encode(), 'application/vnd.apple.mpegurl')

    def handle_stream_playlist(self, filepath: str, audio: int, resolution: str):
        """Generate and serve stream playlist for specific audio track and resolution."""
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return

        playlist = transcoder.generate_stream_playlist(info, audio, resolution)
        self.send_data(playlist.encode(), 'application/vnd.apple.mpegurl')

    def handle_segment(self, filepath: str, audio: int, resolution: str, segment: int):
        """Transcode and serve a muxed video+audio segment."""
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return

        # Update codec info and current file in metrics
        video_codec, audio_codec = transcoder.extract_codecs(info)
        transcoder.segment_manager.set_codec_info(video_codec, audio_codec, filepath)

        data = transcoder.get_or_transcode_segment(filepath, file_hash, audio, resolution, segment, info)
        if data:
            self.send_data(data, 'video/mp2t')
        else:
            self.send_error(500, "Transcode failed")

    def handle_subtitle_playlist(self, filepath: str, sub_index: int):
        """Generate and serve subtitle playlist."""
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return

        playlist = transcoder.generate_subtitle_playlist(info, sub_index)
        self.send_data(playlist.encode(), 'application/vnd.apple.mpegurl')

    def handle_subtitle_vtt(self, filepath: str, sub_index: int):
        """Extract and serve subtitle as WebVTT."""
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return

        key = f"{file_hash}:sub:{sub_index}"
        content, error = transcoder.subtitle_manager.get_subtitle(key, full_path, file_hash, sub_index, info)

        if content:
            self.send_data(content.encode('utf-8'), 'text/vtt')
        else:
            # Return empty VTT with error as note so playback continues
            error_vtt = f"WEBVTT\n\nNOTE Subtitle extraction failed: {error or 'Unknown error'}\n"
            self.send_data(error_vtt.encode('utf-8'), 'text/vtt')

    def handle_direct_file(self, filepath: str):
        """Serve raw video file with range request support for seeking."""
        full_path = os.path.join(transcoder.MEDIA_DIR, filepath)
        if not os.path.exists(full_path):
            self.send_error(404, f"File not found: {filepath}")
            return

        # Determine content type
        ext = os.path.splitext(filepath)[1].lower()
        content_types = {
            '.mp4': 'video/mp4',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.m4v': 'video/x-m4v',
            '.ts': 'video/mp2t',
            '.m2ts': 'video/mp2t',
        }
        content_type = content_types.get(ext, 'application/octet-stream')

        file_size = os.path.getsize(full_path)
        range_header = self.headers.get('Range')

        if range_header:
            # Parse range request (e.g., "bytes=0-1023")
            range_match = re.match(r'bytes=(\d*)-(\d*)', range_header)
            if range_match:
                start = int(range_match.group(1)) if range_match.group(1) else 0
                end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(206)  # Partial Content
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', length)
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                with open(full_path, 'rb') as f:
                    f.seek(start)
                    remaining = length
                    chunk_size = 64 * 1024  # 64KB chunks
                    while remaining > 0:
                        chunk = f.read(min(chunk_size, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return

        # No range request - serve entire file
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', file_size)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        with open(full_path, 'rb') as f:
            chunk_size = 64 * 1024
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def get_base_url(self) -> str:
        """
        Extract base URL from request headers with intelligent protocol detection.
        Handles reverse proxy headers (X-Forwarded-Proto, Forwarded, Referer).
        """
        host = self.headers.get('X-Forwarded-Host') or self.headers.get('Host', 'localhost')

        # Protocol detection with multiple fallbacks
        proto = self.headers.get('X-Forwarded-Proto', '')

        # Check standard Forwarded header (RFC 7239)
        if not proto:
            forwarded = self.headers.get('Forwarded', '')
            if forwarded:
                match = re.search(r'proto=([^;,\s]+)', forwarded, re.IGNORECASE)
                if match:
                    proto = match.group(1).lower()

        # Try to infer from Referer header
        if not proto:
            referer = self.headers.get('Referer', '')
            if referer:
                match = re.match(r'^(https?)://', referer, re.IGNORECASE)
                if match:
                    proto = match.group(1).lower()

        # Final fallback: Assume HTTPS for non-localhost domains
        if not proto:
            is_localhost = 'localhost' in host or '127.0.0.1' in host or '::1' in host
            proto = 'http' if is_localhost else 'https'

        # Remove default ports
        clean_host = host.replace(':80', '').replace(':443', '')

        return f"{proto}://{clean_host}"


class ThreadedServer(HTTPServer):
    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address), daemon=True)
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def main():
    print(f"Stremio Addon with HLS Transcoder starting on port {PORT}")
    print(f"Media: {transcoder.MEDIA_DIR} | Cache: {transcoder.CACHE_DIR}")
    print(f"Segment: {transcoder.SEGMENT_DURATION}s | Prefetch: {transcoder.PREFETCH_SEGMENTS} segments")
    print(f"Manifest URL: http://localhost:{PORT}/manifest.json")
    print(f"Metrics URL: http://localhost:{PORT}/transcode/metrics")
    print("Adaptive quality: target 60-80% transcode ratio")
    ThreadedServer(('0.0.0.0', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
