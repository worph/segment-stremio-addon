#!/usr/bin/env python3
"""
Stremio Addon Server for SegmentPlayer

A standalone Stremio addon that catalogs media files and generates stream URLs
pointing to a SegmentPlayer instance for transcoding.

Environment Variables:
- MEDIA_DIR: Directory containing video files (default: /data/media)
- PORT: HTTP server port (default: 7000)
"""
from __future__ import annotations

import os
import re
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote
import threading

from stremio import StremioHandler

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
        """Handle HEAD requests for health checks."""
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
        data, content_type = stremio_handler.handle_manifest()
        self.send_data(data, content_type)

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

    def get_base_url(self) -> str:
        """
        Extract base URL from request headers with intelligent protocol detection.
        Handles reverse proxy headers (X-Forwarded-Proto, Forwarded, Referer).
        """
        host = self.headers.get('X-Forwarded-Host') or self.headers.get('Host', 'localhost:7000')

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
    media_dir = os.environ.get('MEDIA_DIR', '/data/media')
    print(f"Stremio Addon Server starting on port {PORT}")
    print(f"Media directory: {media_dir}")
    print(f"Manifest URL: http://localhost:{PORT}/manifest.json")
    print("Stream URLs auto-detected from request headers")
    ThreadedServer(('0.0.0.0', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
