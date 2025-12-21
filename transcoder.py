#!/usr/bin/env python3
"""
HLS Live Transcoder Engine

Transcodes video segments on-demand using FFmpeg with adaptive quality.

Design: Single-threaded transcoding with multi-ahead prefetch.
- One FFmpeg process gets 100% CPU for fastest segment time
- After serving segment N, prefetch N+1..N+4 in background
- No parallel transcoding (would split CPU and increase latency)

Adapted from SegmentPlayer's transcoder for standalone Stremio addon use.
"""
from __future__ import annotations

import os
import subprocess
import hashlib
import threading
import time
import json
import re
from typing import Optional

# Configuration
MEDIA_DIR = os.environ.get('MEDIA_DIR', '/data/media')
CACHE_DIR = os.environ.get('CACHE_DIR', '/data/cache')
SEGMENT_DURATION = int(os.environ.get('SEGMENT_DURATION', '4'))
PREFETCH_SEGMENTS = int(os.environ.get('PREFETCH_SEGMENTS', '4'))

os.makedirs(CACHE_DIR, exist_ok=True)

# Resolution presets: (width, height, crf)
RESOLUTIONS = {
    'original': (None, None, 23),
    '1080p': (1920, 1080, 23),
    '720p': (1280, 720, 24),
    '480p': (854, 480, 25),
    '360p': (640, 360, 26),
}

# x264 presets ordered from fastest to slowest
X264_PRESETS = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']

# Subtitle codecs that cannot be converted to WebVTT (image-based)
UNSUPPORTED_SUBTITLE_CODECS = {'hdmv_pgs_subtitle', 'dvd_subtitle', 'dvb_subtitle', 'xsub'}


class AdaptiveQuality:
    """
    Coordinated adaptive quality control using both preset and CRF.

    Strategy - PRIORITIZE LOW CRF (quality over compression efficiency):
    - CRF directly affects output quality (lower = better visual quality)
    - Preset affects encoding speed (faster = less CPU, slightly worse compression)
    - For streaming, visual quality matters more than file size
    - So: keep CRF low, use faster presets when needed

    Target: 60-80% of segment duration (transcode ratio)
    - Below 60%: We have headroom, can increase quality
    - 60-80%: Sweet spot, maintain current settings
    - Above 80%: Too slow, decrease quality (preset first!)
    - Above 100%: EMERGENCY, drop to fastest settings

    Coordination rules:
    - DECREASE quality: Drop preset first (faster encoding), CRF only as last resort
    - INCREASE quality: Decrease CRF first (better quality), then preset (capped at 'medium')
    - Max preset is 'medium' - slower presets waste CPU for minimal streaming benefit
    """

    MAX_PRESET_INDEX = 5  # 'medium' - slower presets not worth it for streaming

    def __init__(self, initial_preset: str = 'fast'):
        self._lock = threading.Lock()

        # Preset state (capped at medium)
        initial_index = X264_PRESETS.index(initial_preset) if initial_preset in X264_PRESETS else 4
        self._preset_index = min(initial_index, self.MAX_PRESET_INDEX)

        # CRF state
        self._crf_offset = 0  # 0 to 7

        # Shared state
        self._recent_ratios: list[float] = []
        self._window_size = 5
        self._consecutive_good_signals = 0
        self._last_change_time = 0.0

        # Thresholds
        self._target_min = 60.0
        self._target_max = 80.0

        # Stats
        self._preset_ups = 0
        self._preset_downs = 0
        self._crf_ups = 0
        self._crf_downs = 0

    @property
    def preset(self) -> str:
        with self._lock:
            return X264_PRESETS[self._preset_index]

    def get_crf(self, base_crf: int) -> int:
        with self._lock:
            return min(base_crf + self._crf_offset, 30)

    def record_transcode(self, elapsed: float) -> dict | None:
        """Record transcode time and adjust quality settings."""
        last_ratio = (elapsed / SEGMENT_DURATION) * 100

        with self._lock:
            self._recent_ratios.append(last_ratio)
            if len(self._recent_ratios) > self._window_size:
                self._recent_ratios.pop(0)

            avg_ratio = sum(self._recent_ratios) / len(self._recent_ratios)
            current_time = time.time()

            old_preset = X264_PRESETS[self._preset_index]
            old_crf = self._crf_offset

            # EMERGENCY: ratio > 100%
            if last_ratio > 100:
                if self._preset_index > 0 or self._crf_offset < 7:
                    self._preset_index = 0  # ultrafast
                    self._crf_offset = 7
                    self._consecutive_good_signals = 0
                    self._last_change_time = current_time
                    self._recent_ratios.clear()
                    self._preset_downs += 1
                    self._crf_ups += 1
                    print(f"[AdaptiveQuality] EMERGENCY {last_ratio:.1f}% → ultrafast, CRF +7")
                    return {'preset': 'ultrafast', 'crf_offset': 7, 'emergency': True}

            # DECREASE QUALITY: last ratio > 80%
            if last_ratio > self._target_max:
                self._consecutive_good_signals = 0

                if self._preset_index > 0:
                    self._preset_index -= 1
                    self._preset_downs += 1
                    self._last_change_time = current_time
                    new_preset = X264_PRESETS[self._preset_index]
                    print(f"[AdaptiveQuality] Last {last_ratio:.1f}% > {self._target_max}% → {old_preset} → {new_preset}")
                    return {'preset': new_preset}
                elif self._crf_offset < 7:
                    self._crf_offset = min(self._crf_offset + 2, 7)
                    self._crf_ups += 1
                    self._last_change_time = current_time
                    print(f"[AdaptiveQuality] Last {last_ratio:.1f}% > {self._target_max}% (ultrafast) → CRF +{old_crf} → +{self._crf_offset}")
                    return {'crf_offset': self._crf_offset}

            # INCREASE QUALITY: avg ratio < 60%
            elif avg_ratio < self._target_min:
                self._consecutive_good_signals += 1

                if self._consecutive_good_signals >= 3 and current_time - self._last_change_time >= 5.0:
                    if self._crf_offset > 0:
                        self._crf_offset -= 1
                        self._crf_downs += 1
                        self._consecutive_good_signals = 0
                        self._last_change_time = current_time
                        print(f"[AdaptiveQuality] Avg {avg_ratio:.1f}% < {self._target_min}% → CRF +{old_crf} → +{self._crf_offset}")
                        return {'crf_offset': self._crf_offset}
                    elif self._preset_index < self.MAX_PRESET_INDEX:
                        self._preset_index += 1
                        self._preset_ups += 1
                        self._consecutive_good_signals = 0
                        self._last_change_time = current_time
                        new_preset = X264_PRESETS[self._preset_index]
                        print(f"[AdaptiveQuality] Avg {avg_ratio:.1f}% < {self._target_min}% → {old_preset} → {new_preset}")
                        return {'preset': new_preset}
            else:
                self._consecutive_good_signals = 0

            return None

    def get_stats(self) -> dict:
        with self._lock:
            avg_ratio = sum(self._recent_ratios) / len(self._recent_ratios) if self._recent_ratios else 0
            last_ratio = self._recent_ratios[-1] if self._recent_ratios else 0
            return {
                'preset': X264_PRESETS[self._preset_index],
                'crf_offset': self._crf_offset,
                'target_range': f"{self._target_min:.0f}-{self._target_max:.0f}%",
                'avg_ratio': round(avg_ratio, 1),
                'last_ratio': round(last_ratio, 1),
                'preset_adjustments': {'up': self._preset_ups, 'down': self._preset_downs},
                'crf_adjustments': {'up': self._crf_ups, 'down': self._crf_downs},
            }


class SegmentManager:
    """
    Manages segment transcoding with single-threaded execution.

    Design rationale:
    - Single FFmpeg gets 100% CPU → fastest per-segment time
    - Multi-ahead prefetch uses idle time while user watches current segment
    - Multiple requests for same segment wait on shared Event
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._in_progress: dict[str, threading.Event] = {}
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_queue: list[tuple[str, callable]] = []

        # Metrics
        self.total_segments = 0
        self.cache_hits = 0
        self.total_transcode_time = 0.0
        self.last_segment_time = 0.0
        self.min_segment_time = float('inf')
        self.max_segment_time = 0.0
        self._transcode_times: list[float] = []

        # Current file codec info
        self.current_video_codec: str | None = None
        self.current_audio_codec: str | None = None
        self.current_file: str | None = None

    def get_segment(self, key: str, transcode_fn) -> str | None:
        """Get a segment, transcoding if necessary."""
        with self._lock:
            self.total_segments += 1

            if key in self._in_progress:
                event = self._in_progress[key]
                wait_for_other = True
            else:
                event = threading.Event()
                self._in_progress[key] = event
                wait_for_other = False

        if wait_for_other:
            completed = event.wait(timeout=120)
            if completed:
                return transcode_fn()
            else:
                return None

        try:
            result = transcode_fn()
            return result
        finally:
            with self._lock:
                self._in_progress.pop(key, None)
            event.set()

    def schedule_prefetch(self, key: str, transcode_fn):
        """Schedule prefetch for a segment."""
        with self._lock:
            if key in self._in_progress:
                return
            if any(k == key for k, _ in self._prefetch_queue):
                return

            self._prefetch_queue.append((key, transcode_fn))

            if not self._prefetch_thread or not self._prefetch_thread.is_alive():
                self._prefetch_thread = threading.Thread(
                    target=self._process_prefetch_queue,
                    daemon=True
                )
                self._prefetch_thread.start()

    def _process_prefetch_queue(self):
        """Process prefetch queue sequentially."""
        while True:
            with self._lock:
                if not self._prefetch_queue:
                    return

                key, transcode_fn = self._prefetch_queue.pop(0)
                if key in self._in_progress:
                    continue

                event = threading.Event()
                self._in_progress[key] = event

            try:
                transcode_fn()
            finally:
                with self._lock:
                    self._in_progress.pop(key, None)
                event.set()

    def is_in_progress(self, key: str) -> bool:
        with self._lock:
            return key in self._in_progress

    def record_cache_hit(self):
        with self._lock:
            self.total_segments += 1
            self.cache_hits += 1

    def record_transcode_time(self, elapsed: float):
        with self._lock:
            self.total_transcode_time += elapsed
            self.last_segment_time = elapsed
            if elapsed < self.min_segment_time:
                self.min_segment_time = elapsed
            if elapsed > self.max_segment_time:
                self.max_segment_time = elapsed
            self._transcode_times.append(elapsed)
            if len(self._transcode_times) > 100:
                self._transcode_times.pop(0)

    def set_codec_info(self, video_codec: str | None, audio_codec: str | None, filepath: str | None = None):
        with self._lock:
            self.current_video_codec = video_codec
            self.current_audio_codec = audio_codec
            if filepath:
                self.current_file = filepath

    def get_metrics(self) -> dict:
        with self._lock:
            avg_time = sum(self._transcode_times) / len(self._transcode_times) if self._transcode_times else 0
            last_ratio = (self.last_segment_time / SEGMENT_DURATION * 100) if self.last_segment_time > 0 else 0
            avg_ratio = (avg_time / SEGMENT_DURATION * 100) if avg_time > 0 else 0
            return {
                'total_segments': self.total_segments,
                'cache_hits': self.cache_hits,
                'cache_hit_rate': round((self.cache_hits / self.total_segments * 100), 1) if self.total_segments > 0 else 0,
                'prefetch_queue_size': len(self._prefetch_queue),
                'total_transcode_time': round(self.total_transcode_time, 2),
                'last_segment_time': round(self.last_segment_time, 2),
                'avg_segment_time': round(avg_time, 2),
                'segment_duration': SEGMENT_DURATION,
                'transcode_ratio_last': round(last_ratio, 1),
                'transcode_ratio_avg': round(avg_ratio, 1),
                'video_codec': self.current_video_codec,
                'audio_codec': self.current_audio_codec,
                'current_file': self.current_file,
            }

    def reset_metrics(self):
        with self._lock:
            self.total_segments = 0
            self.cache_hits = 0
            self.total_transcode_time = 0.0
            self.last_segment_time = 0.0
            self.min_segment_time = float('inf')
            self.max_segment_time = 0.0
            self._transcode_times.clear()


class SubtitleManager:
    """Manages subtitle extraction with background processing."""

    def __init__(self):
        self._lock = threading.Lock()
        self._extractions: dict[str, tuple[threading.Event, list]] = {}

    def get_subtitle(self, key: str, filepath: str, file_hash: str, sub_index: int,
                     info: dict = None, timeout: float = 300) -> tuple[str | None, str | None]:
        """Get subtitle content, extracting if necessary."""
        cache_dir = os.path.join(CACHE_DIR, file_hash)
        vtt_file = os.path.join(cache_dir, f"subtitle_{sub_index}.vtt")
        error_file = os.path.join(cache_dir, f"subtitle_{sub_index}.error")

        # Fast path: already cached
        if os.path.exists(vtt_file):
            with open(vtt_file, 'r', encoding='utf-8') as f:
                return f.read(), None
        if os.path.exists(error_file):
            with open(error_file, 'r', encoding='utf-8') as f:
                return None, f.read()

        with self._lock:
            if os.path.exists(vtt_file):
                with open(vtt_file, 'r', encoding='utf-8') as f:
                    return f.read(), None
            if os.path.exists(error_file):
                with open(error_file, 'r', encoding='utf-8') as f:
                    return None, f.read()

            if key in self._extractions:
                event, result_holder = self._extractions[key]
            else:
                event = threading.Event()
                result_holder = [None]
                self._extractions[key] = (event, result_holder)

                thread = threading.Thread(
                    target=self._extract_background,
                    args=(key, filepath, file_hash, sub_index, info, event, result_holder),
                    daemon=True
                )
                thread.start()

        completed = event.wait(timeout=timeout)

        if completed:
            if result_holder[0] is not None:
                return result_holder[0]
            if os.path.exists(vtt_file):
                with open(vtt_file, 'r', encoding='utf-8') as f:
                    return f.read(), None
            if os.path.exists(error_file):
                with open(error_file, 'r', encoding='utf-8') as f:
                    return None, f.read()
            return None, "Extraction completed but result not found"

        return None, f"Extraction timed out after {timeout}s"

    def _extract_background(self, key: str, filepath: str, file_hash: str, sub_index: int,
                           info: dict, event: threading.Event, result_holder: list):
        try:
            result = self._do_extract(filepath, file_hash, sub_index, info)
            result_holder[0] = result
        except Exception as e:
            result_holder[0] = (None, f"Extraction thread error: {e}")
        finally:
            event.set()
            def cleanup():
                time.sleep(30)
                with self._lock:
                    self._extractions.pop(key, None)
            threading.Thread(target=cleanup, daemon=True).start()

    def _do_extract(self, filepath: str, file_hash: str, sub_index: int,
                    info: dict = None) -> tuple[str | None, str | None]:
        cache_dir = os.path.join(CACHE_DIR, file_hash)
        os.makedirs(cache_dir, exist_ok=True)

        vtt_file = os.path.join(cache_dir, f"subtitle_{sub_index}.vtt")
        temp_file = vtt_file + '.tmp'
        error_file = os.path.join(cache_dir, f"subtitle_{sub_index}.error")

        if os.path.exists(vtt_file):
            with open(vtt_file, 'r', encoding='utf-8') as f:
                return f.read(), None
        if os.path.exists(error_file):
            with open(error_file, 'r', encoding='utf-8') as f:
                return None, f.read()

        # Check subtitle codec
        if info:
            subtitle_streams = [s for s in info.get('streams', []) if s.get('codec_type') == 'subtitle']
            if sub_index < len(subtitle_streams):
                codec = subtitle_streams[sub_index].get('codec_name', '')
                if codec in UNSUPPORTED_SUBTITLE_CODECS:
                    error_msg = f"Subtitle format '{codec}' is image-based and cannot be converted to WebVTT"
                    with open(error_file, 'w', encoding='utf-8') as f:
                        f.write(error_msg)
                    return None, error_msg

        try:
            print(f"[Subtitle {sub_index}] Extracting from {os.path.basename(filepath)}...")

            result = subprocess.run(
                ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                 '-probesize', '5M', '-analyzeduration', '5M',
                 '-i', filepath,
                 '-map', f'0:s:{sub_index}',
                 '-vn', '-an',
                 '-c:s', 'webvtt', '-f', 'webvtt', temp_file],
                capture_output=True,
                timeout=600
            )

            if result.returncode == 0 and os.path.exists(temp_file):
                file_size = os.path.getsize(temp_file)
                if file_size > 10:
                    os.rename(temp_file, vtt_file)
                    print(f"[Subtitle {sub_index}] Extraction complete ({file_size} bytes)")
                    with open(vtt_file, 'r', encoding='utf-8') as f:
                        return f.read(), None
                else:
                    error_msg = "Extraction produced empty VTT file"
                    os.remove(temp_file)
            else:
                stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else 'Unknown error'
                error_msg = f"FFmpeg error: {stderr[:200]}"

            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(error_msg)
            return None, error_msg

        except subprocess.TimeoutExpired:
            error_msg = "Extraction timed out"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(error_msg)
            return None, error_msg


# Global managers
adaptive_quality = AdaptiveQuality(initial_preset='fast')
segment_manager = SegmentManager()
subtitle_manager = SubtitleManager()


def get_file_hash(filepath: str) -> str:
    """Generate cache key from filepath."""
    return hashlib.md5(filepath.encode()).hexdigest()[:16]


def get_video_info(filepath: str) -> dict | None:
    """Get video metadata using ffprobe."""
    full_path = os.path.join(MEDIA_DIR, filepath) if not filepath.startswith('/') else filepath
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', full_path],
            capture_output=True, text=True, timeout=30
        )
        return json.loads(result.stdout) if result.returncode == 0 else None
    except Exception:
        return None


def extract_codecs(info: dict) -> tuple[str | None, str | None]:
    """Extract video and audio codec names from ffprobe info."""
    streams = info.get('streams', [])
    video_codec = None
    audio_codec = None

    for stream in streams:
        codec_type = stream.get('codec_type')
        if codec_type == 'video' and video_codec is None:
            video_codec = stream.get('codec_name')
        elif codec_type == 'audio' and audio_codec is None:
            audio_codec = stream.get('codec_name')

    return video_codec, audio_codec


def get_segment_path(file_hash: str, audio: int, resolution: str, segment: int) -> str:
    """Get cache path for a muxed video+audio segment."""
    return os.path.join(CACHE_DIR, file_hash, f"seg_a{audio}_{resolution}_{segment:05d}.ts")


def transcode_segment(filepath: str, file_hash: str, audio: int, resolution: str, segment: int) -> str | None:
    """
    Transcode a single segment using FFmpeg.

    Optimized for speed with adaptive quality:
    - adaptive preset: auto-adjusts based on transcode ratio (target 60-80%)
    - threads 0: use all CPU cores
    - tune zerolatency: reduce encoding latency
    - Muxed video+audio: single segment contains both streams
    """
    cache_dir = os.path.join(CACHE_DIR, file_hash)
    os.makedirs(cache_dir, exist_ok=True)

    output = get_segment_path(file_hash, audio, resolution, segment)

    # Already cached?
    if os.path.exists(output):
        return output

    full_path = os.path.join(MEDIA_DIR, filepath) if not filepath.startswith('/') else filepath
    start_offset = segment * SEGMENT_DURATION
    res_preset = RESOLUTIONS.get(resolution, RESOLUTIONS['original'])
    width, height, base_crf = res_preset

    # Get current adaptive preset and CRF
    current_preset = adaptive_quality.preset
    current_crf = adaptive_quality.get_crf(base_crf)

    cpu_count = os.cpu_count() or 8

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-threads', str(cpu_count),
        '-ss', str(start_offset),
        '-i', full_path,
        '-t', str(SEGMENT_DURATION),
        '-map', '0:v:0',
        '-map', f'0:a:{audio}?',  # ? = optional, don't fail if audio track doesn't exist
        # Video: H.264 with adaptive quality
        '-c:v', 'libx264',
        '-preset', current_preset,
        '-tune', 'zerolatency',
        '-crf', str(current_crf),
        '-pix_fmt', 'yuv420p',
        '-x264-params', f'threads={cpu_count}:lookahead_threads={min(cpu_count, 8)}',
        '-force_key_frames', 'expr:gte(t,0)',
        # Audio: AAC (muxed with video)
        '-c:a', 'aac', '-b:a', '128k', '-ac', '2',
        # Output format
        '-f', 'mpegts',
        '-mpegts_copyts', '1',
        '-output_ts_offset', str(start_offset),
    ]

    if width and height:
        cmd.extend(['-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'])

    cmd.append(output)

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        elapsed = time.time() - start_time
        if result.returncode == 0 and os.path.exists(output):
            segment_manager.record_transcode_time(elapsed)
            adaptive_quality.record_transcode(elapsed)
            return output
        print(f"FFmpeg error: {result.stderr.decode()[-500:]}")
    except Exception as e:
        print(f"Transcode error: {e}")

    return None


def get_or_transcode_segment(filepath: str, file_hash: str, audio: int, resolution: str,
                              segment: int, info: dict) -> bytes | None:
    """Get segment data, transcoding if necessary."""
    output = get_segment_path(file_hash, audio, resolution, segment)
    key = f"{file_hash}:{audio}:{resolution}:{segment}"

    # Check if transcode in progress
    if segment_manager.is_in_progress(key):
        result = segment_manager.get_segment(
            key,
            lambda: transcode_segment(filepath, file_hash, audio, resolution, segment)
        )
        if result and os.path.exists(result):
            trigger_prefetch(filepath, file_hash, audio, resolution, segment, info)
            with open(result, 'rb') as f:
                return f.read()
        return None

    # Fast path: cached
    if os.path.exists(output):
        segment_manager.record_cache_hit()
        trigger_prefetch(filepath, file_hash, audio, resolution, segment, info)
        with open(output, 'rb') as f:
            return f.read()

    # Need to transcode
    result = segment_manager.get_segment(
        key,
        lambda: transcode_segment(filepath, file_hash, audio, resolution, segment)
    )

    if result and os.path.exists(result):
        trigger_prefetch(filepath, file_hash, audio, resolution, segment, info)
        with open(result, 'rb') as f:
            return f.read()

    return None


def trigger_prefetch(filepath: str, file_hash: str, audio: int, resolution: str,
                     current: int, info: dict):
    """Prefetch upcoming segments."""
    duration = float(info.get('format', {}).get('duration', 0))
    total = int(duration / SEGMENT_DURATION) + 1

    for offset in range(1, PREFETCH_SEGMENTS + 1):
        next_seg = current + offset
        if next_seg >= total:
            break

        output = get_segment_path(file_hash, audio, resolution, next_seg)
        if os.path.exists(output):
            continue

        key = f"{file_hash}:{audio}:{resolution}:{next_seg}"
        seg = next_seg
        segment_manager.schedule_prefetch(
            key,
            lambda s=seg: transcode_segment(filepath, file_hash, audio, resolution, s)
        )


def generate_master_playlist(info: dict, resolution_filter: str = None) -> str:
    """
    Generate HLS master playlist with multi-audio support.

    Each audio track gets its own stream variant with that audio muxed into the video.
    """
    streams = info.get('streams', [])
    video = next((s for s in streams if s.get('codec_type') == 'video'), None)
    audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
    subtitle_streams = [s for s in streams if s.get('codec_type') == 'subtitle']

    lines = ["#EXTM3U", "#EXT-X-VERSION:4", ""]

    # Build audio track metadata
    audio_tracks = []
    for i, a in enumerate(audio_streams):
        lang = a.get('tags', {}).get('language', 'und')
        title = a.get('tags', {}).get('title', '')
        channels = a.get('channels', 2)

        codec = a.get('codec_name', 'AAC').upper()
        ch_str = f"{channels}.0" if channels else "2.0"
        if title:
            name = title
        else:
            lang_name = lang.upper() if lang != 'und' else f"Audio {i+1}"
            name = f"{lang_name} - {codec} {ch_str}"

        audio_tracks.append({
            'index': i,
            'name': name,
            'lang': lang,
            'channels': channels,
        })

    # Subtitle tracks
    sub_counter = 0
    for i, s in enumerate(subtitle_streams):
        codec = s.get('codec_name', '')
        if codec in UNSUPPORTED_SUBTITLE_CODECS:
            continue
        lang = s.get('tags', {}).get('language', 'und')
        title = s.get('tags', {}).get('title', '')
        name = f"{title}" if title else (lang.upper() if lang != 'und' else f"Subtitle {sub_counter+1}")
        lines.append(f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="{name}",LANGUAGE="{lang}",DEFAULT=NO,AUTOSELECT=YES,URI="subtitle_{i}.m3u8"')
        sub_counter += 1

    lines.append("")

    # Video variants
    if video:
        src_w = video.get('width', 1920)
        src_h = video.get('height', 1080)
        subs_ref = ',SUBTITLES="subs"' if sub_counter > 0 else ''
        audio_ref = ',AUDIO="audio"' if audio_tracks else ''

        # Build quality ladder
        if resolution_filter:
            res_info = RESOLUTIONS.get(resolution_filter, RESOLUTIONS['original'])
            bw_map = {None: 5000000, 1080: 4000000, 720: 2500000, 480: 1200000, 360: 800000}
            quality_order = [(resolution_filter, res_info[1], bw_map.get(res_info[1], 5000000))]
        else:
            quality_order = [
                ('original', None, 5000000),
                ('1080p', 1080, 4000000),
                ('720p', 720, 2500000),
                ('480p', 480, 1200000),
                ('360p', 360, 800000),
            ]

        # Audio track declarations - all in same group "audio" with URIs to muxed streams
        # Each audio track points to a stream playlist that has that audio muxed in
        # Player switches to different stream when audio is changed
        for i, track in enumerate(audio_tracks):
            is_default = 'YES' if i == 0 else 'NO'
            # Only default track should autoselect - prevents players from prefetching all audio streams
            autoselect = 'YES' if i == 0 else 'NO'
            # Use first quality in ladder for audio URI (typically 'original' or the filtered resolution)
            first_res = quality_order[0][0]
            lines.append(
                f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="{track["name"]}",'
                f'LANGUAGE="{track["lang"]}",CHANNELS="{track["channels"]}",DEFAULT={is_default},AUTOSELECT={autoselect},'
                f'URI="stream_a{track["index"]}_{first_res}.m3u8"'
            )

        if audio_tracks:
            lines.append("")

        # Stream variants - one per resolution, referencing single audio group
        # Only generate variants for the default (first) audio track
        # Audio switching is handled by EXT-X-MEDIA URIs above
        for res_key, target_h, bw in quality_order:
            if target_h and target_h > src_h:
                continue
            if target_h and target_h == src_h:
                continue
            w, h, _ = RESOLUTIONS.get(res_key, RESOLUTIONS['original'])
            width = w or src_w
            height = h or src_h
            label = f"{height}p (Original)" if res_key == 'original' else f"{height}p"
            lines.append(f'#EXT-X-STREAM-INF:BANDWIDTH={bw},RESOLUTION={width}x{height}{audio_ref}{subs_ref},NAME="{label}"')
            lines.append(f"stream_a0_{res_key}.m3u8")

    return "\n".join(lines) + "\n"


def generate_stream_playlist(info: dict, audio: int, resolution: str) -> str:
    """Generate HLS stream playlist with muxed video+audio segments."""
    duration = float(info.get('format', {}).get('duration', 0))
    num_segments = int(duration / SEGMENT_DURATION) + 1

    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{SEGMENT_DURATION}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        ""
    ]

    for i in range(num_segments):
        seg_dur = min(SEGMENT_DURATION, duration - (i * SEGMENT_DURATION))
        if seg_dur > 0.1:
            lines.append(f"#EXTINF:{seg_dur:.3f},")
            lines.append(f"seg_a{audio}_{resolution}_{i:05d}.ts")

    lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines)


def generate_subtitle_playlist(info: dict, sub_index: int) -> str:
    """Generate subtitle playlist."""
    duration = float(info.get('format', {}).get('duration', 0))
    return "\n".join([
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{int(duration) + 1}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        "",
        f"#EXTINF:{duration:.3f},",
        f"subtitle_{sub_index}.vtt",
        "#EXT-X-ENDLIST"
    ])


def get_metrics() -> dict:
    """Get transcoding metrics."""
    return {
        **segment_manager.get_metrics(),
        'adaptive_quality': adaptive_quality.get_stats(),
    }


def reset_metrics():
    """Reset transcoding metrics."""
    segment_manager.reset_metrics()
