FROM python:3.12-slim

# Install ffmpeg for transcoding and metadata extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application files
COPY server.py stremio.py transcoder.py ./
COPY www/ ./www/

# Create media and cache directory mount points
RUN mkdir -p /data/media /data/cache

# Environment variables
ENV MEDIA_DIR=/data/media \
    CACHE_DIR=/data/cache \
    SEGMENT_DURATION=4 \
    PREFETCH_SEGMENTS=4 \
    PORT=7000 \
    PYTHONUNBUFFERED=1

EXPOSE 7000

# Volume for transcoded segment cache (optional but recommended)
VOLUME /data/cache

CMD ["python3", "server.py"]
