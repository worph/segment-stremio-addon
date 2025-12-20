FROM python:3.12-slim

# Install ffprobe for metadata extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application files
COPY server.py stremio.py ./
COPY www/ ./www/

# Create media directory mount point
RUN mkdir -p /data/media

# Environment variables
ENV MEDIA_DIR=/data/media \
    PORT=7000 \
    PYTHONUNBUFFERED=1

EXPOSE 7000

CMD ["python3", "server.py"]
