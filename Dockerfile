# autoedit backend — python + ffmpeg, dependency-free core.
FROM python:3.11-slim

# ffmpeg is the only system dependency the pipeline needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# The core pipeline + HTTP backend use only the Python stdlib + ffmpeg, so no
# pip install is required. To enable real speech-to-text, add faster-whisper:
#   RUN pip install --no-cache-dir faster-whisper

ENV PORT=8000
EXPOSE 8000

# server.py binds 0.0.0.0:$PORT (Render/Fly inject PORT).
CMD ["python", "server.py"]
