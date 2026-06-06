"""Real speech-to-text adapter (faster-whisper).

Drop-in for the pipeline's `asr` callable: returns the same dict schema as the
fixture transcript, so nothing downstream changes. faster-whisper is imported
lazily so the core MVP runs even when it isn't installed.

    pip install faster-whisper        # needs ffmpeg; GPU optional

Env knobs: WHISPER_MODEL (default large-v3), WHISPER_DEVICE (cuda|cpu),
WHISPER_COMPUTE (float16|int8).
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel  # lazy

    return WhisperModel(
        os.environ.get("WHISPER_MODEL", "large-v3"),
        device=os.environ.get("WHISPER_DEVICE", "cpu"),
        compute_type=os.environ.get("WHISPER_COMPUTE", "int8"),
    )


def transcribe(video_path: str) -> dict:
    segments, info = _model().transcribe(video_path, word_timestamps=True)
    out = []
    for s in segments:
        out.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text,
            "words": [
                {"text": w.word, "start": float(w.start), "end": float(w.end)}
                for w in (s.words or [])
            ],
        })
    return {"language": getattr(info, "language", "en"), "segments": out}
