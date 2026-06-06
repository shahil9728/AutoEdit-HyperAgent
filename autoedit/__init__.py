"""autoedit — a minimal autonomous video-editing pipeline (MVP).

A miniature of the six-agent "AI video production team":

    Analysis -> Selection -> Storytelling -> Editing -> Captions -> Platform export

All agents coordinate through a single shared artifact: the Edit Decision List
(EDL). Each agent reads the EDL (and the transcript), makes a decision, and
writes its decision back. The render layer is the only component that touches
ffmpeg; everything upstream is pure data.

The ML-heavy steps (speech-to-text, highlight ranking) are *pluggable*: pass an
`asr` callable to replace the transcript fixture with real Whisper output, and
an `llm` callable to replace the heuristic highlight picker with a real model.
"""

from .pipeline import run_pipeline  # noqa: F401

__all__ = ["run_pipeline"]
__version__ = "0.3.0"
