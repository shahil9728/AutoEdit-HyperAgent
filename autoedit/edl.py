"""The Edit Decision List (EDL) — the shared source of truth for the pipeline.

Every agent communicates by reading and mutating these dataclasses, never by
calling each other directly. That keeps the pipeline debuggable: dump the EDL
to JSON at any stage and you can see exactly what each agent decided.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Transcript (output of the Analysis Agent / ASR)
# --------------------------------------------------------------------------- #
@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Transcript:
    segments: List[Segment] = field(default_factory=list)
    language: str = "en"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transcript":
        segs: List[Segment] = []
        for s in data.get("segments", []):
            words: List[Word] = []
            for w in s.get("words", []):
                # Accept both our fixture schema ("text") and Whisper's ("word").
                wt = (w.get("text") if "text" in w else w.get("word", "")).strip()
                words.append(Word(wt, float(w["start"]), float(w["end"])))
            segs.append(
                Segment(float(s["start"]), float(s["end"]), str(s["text"]).strip(), words)
            )
        return cls(segs, data.get("language", "en"))

    @classmethod
    def load(cls, path: str) -> "Transcript":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# --------------------------------------------------------------------------- #
# EDL pieces
# --------------------------------------------------------------------------- #
@dataclass
class SourceMeta:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


@dataclass
class Clip:
    """One selected slice of the source, in source time."""
    id: str
    src_in: float
    src_out: float
    score: float = 0.0          # final blended score used for ranking
    visual_score: float = 0.0   # from the visual-analysis pass (0..1)
    speech_score: float = 0.0   # from transcript hook analysis (0..1)
    reason: str = ""
    focus_x: float = 0.5  # 0..1 horizontal crop centre for reframing
    speed: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, (self.src_out - self.src_in) / max(self.speed, 1e-6))


@dataclass
class CaptionWord:
    """A word remapped into the *target* timeline (after cutting)."""
    text: str
    start: float
    end: float


@dataclass
class EDL:
    source: SourceMeta
    target: Dict[str, Any]
    timeline: List[Clip] = field(default_factory=list)
    captions: List[CaptionWord] = field(default_factory=list)
    music: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None  # shot metrics + weights (transparency)

    def total_duration(self) -> float:
        return sum(c.duration for c in self.timeline)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": asdict(self.source),
            "target": self.target,
            "analysis": self.analysis,
            "timeline": [asdict(c) for c in self.timeline],
            "captions": [asdict(c) for c in self.captions],
            "music": self.music,
            "total_duration": round(self.total_duration(), 3),
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
