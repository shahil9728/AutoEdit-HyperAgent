"""The decision-making agents.

These are deliberately small and *pure* (no ffmpeg here): they take data and
return data. Each maps to one of the six agents in the product vision:

    AnalysisAgent       -> understand the footage (probe + transcribe)
    SelectionAgent      -> rank & pick the best moments
    StoryAgent          -> order them into a coherent sequence
    CaptionAgent        -> remap words onto the cut timeline

EditingAgent and PlatformAgent live in render.py because they drive ffmpeg.

The two ML steps are pluggable:
  * `asr(video_path) -> dict`  replaces the transcript fixture with Whisper.
  * `llm(prompt) -> str`       replaces the heuristic ranker with a real model.
If neither is supplied, deterministic offline fallbacks run so the whole
pipeline still works end-to-end with zero credentials.
"""

from __future__ import annotations

import json
import subprocess
from typing import Callable, List, Optional

from .edl import CaptionWord, Clip, Segment, SourceMeta, Transcript


# --------------------------------------------------------------------------- #
# Analysis Agent
# --------------------------------------------------------------------------- #
def probe_source(path: str) -> SourceMeta:
    """Read real container metadata with ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", path,
    ]
    data = json.loads(subprocess.check_output(cmd).decode())
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    num, _, den = v.get("r_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or "1")
    duration = float(data["format"].get("duration") or v.get("duration") or 0.0)
    return SourceMeta(
        path=path,
        width=int(v["width"]),
        height=int(v["height"]),
        fps=round(fps, 3),
        duration=round(duration, 3),
        has_audio=a is not None,
    )


def transcribe(
    video_path: str,
    transcript_path: Optional[str] = None,
    asr: Optional[Callable[[str], dict]] = None,
) -> Transcript:
    """Produce a word-timed transcript.

    Priority: real ASR callable > fixture file > error.
    A production ASR adapter (faster-whisper) is sketched in README.
    """
    if asr is not None:
        return Transcript.from_dict(asr(video_path))
    if transcript_path:
        return Transcript.load(transcript_path)
    # Visual-only mode: no speech available. Selection falls back to pure visual
    # scoring (correct for silent / cinematic footage); captions are skipped.
    return Transcript(segments=[])


# --------------------------------------------------------------------------- #
# Selection Agent
# --------------------------------------------------------------------------- #
# A tiny engagement lexicon. The real system replaces this whole function with
# an LLM call that reasons about hooks, payoff, and emotional beats.
_HOOK_WORDS = {
    "you", "your", "secret", "never", "always", "best", "worst", "mistake",
    "free", "amazing", "stop", "why", "how", "most", "biggest", "tip", "trick",
    "truth", "nobody", "everyone", "easy", "fast", "money", "now", "need", "hook",
}


def _score_segment(seg: Segment, idx: int) -> tuple:
    tokens = [w.lower().strip(".,!?;:") for w in seg.text.split()]
    hits = sum(1 for w in tokens if w in _HOOK_WORDS)
    density = hits / max(len(tokens), 1)
    score = 0.55 * min(hits, 3) / 3 + 0.45 * density
    if idx == 0:
        score += 0.10  # a natural opener is a decent hook candidate
    if not (1.2 <= seg.duration <= 9.0):
        score -= 0.12  # avoid micro-fragments and rambles
    matched = [w for w in tokens if w in _HOOK_WORDS]
    reason = (
        f"{hits} hook word(s){' (' + ', '.join(matched) + ')' if matched else ''}, "
        f"{seg.duration:.1f}s"
    )
    return round(max(score, 0.0), 4), reason


def select_highlights(
    transcript: Transcript,
    target_duration: float,
    llm: Optional[Callable[[str], str]] = None,
) -> List[Clip]:
    """Rank candidate segments and greedily pick the best within a time budget.

    With `llm` supplied, the model is asked to return the indices to keep; the
    heuristic is the offline fallback.
    """
    segs = transcript.segments
    if not segs:
        return []

    if llm is not None:
        keep = _llm_pick(transcript, target_duration, llm)
    else:
        scored = sorted(
            range(len(segs)), key=lambda i: _score_segment(segs[i], i)[0], reverse=True
        )
        keep, used = [], 0.0
        for i in scored:
            if used >= target_duration:
                break
            keep.append(i)
            used += segs[i].duration

    clips: List[Clip] = []
    for i in sorted(keep):
        s = segs[i]
        score, reason = _score_segment(s, i)
        clips.append(
            Clip(id=f"seg{i}", src_in=s.start, src_out=s.end, score=score, reason=reason)
        )
    return clips


def _llm_pick(transcript: Transcript, target_duration: float, llm) -> List[int]:
    """Ask an LLM which segment indices to keep. Expects a JSON array reply."""
    lines = [
        f"[{i}] ({s.start:.1f}-{s.end:.1f}s) {s.text}"
        for i, s in enumerate(transcript.segments)
    ]
    prompt = (
        "You are a short-form video editor. From the transcript segments below, "
        f"choose the most engaging ones totalling about {target_duration:.0f}s, "
        "ordered for maximum retention (strongest hook first is fine). "
        "Reply ONLY with a JSON array of segment indices.\n\n" + "\n".join(lines)
    )
    raw = llm(prompt)
    start, end = raw.find("["), raw.rfind("]")
    return [int(x) for x in json.loads(raw[start : end + 1])]


# --------------------------------------------------------------------------- #
# Story Agent
# --------------------------------------------------------------------------- #
def build_story(clips: List[Clip], hook_first: bool = True) -> List[Clip]:
    """Order clips into a sequence.

    Short-form heuristic: lead with the strongest line (the hook), then play the
    rest in chronological order so the narrative still reads naturally.
    """
    if not clips:
        return []
    ordered = sorted(clips, key=lambda c: c.src_in)
    if hook_first and len(ordered) > 1:
        hook = max(ordered, key=lambda c: c.score)
        rest = [c for c in ordered if c is not hook]
        ordered = [hook] + rest
    return ordered


# --------------------------------------------------------------------------- #
# Caption Agent
# --------------------------------------------------------------------------- #
def build_captions(timeline: List[Clip], transcript: Transcript) -> List[CaptionWord]:
    """Remap each kept word from source time into the cut (target) timeline."""
    # Flatten all words once.
    all_words = [w for seg in transcript.segments for w in seg.words]
    captions: List[CaptionWord] = []
    offset = 0.0
    for clip in timeline:
        for w in all_words:
            if w.start >= clip.src_in - 1e-6 and w.end <= clip.src_out + 1e-6:
                ns = offset + (w.start - clip.src_in) / clip.speed
                ne = offset + (w.end - clip.src_in) / clip.speed
                if ne > ns and w.text:
                    captions.append(CaptionWord(w.text, round(ns, 3), round(ne, 3)))
        offset += clip.duration
    captions.sort(key=lambda c: c.start)
    return captions


# --------------------------------------------------------------------------- #
# Blended Selection — visual + speech (the cinematic-aware path)
# --------------------------------------------------------------------------- #
def speech_score_in_range(transcript: Transcript, start: float, end: float):
    """Hook-word engagement of the speech inside a time window -> (score, matched)."""
    toks = []
    for seg in transcript.segments:
        for w in seg.words:
            if start - 1e-6 <= w.start < end:
                toks.append(w.text.lower().strip(".,!?;:"))
    if not toks:
        return 0.0, []
    matched = [t for t in toks if t in _HOOK_WORDS]
    hits = len(matched)
    density = hits / len(toks)
    score = 0.6 * min(hits, 3) / 3 + 0.4 * min(density * 3, 1.0)
    return round(min(score, 1.0), 3), matched


def select_blended(shots, transcript: Transcript, visual_w: float, speech_w: float,
                   budget: float) -> List[Clip]:
    """Rank shots by a weighted blend of visual quality and speech engagement.

    `shots` is the output of visual.analyze(): each has start/end/visual_score.
    For cinematic formats visual_w dominates, so a stunning silent shot wins;
    for a vlog speech_w dominates, so the hooky line wins even if it's plain.
    """
    cands = []
    for sh in shots:
        sp, matched = speech_score_in_range(transcript, sh["start"], sh["end"])
        vis = float(sh.get("visual_score", 0.0))
        blended = visual_w * vis + speech_w * sp
        reason = (f"vis {vis:.2f}×{visual_w:.2f} + speech {sp:.2f}×{speech_w:.2f}"
                  f" = {blended:.2f}" + (f" [{', '.join(matched)}]" if matched else ""))
        cands.append({"sh": sh, "vis": vis, "sp": sp, "blended": round(blended, 4),
                      "reason": reason})

    order = sorted(range(len(cands)), key=lambda i: cands[i]["blended"], reverse=True)
    keep, used = [], 0.0
    for i in order:
        if used >= budget:
            break
        keep.append(i)
        used += cands[i]["sh"]["end"] - cands[i]["sh"]["start"]

    clips: List[Clip] = []
    for i in sorted(keep):
        c = cands[i]
        sh = c["sh"]
        clips.append(Clip(
            id=f"shot@{sh['start']:.0f}s", src_in=sh["start"], src_out=sh["end"],
            score=c["blended"], visual_score=c["vis"], speech_score=c["sp"],
            motion=sh.get("motion_norm", 0.5), reason=c["reason"],
        ))
    return clips


# --------------------------------------------------------------------------- #
# Effects Agent — choose a styling effect PER CLIP from its analysis. Every clip
# that is long enough gets *some* camera move: a flat, motionless lead clip is
# exactly what made earlier reels feel like a plain cut-and-slide. The move is
# content-aware — calm clips get a stronger push/pull, already-busy clips get a
# gentle pan that complements (rather than fights) their motion.
# --------------------------------------------------------------------------- #
_CALM_POOL = ["push_in", "pull_out", "pan_right", "pan_left"]   # add life
_BUSY_POOL = ["pan_right", "pan_left", "push_in"]               # gentle, complements


def _next(pool, i, prev):
    """Next effect from a pool, skipping a repeat of the previous clip's."""
    eff = pool[i % len(pool)]
    if eff == prev:
        i += 1
        eff = pool[i % len(pool)]
    return eff, i + 1


def assign_effects(timeline, target):
    """Give every clip a content-aware camera move (the Effects agent).

    Heuristic 'understanding' using the visual-analysis motion metric:
      * ultra-short clips (<1.0s) -> none (a move would read as jitter)
      * busy clips (motion >= 0.6) -> a gentle pan that complements the motion
      * calmer clips               -> a stronger push / pull / pan
    Adjacent clips never share an effect, so a multi-clip reel always shows
    visible, varied movement instead of a flat slide.
    Swap this for an LLM over the same metrics when you want real taste.
    """
    if not target.get("motion"):
        for c in timeline:
            c.effect = "none"
        return timeline

    prev, ci, bi = None, 0, 0
    for c in timeline:
        if c.duration < 1.0:
            eff = "none"
        elif c.motion >= 0.6:        # already moving — complement with a gentle pan
            eff, bi = _next(_BUSY_POOL, bi, prev)
        else:                        # calm — add a stronger move
            eff, ci = _next(_CALM_POOL, ci, prev)
        c.effect = eff
        prev = eff
    return timeline
