"""The Editing + Platform agents: turn an EDL into a rendered MP4 via ffmpeg.

Two passes (kept separate for debuggability):
  Pass 1  cut every clip, reframe to the target aspect, concatenate  -> body.mp4
  Pass 2  burn word-pop captions (and optionally mix music)          -> final.mp4

Reframing is a saliency-free centre/anchor crop driven by Clip.focus_x. Real
subject tracking (face/saliency) would set focus_x per-frame; see README.
"""

from __future__ import annotations

import os
import subprocess
from typing import List

from .edl import EDL, CaptionWord

# Pin ffmpeg/x264 threads low: x264 allocates frame buffers per thread, which is
# the main RAM driver. 1 keeps a 512MB box alive; raise on a bigger instance.
_THREADS = os.environ.get("AUTOEDIT_THREADS", "1")


# --------------------------------------------------------------------------- #
# Pass 1 — cut + reframe + concat
# --------------------------------------------------------------------------- #
def _reframe_chain(target: dict) -> str:
    """Crop the source to the target aspect (anchored at focus_x) then scale.

    Expressed with ffmpeg expressions so it works for any source resolution:
    crop width = ih * (W/H), clamped to iw; x = (iw - cropw) * focus.
    """
    w, h = target["width"], target["height"]
    ar = w / h
    # Even-width crop (yuv420p needs mod-2 dims), anchored at focus_x, then scale.
    cropw = f"floor(min(iw\\,ih*{ar:.6f})/2)*2"
    x = f"(iw-{cropw})*{{focus}}"
    return (
        f"crop={cropw}:ih:{x}:0,"
        f"scale={w}:{h}:force_original_aspect_ratio=disable,"
        f"setsar=1,fps={target['fps']},format=yuv420p"
    )


def build_pass1_cmd(edl: EDL, body_name: str) -> List[str]:
    # Absolute path: ffmpeg runs with cwd=workdir, source lives elsewhere.
    src = os.path.abspath(edl.source.path)
    chain_tpl = _reframe_chain(edl.target)
    n = len(edl.timeline)
    parts: List[str] = []

    # Fan the single input out to one branch per clip (ffmpeg won't auto-split).
    parts.append("[0:v]split=" + str(n) + "".join(f"[vs{i}]" for i in range(n)))
    if edl.source.has_audio:
        parts.append("[0:a]asplit=" + str(n) + "".join(f"[as{i}]" for i in range(n)))

    labels: List[int] = []
    for i, c in enumerate(edl.timeline):
        chain = chain_tpl.format(focus=c.focus_x)
        parts.append(
            f"[vs{i}]trim=start={c.src_in:.3f}:end={c.src_out:.3f},"
            f"setpts=PTS-STARTPTS,{chain}[v{i}]"
        )
        if edl.source.has_audio:
            parts.append(
                f"[as{i}]atrim=start={c.src_in:.3f}:end={c.src_out:.3f},"
                f"asetpts=PTS-STARTPTS[a{i}]"
            )
        labels.append(i)

    if edl.source.has_audio:
        concat_in = "".join(f"[v{i}][a{i}]" for i in labels)
        parts.append(f"{concat_in}concat=n={n}:v=1:a=1[vout][aout]")
        maps = ["-map", "[vout]", "-map", "[aout]"]
    else:
        concat_in = "".join(f"[v{i}]" for i in labels)
        parts.append(f"{concat_in}concat=n={n}:v=1:a=0[vout]")
        maps = ["-map", "[vout]"]

    filtergraph = ";".join(parts)
    cmd = ["ffmpeg", "-y", "-threads", _THREADS, "-i", src, "-filter_complex", filtergraph, *maps,
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
           "-pix_fmt", "yuv420p"]
    if edl.source.has_audio:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    cmd += ["-movflags", "+faststart", body_name]
    return cmd


# --------------------------------------------------------------------------- #
# Captions — generate an ASS file with a TikTok-style active-word highlight
# --------------------------------------------------------------------------- #
def _ass_time(t: float) -> str:
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _group_phrases(words: List[CaptionWord], max_words=3, max_gap=0.7):
    phrases, cur = [], []
    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            ends_sentence = cur[-1].text.endswith((".", "!", "?"))
            if len(cur) >= max_words or gap > max_gap or ends_sentence:
                phrases.append(cur)
                cur = []
        cur.append(w)
    if cur:
        phrases.append(cur)
    return phrases


def generate_ass(edl: EDL, ass_path: str) -> None:
    t = edl.target
    style = t.get("caption_style", "wordpop")
    H = t["height"]
    font_size = max(28, round(H * 0.052))
    margin_v = round(H * 0.16)
    outline = max(2, round(H * 0.0035))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {t['width']}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,DejaVu Sans,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{outline},2,2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    HILITE = r"{\c&H0000FFFF&\b1\fscx108\fscy108}"  # bright yellow, bold, +8%
    RESET = r"{\r}"
    lines: List[str] = []

    if style == "none":
        with open(ass_path, "w") as f:
            f.write(header)
        return

    for phrase in _group_phrases(edl.captions):
        upper = [w.text.upper().replace("{", "(").replace("}", ")") for w in phrase]
        for j, w in enumerate(phrase):
            start = w.start
            end = phrase[j + 1].start if j + 1 < len(phrase) else w.end + 0.08
            if end <= start:
                end = start + 0.25
            if style == "wordpop":
                rendered = " ".join(
                    (HILITE + tok + RESET) if k == j else tok
                    for k, tok in enumerate(upper)
                )
            else:  # "line" — whole phrase, no per-word pop
                rendered = " ".join(upper)
                if j > 0:
                    continue  # one event per phrase
                end = phrase[-1].end + 0.08
            lines.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Base,,0,0,0,,{rendered}"
            )

    with open(ass_path, "w") as f:
        f.write(header + "\n".join(lines) + "\n")


def build_pass2_cmd(body_name: str, ass_name: str, final_name: str,
                    has_audio: bool) -> List[str]:
    cmd = ["ffmpeg", "-y", "-threads", _THREADS, "-i", body_name,
           "-vf", f"subtitles={ass_name}",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
           "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "copy"] if has_audio else ["-an"]
    cmd += [final_name]
    return cmd


# --------------------------------------------------------------------------- #
# Orchestration of the two ffmpeg passes
# --------------------------------------------------------------------------- #
def render(edl: EDL, workdir: str, basename: str) -> str:
    os.makedirs(workdir, exist_ok=True)
    tag = edl.target["name"]
    body = f"{basename}_{tag}_body.mp4"
    ass = f"{basename}_{tag}.ass"
    final = f"{basename}_{tag}.mp4"

    _run(build_pass1_cmd(edl, body), workdir)
    generate_ass(edl, os.path.join(workdir, ass))
    has_caps = edl.target.get("caption_style", "wordpop") != "none" and edl.captions
    if has_caps:
        _run(build_pass2_cmd(body, ass, final, edl.source.has_audio), workdir)
        os.remove(os.path.join(workdir, body))
    else:
        os.replace(os.path.join(workdir, body), os.path.join(workdir, final))
    return os.path.join(workdir, final)


def _run(cmd: List[str], cwd: str) -> None:
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{tail}")
