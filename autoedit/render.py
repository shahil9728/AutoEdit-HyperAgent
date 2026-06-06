"""The Editing + Platform agents: turn an EDL into a rendered MP4 via ffmpeg.

Approach (robust + fast):
  Pass 1  cut each selected clip to its own file using fast input-seeking
          (-ss/-t), reframing to the target aspect          -> seg0.mp4, seg1...
  Pass 2  concat the segment files                          -> body.mp4
  Pass 3  burn word-pop captions (if any)                   -> final.mp4

Why per-segment files instead of one `split -> trim -> concat` graph: that graph
*deadlocks* ffmpeg once the source has enough frames (split must feed every
branch while concat consumes them in order, so later branches' buffers fill and
the decode stalls). Cutting each segment independently avoids that entirely and
only decodes the ranges we actually use.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import List

from .edl import EDL, CaptionWord

_THREADS = os.environ.get("AUTOEDIT_THREADS", "1")
_TIMEOUT = int(os.environ.get("AUTOEDIT_FFMPEG_TIMEOUT", "300"))
_NICE = ["nice", "-n", "19"] if shutil.which("nice") else []


def _log(msg: str) -> None:
    print(f"[render] {msg}", flush=True)


def _run(cmd: List[str], cwd: str, label: str) -> None:
    _log(f"{label}: ffmpeg {' '.join(cmd[:16])} …")
    try:
        proc = subprocess.run(_NICE + cmd, cwd=cwd, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{label} timed out (>{_TIMEOUT}s)")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"{label} ffmpeg failed ({proc.returncode}):\n{tail}")
    _log(f"{label}: done")


# --------------------------------------------------------------------------- #
# reframe
# --------------------------------------------------------------------------- #
def _reframe_chain(target: dict) -> str:
    w, h = target["width"], target["height"]
    ar = w / h
    cropw = f"floor(min(iw\\,ih*{ar:.6f})/2)*2"
    x = f"(iw-{cropw})*{{focus}}"
    return (f"crop={cropw}:ih:{x}:0,"
            f"scale={w}:{h}:force_original_aspect_ratio=disable,"
            f"setsar=1,fps={target['fps']},format=yuv420p")


def _segment_cmd(edl: EDL, clip, chain_tpl: str, out_name: str) -> List[str]:
    src = os.path.abspath(edl.source.path)
    chain = chain_tpl.format(focus=clip.focus_x)
    dur = max(0.05, clip.src_out - clip.src_in)
    # -ss before -i = fast seek; -t = duration. Decodes only this range.
    cmd = ["ffmpeg", "-y", "-nostdin", "-threads", _THREADS,
           "-ss", f"{clip.src_in:.3f}", "-i", src, "-t", f"{dur:.3f}"]
    if edl.source.has_audio:
        fc = f"[0:v]{chain}[v];[0:a]aresample=44100,aformat=channel_layouts=stereo[a]"
        cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                "-c:a", "aac", "-ar", "44100", "-ac", "2"]
    else:
        cmd += ["-filter_complex", f"[0:v]{chain}[v]", "-map", "[v]"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-pix_fmt", "yuv420p", out_name]
    return cmd


def _concat_cmd(edl: EDL, segs: List[str], body_name: str) -> List[str]:
    n = len(segs)
    inputs: List[str] = []
    for s in segs:
        inputs += ["-i", s]
    if edl.source.has_audio:
        streams = "".join(f"[{i}:v][{i}:a]" for i in range(n))
        fc = f"{streams}concat=n={n}:v=1:a=1[v][a]"
        maps = ["-map", "[v]", "-map", "[a]", "-c:a", "aac"]
    else:
        streams = "".join(f"[{i}:v]" for i in range(n))
        fc = f"{streams}concat=n={n}:v=1:a=0[v]"
        maps = ["-map", "[v]"]
    return (["ffmpeg", "-y", "-nostdin", "-threads", _THREADS, *inputs,
             "-filter_complex", fc, *maps, "-c:v", "libx264", "-preset", "ultrafast",
             "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", body_name])


# --------------------------------------------------------------------------- #
# captions (ASS with TikTok-style active-word highlight)
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

    HILITE = r"{\c&H0000FFFF&\b1\fscx108\fscy108}"
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
            else:
                rendered = " ".join(upper)
                if j > 0:
                    continue
                end = phrase[-1].end + 0.08
            lines.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Base,,0,0,0,,{rendered}"
            )

    with open(ass_path, "w") as f:
        f.write(header + "\n".join(lines) + "\n")


def _caption_cmd(body_name: str, ass_name: str, final_name: str, has_audio: bool) -> List[str]:
    cmd = ["ffmpeg", "-y", "-nostdin", "-threads", _THREADS, "-i", body_name,
           "-vf", f"subtitles={ass_name}", "-c:v", "libx264", "-preset", "ultrafast",
           "-crf", "22", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "copy"] if has_audio else ["-an"]
    cmd += [final_name]
    return cmd


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def render(edl: EDL, workdir: str, basename: str) -> str:
    os.makedirs(workdir, exist_ok=True)
    tag = edl.target["name"]
    body = f"{basename}_{tag}_body.mp4"
    ass = f"{basename}_{tag}.ass"
    final = f"{basename}_{tag}.mp4"
    n = len(edl.timeline)

    chain_tpl = _reframe_chain(edl.target)
    segs: List[str] = []
    for i, clip in enumerate(edl.timeline):
        seg = f"{basename}_{tag}_seg{i}.mp4"
        _run(_segment_cmd(edl, clip, chain_tpl, seg), workdir,
             f"cut {i + 1}/{n} ({clip.src_in:.1f}-{clip.src_out:.1f}s)")
        segs.append(seg)

    if len(segs) == 1:
        os.replace(os.path.join(workdir, segs[0]), os.path.join(workdir, body))
    else:
        _run(_concat_cmd(edl, segs, body), workdir, f"concat {n} segments")
        for s in segs:
            try:
                os.remove(os.path.join(workdir, s))
            except OSError:
                pass

    generate_ass(edl, os.path.join(workdir, ass))
    has_caps = edl.target.get("caption_style", "wordpop") != "none" and edl.captions
    if has_caps:
        _run(_caption_cmd(body, ass, final, edl.source.has_audio), workdir, "burn captions")
        os.remove(os.path.join(workdir, body))
    else:
        os.replace(os.path.join(workdir, body), os.path.join(workdir, final))
    _log(f"final -> {final}")
    return os.path.join(workdir, final)
