"""The Editing + Style agents: turn an EDL into a polished MP4 via ffmpeg.

Pipeline:
  Pass 1  cut each clip (fast -ss/-t seek) and STYLE it: reframe to target aspect,
          apply the per-clip EFFECT chosen by the Effects agent (push/pull/pan or
          none), and a colour grade                          -> seg_i.mp4
  Pass 2  join clips — xfade transitions when there are no burned captions
          (transitions shift timing, which would desync captions), else concat
                                                              -> body.mp4
  Pass 3  burn word-pop captions if any                      -> final.mp4

Per-clip cutting (instead of one split->trim->concat graph) avoids the ffmpeg
split/concat deadlock and only decodes the ranges actually used.
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
_ZOOM = float(os.environ.get("AUTOEDIT_ZOOM", "1.15"))       # push/pull max scale
_PANZ = float(os.environ.get("AUTOEDIT_PAN_ZOOM", "1.12"))   # zoom held during a pan


def _log(msg: str) -> None:
    print(f"[render] {msg}", flush=True)


def _run(cmd: List[str], cwd: str, label: str) -> None:
    _log(f"{label}: ffmpeg {' '.join(cmd[:14])} …")
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
# per-clip styling: reframe + camera move + colour/light look
# --------------------------------------------------------------------------- #
def _look_filters(look: str, s: float):
    """Map a look name to (linear ffmpeg filter list, needs_bloom).

    `s` scales intensity (AUTOEDIT_LOOK_STRENGTH). Bloom (a highlight glow) is a
    split/blend subgraph the caller splices in, so it is flagged here, not
    returned as a linear filter.
    """
    if look == "vibrant":
        return ([f"eq=contrast={1 + 0.14 * s:.3f}:saturation={1 + 0.30 * s:.3f}:gamma={1 - 0.02 * s:.3f}",
                 f"vibrance=intensity={0.30 * s:.3f}"], False)
    if look == "teal_orange":
        return ([f"colorbalance=rs={-0.06 * s:.3f}:bs={0.06 * s:.3f}:rh={0.07 * s:.3f}:bh={-0.07 * s:.3f}",
                 f"eq=contrast={1 + 0.10 * s:.3f}:saturation={1 + 0.10 * s:.3f}"], False)
    if look == "warm_golden":
        return ([f"colortemperature=temperature={int(5500 - 900 * s)}:mix={min(1.0, 0.70 * s):.3f}",
                 f"eq=contrast={1 + 0.08 * s:.3f}:saturation={1 + 0.12 * s:.3f}:brightness={0.02 * s:.3f}"], False)
    if look == "moody_cool":
        return ([f"colortemperature=temperature={int(6500 + 2000 * s)}:mix={min(1.0, 0.55 * s):.3f}",
                 f"eq=contrast={1 + 0.15 * s:.3f}:saturation={1 - 0.20 * s:.3f}:gamma={1 - 0.04 * s:.3f}"], False)
    if look == "lift_glow":          # dark footage: lift shadows + warm + soft glow
        return ([f"eq=brightness={0.07 * s:.3f}:gamma={1 + 0.14 * s:.3f}:contrast={1 + 0.05 * s:.3f}:saturation={1 + 0.06 * s:.3f}",
                 f"colortemperature=temperature={int(5400 - 300 * s)}:mix={min(1.0, 0.40 * s):.3f}"], True)
    if look == "bloom_warm":         # bright/scenic: warm + dreamy highlight bloom
        return ([f"eq=contrast={1 + 0.06 * s:.3f}:saturation={1 + 0.10 * s:.3f}",
                 f"colortemperature=temperature={int(5400 - 300 * s)}:mix={min(1.0, 0.35 * s):.3f}"], True)
    if look == "film_vintage":
        return (["curves=preset=vintage",
                 f"eq=contrast={1 + 0.05 * s:.3f}:saturation={1 - 0.05 * s:.3f}",
                 f"noise=alls={max(1, int(7 * s))}:allf=t"], False)
    if look == "mono":
        return (["hue=s=0", f"eq=contrast={1 + 0.16 * s:.3f}:gamma={1 - 0.03 * s:.3f}"], False)
    # neutral / none -> the classic subtle grade
    return (["eq=contrast=1.06:saturation=1.12:gamma=0.98"], False)


def _zoompan(effect: str, dur: float, w: int, h: int, fps: int):
    frames = max(1, int(round(dur * fps)))
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    speed = (_ZOOM - 1.0) / frames
    if effect == "push_in":
        return f"zoompan=z='min(1+{speed:.6f}*on,{_ZOOM})':x='{cx}':y='{cy}':d=1:s={w}x{h}:fps={fps}"
    if effect == "pull_out":
        return f"zoompan=z='max({_ZOOM}-{speed:.6f}*on,1.0)':x='{cx}':y='{cy}':d=1:s={w}x{h}:fps={fps}"
    if effect == "pan_left":
        return f"zoompan=z='{_PANZ}':x='(iw-iw/zoom)*on/{frames}':y='{cy}':d=1:s={w}x{h}:fps={fps}"
    if effect == "pan_right":
        return f"zoompan=z='{_PANZ}':x='(iw-iw/zoom)*(1-on/{frames})':y='{cy}':d=1:s={w}x{h}:fps={fps}"
    return None


def _video_graph(target: dict, clip, dur: float, intro: bool) -> str:
    """Full per-clip video filtergraph from [0:v] to [v]:
    reframe -> camera move -> colour/light look -> (bloom) -> intro fade -> vignette.
    """
    w, h, fps = target["width"], target["height"], target["fps"]
    ar = w / h
    cropw = f"floor(min(iw\\,ih*{ar:.6f})/2)*2"
    x = f"(iw-{cropw})*{clip.focus_x}"
    pre = [f"crop={cropw}:ih:{x}:0", f"scale={w}:{h}", "setsar=1", f"fps={fps}", "format=yuv420p"]

    if clip.effect and clip.effect != "none":
        zp = _zoompan(clip.effect, dur, w, h, fps)
        if zp:
            pre.append(zp)

    graded = bool(target.get("grade"))
    bloom = False
    strength = float(target.get("look_strength", 1.0))
    if graded:
        look_fs, bloom = _look_filters(clip.look or "neutral", strength)
        pre += look_fs

    post = []
    if intro:                       # cinematic exposure ramp-in on the opening clip
        post.append("fade=t=in:st=0:d=0.4")
    if graded:
        vig = "vignette=PI/4.5" if clip.look in ("moody_cool", "film_vintage") else "vignette=PI/6"
        post.append(vig)

    if not bloom:
        chain = ",".join(pre + post)
        return f"[0:v]{chain}[v]"

    # Bloom = highlight glow: split, blur one copy, screen-blend it back, then post.
    base = ",".join(pre)
    op = min(0.5, 0.38 * strength)
    g = (f"[0:v]{base}[b];"
         f"[b]split[b1][b2];[b2]gblur=sigma=14[bb];")
    if post:
        g += (f"[b1][bb]blend=all_mode=screen:all_opacity={op:.3f}[bl];"
              f"[bl]{','.join(post)}[v]")
    else:
        g += f"[b1][bb]blend=all_mode=screen:all_opacity={op:.3f}[v]"
    return g


def _segment_cmd(edl: EDL, clip, out_name: str, intro: bool = False) -> List[str]:
    src = os.path.abspath(edl.source.path)
    dur = max(0.05, clip.src_out - clip.src_in)
    vgraph = _video_graph(edl.target, clip, dur, intro)
    cmd = ["ffmpeg", "-y", "-nostdin", "-threads", _THREADS,
           "-ss", f"{clip.src_in:.3f}", "-i", src, "-t", f"{dur:.3f}"]
    if edl.source.has_audio:
        fc = f"{vgraph};[0:a]aresample=44100,aformat=channel_layouts=stereo[a]"
        cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                "-c:a", "aac", "-ar", "44100", "-ac", "2"]
    else:
        cmd += ["-filter_complex", vgraph, "-map", "[v]"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-pix_fmt", "yuv420p", out_name]
    return cmd


# --------------------------------------------------------------------------- #
# joining clips
# --------------------------------------------------------------------------- #
def _trans_dur(durs: List[float]) -> float:
    m = min(durs) if durs else 1.0
    d = min(0.5, m * 0.4)
    if d >= m:
        d = m * 0.3
    return round(max(0.2, min(d, m - 0.05)), 3)


def _concat_cmd(edl: EDL, segs: List[str], body_name: str) -> List[str]:
    n = len(segs)
    inputs: List[str] = []
    for s in segs:
        inputs += ["-i", s]
    if edl.source.has_audio:
        streams = "".join(f"[{i}:v][{i}:a]" for i in range(n))
        fc, maps = f"{streams}concat=n={n}:v=1:a=1[v][a]", ["-map", "[v]", "-map", "[a]", "-c:a", "aac"]
    else:
        streams = "".join(f"[{i}:v]" for i in range(n))
        fc, maps = f"{streams}concat=n={n}:v=1:a=0[v]", ["-map", "[v]"]
    return ["ffmpeg", "-y", "-nostdin", "-threads", _THREADS, *inputs, "-filter_complex", fc,
            *maps, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", body_name]


def _xfade_cmd(edl: EDL, segs: List[str], durs: List[float], body_name: str) -> List[str]:
    n = len(segs)
    d = _trans_dur(durs)
    trans = edl.target.get("transitions") or ["fade"]
    inputs: List[str] = []
    for s in segs:
        inputs += ["-i", s]

    vparts, prev, cum = [], "[0:v]", 0.0
    for k in range(1, n):
        cum += durs[k - 1]
        offset = cum - k * d
        t = trans[(k - 1) % len(trans)]
        # The Style agent can request a white flash punching into a clip.
        if k < len(edl.timeline) and getattr(edl.timeline[k], "flash_in", False):
            t = "fadewhite"
        out = "[vout]" if k == n - 1 else f"[vx{k}]"
        vparts.append(f"{prev}[{k}:v]xfade=transition={t}:duration={d}:offset={offset:.3f}{out}")
        prev = out
    fc = ";".join(vparts)
    maps = ["-map", "[vout]"]

    if edl.source.has_audio:
        aparts, aprev = [], "[0:a]"
        for k in range(1, n):
            aout = "[aout]" if k == n - 1 else f"[ax{k}]"
            aparts.append(f"{aprev}[{k}:a]acrossfade=d={d}{aout}")
            aprev = aout
        fc += ";" + ";".join(aparts)
        maps += ["-map", "[aout]", "-c:a", "aac"]

    return ["ffmpeg", "-y", "-nostdin", "-threads", _THREADS, *inputs, "-filter_complex", fc,
            *maps, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", body_name]


# --------------------------------------------------------------------------- #
# captions (ASS, TikTok-style active-word highlight)
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
                rendered = " ".join((HILITE + tok + RESET) if k == j else tok
                                    for k, tok in enumerate(upper))
            else:
                rendered = " ".join(upper)
                if j > 0:
                    continue
                end = phrase[-1].end + 0.08
            lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Base,,0,0,0,,{rendered}")
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

    looks_on = bool(edl.target.get("looks_on"))
    segs, durs = [], []
    for i, clip in enumerate(edl.timeline):
        seg = f"{basename}_{tag}_seg{i}.mp4"
        intro = i == 0 and looks_on and n > 1
        _run(_segment_cmd(edl, clip, seg, intro=intro), workdir,
             f"cut+style {i + 1}/{n} fx={clip.effect} look={clip.look} "
             f"({clip.src_in:.1f}-{clip.src_out:.1f}s)")
        segs.append(seg)
        durs.append(max(0.05, clip.src_out - clip.src_in))

    use_xfade = n > 1 and (edl.target.get("transitions") or []) and not edl.captions
    if n == 1:
        os.replace(os.path.join(workdir, segs[0]), os.path.join(workdir, body))
    elif use_xfade:
        _run(_xfade_cmd(edl, segs, durs, body), workdir, f"transitions ({n} clips)")
    else:
        _run(_concat_cmd(edl, segs, body), workdir, f"concat ({n} clips)")
    if n > 1:
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
