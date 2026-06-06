"""Visual Analysis — judge the *picture*, not just the speech.

This is the upgrade that makes the system work for cinematic / b-roll footage,
where the best moments are visual and there may be no dialogue at all.

It runs entirely on ffmpeg (no numpy / OpenCV), so it works anywhere ffmpeg is:

  * shot detection      ffmpeg scene-score (`select=gt(scene,T)`) -> shot ranges
  * sharpness / focus    Laplacian via `convolution` + mean energy (`signalstats`)
  * exposure             mean luma (signalstats YAVG), penalise dark / blown-out
  * saturation           signalstats SATAVG (colour punch)
  * motion               inter-frame difference (`tblend=difference`) energy

Per-shot metrics are normalised across the clip and blended into a 0..1
`visual_score`. The Selection agent then weights this against the speech score
according to the chosen format (cinematic = visual-dominant; vlog = speech).
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Tuple

_YAVG = "lavfi.signalstats.YAVG"
_SAT = "lavfi.signalstats.SATAVG"
_LAPLACIAN = "0 -1 0 -1 4 -1 0 -1 0"  # 4-neighbour discrete Laplacian kernel
_PTS = re.compile(r"pts_time:([0-9.]+)")


# --------------------------------------------------------------------------- #
# low-level ffmpeg passes
# --------------------------------------------------------------------------- #
def _run_meta_pass(path: str, vf: str, fps: int) -> List[Dict[str, float]]:
    """Run one analysis pass; return per-sampled-frame dicts {t, lavfi keys...}."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp.close()
    full_vf = f"fps={fps},{vf},metadata=print:file={tmp.name}"
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-i", path,
           "-vf", full_vf, "-an", "-f", "null", "-"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    samples: List[Dict[str, float]] = []
    cur: Dict[str, float] = None
    with open(tmp.name) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("frame:"):
                if cur is not None:
                    samples.append(cur)
                m = _PTS.search(line)
                cur = {"t": float(m.group(1)) if m else 0.0}
            elif line.startswith("lavfi.") and "=" in line and cur is not None:
                k, v = line.split("=", 1)
                try:
                    cur[k.strip()] = float(v)
                except ValueError:
                    pass
    if cur is not None:
        samples.append(cur)
    os.unlink(tmp.name)
    return samples


def detect_shots(path: str, duration: float, threshold: float = 0.4,
                 min_len: float = 1.5, max_len: float = 8.0) -> List[Tuple[float, float]]:
    """Scene-cut detection via ffmpeg scene score; long takes are subdivided."""
    cmd = ["ffmpeg", "-hide_banner", "-i", path,
           "-vf", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"]
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    cuts = sorted(float(t) for t in _PTS.findall(p.stderr))

    bounds = [0.0]
    for c in cuts:
        if c - bounds[-1] >= min_len and duration - c >= min_len:
            bounds.append(c)
    bounds.append(duration)

    shots: List[Tuple[float, float]] = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e - s > max_len:  # split a long take into even windows
            k = math.ceil((e - s) / max_len)
            step = (e - s) / k
            for j in range(k):
                shots.append((s + j * step, e if j == k - 1 else s + (j + 1) * step))
        else:
            shots.append((s, e))
    return shots


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def analyze(path: str, duration: float, fps: int = 4) -> List[Dict]:
    """Return one dict per shot with raw metrics + a normalised visual_score."""
    shots = detect_shots(path, duration)
    raw = _run_meta_pass(path, "signalstats", fps)
    sharp = _run_meta_pass(path, f"format=gray,convolution={_LAPLACIAN},signalstats", fps)
    motion = _run_meta_pass(path, "tblend=all_mode=difference,signalstats", fps)

    def avg(samples: List[Dict[str, float]], key: str, s: float, e: float) -> float:
        vals = [smp[key] for smp in samples if key in smp and s - 1e-6 <= smp["t"] < e]
        return sum(vals) / len(vals) if vals else 0.0

    rows: List[Dict] = []
    for s, e in shots:
        rows.append({
            "start": round(s, 2), "end": round(e, 2),
            "brightness": round(avg(raw, _YAVG, s, e), 1),
            "saturation": round(avg(raw, _SAT, s, e), 1),
            "sharpness": round(avg(sharp, _YAVG, s, e), 3),
            "motion": round(avg(motion, _YAVG, s, e), 3),
        })

    def norm(vals: List[float]) -> List[float]:
        lo, hi = min(vals), max(vals)
        return [0.5 if hi - lo < 1e-9 else (v - lo) / (hi - lo) for v in vals]

    if rows:
        sN = norm([r["sharpness"] for r in rows])
        cN = norm([r["saturation"] for r in rows])
        mN = norm([r["motion"] for r in rows])
        for i, r in enumerate(rows):
            # exposure: ideal mid-luma ~110; punish dark and blown-out shots
            expo = max(0.0, 1.0 - abs(r["brightness"] - 110.0) / 110.0)
            vs = 0.45 * sN[i] + 0.30 * expo + 0.15 * cN[i] + 0.10 * mN[i]
            r["exposure"] = round(expo, 3)
            r["visual_score"] = round(min(1.0, max(0.0, vs)), 3)
    return rows
