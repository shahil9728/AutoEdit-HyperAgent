"""Platform presets — the Platform Optimization Agent's knowledge of each target.

Each preset carries:
  * resolution / aspect / fps / max_duration / caption_style
  * (visual_weight, speech_weight) — how much Selection trusts picture vs speech
  * a "style" (motion + grade + transition set) — the look of the final edit

Set AUTOEDIT_FX=0 to disable motion/grade/transitions (plain cuts).
Set AUTOEDIT_OUTPUT_SCALE=1 on a bigger box for full-resolution output.
"""

import os
from typing import Any, Dict

PRESETS: Dict[str, Dict[str, Any]] = {
    "short": dict(name="short", platform="YouTube Shorts", aspect="9:16",
                  width=1080, height=1920, fps=30, max_duration=60, caption_style="wordpop"),
    "reel": dict(name="reel", platform="Instagram Reel", aspect="9:16",
                 width=1080, height=1920, fps=30, max_duration=90, caption_style="wordpop"),
    "tiktok": dict(name="tiktok", platform="TikTok", aspect="9:16",
                   width=1080, height=1920, fps=30, max_duration=60, caption_style="wordpop"),
    "square": dict(name="square", platform="Feed (1:1)", aspect="1:1",
                   width=1080, height=1080, fps=30, max_duration=60, caption_style="wordpop"),
    "vlog": dict(name="vlog", platform="YouTube", aspect="16:9",
                 width=1920, height=1080, fps=30, max_duration=900, caption_style="line"),
    "cinematic": dict(name="cinematic", platform="YouTube / web", aspect="16:9",
                      width=1920, height=1080, fps=30, max_duration=240, caption_style="none"),
    "travel": dict(name="travel", platform="Travel montage", aspect="16:9",
                   width=1920, height=1080, fps=30, max_duration=180, caption_style="none"),
}

# (visual_weight, speech_weight) — must sum to 1.0
_WEIGHTS = {
    "short": (0.45, 0.55), "reel": (0.45, 0.55), "tiktok": (0.45, 0.55),
    "square": (0.50, 0.50), "vlog": (0.20, 0.80),
    "cinematic": (0.85, 0.15), "travel": (0.80, 0.20),
}

# Look & feel: motion (punch-in zoom), grade (colour), transition palette (xfade)
_PUNCHY = ["slideleft", "slideright", "fade", "zoomin", "circleopen"]
_FX = {
    "short": dict(motion=True, grade=True, transitions=_PUNCHY),
    "reel": dict(motion=True, grade=True, transitions=_PUNCHY),
    "tiktok": dict(motion=True, grade=True, transitions=_PUNCHY),
    "square": dict(motion=True, grade=True, transitions=["fade", "circleopen", "slideleft"]),
    "vlog": dict(motion=True, grade=True, transitions=["fade"]),
    "cinematic": dict(motion=True, grade=True, transitions=["dissolve", "fade", "fadeblack"]),
    "travel": dict(motion=True, grade=True, transitions=["slideup", "dissolve", "circleopen", "slideleft"]),
}


def get_preset(name: str) -> Dict[str, Any]:
    key = name.lower().strip()
    if key not in PRESETS:
        raise ValueError(f"Unknown format '{name}'. Available: {', '.join(sorted(PRESETS))}")
    p = dict(PRESETS[key])

    scale = float(os.environ.get("AUTOEDIT_OUTPUT_SCALE", "0.7"))
    if scale != 1.0:
        p["width"] = max(2, int(round(p["width"] * scale / 2)) * 2)
        p["height"] = max(2, int(round(p["height"] * scale / 2)) * 2)

    vw, sw = _WEIGHTS.get(key, (0.5, 0.5))
    p["visual_weight"], p["speech_weight"] = vw, sw

    fx = _FX.get(key, dict(motion=True, grade=True, transitions=["fade"]))
    if os.environ.get("AUTOEDIT_FX", "1") == "0":
        p["motion"], p["grade"], p["transitions"] = False, False, []
    else:
        p["motion"], p["grade"], p["transitions"] = fx["motion"], fx["grade"], list(fx["transitions"])
    return p
