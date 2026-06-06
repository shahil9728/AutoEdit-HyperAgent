#!/usr/bin/env python3
"""Generate a visually-differentiated synthetic source + matching transcript.

The clip is six 4-second "shots" with deliberately different visual quality, so
the visual-analysis pass has something real to score:

  shot 0  HERO   sharp, colorful (testsrc)               -> high visual
  shot 1  WEAK   blurred (testsrc2 + heavy gaussian)      -> low  visual
  shot 2  DARK   near-black flat frame                    -> low  visual
  shot 3  HERO   sharp, colorful (testsrc2)               -> high visual
  shot 4  MED    mildly soft (testsrc + light blur)       -> mid  visual
  shot 5  HERO   sharp, extra-saturated (testsrc2 + eq)   -> high visual

The transcript puts the hook-y *speech* on the visually WEAK shots (1 and 4),
so cinematic mode (visual-dominant) and vlog mode (speech-dominant) will pick
different clips from the very same footage — the whole point of the upgrade.
"""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DUR, SEG = 24, 4

# (start, end, sentence) — hooks ("biggest/mistake", "secret/you") land on shots 1 & 4
SENTENCES = [
    (0.0, 3.7, "Sunrise breaks over the northern coastline."),
    (4.0, 7.7, "Here is the biggest mistake every creator makes."),
    (8.0, 11.7, "We drove through the quiet tunnel before dawn."),
    (12.0, 15.7, "The valley opens up beneath the drifting clouds."),
    (16.0, 19.7, "And this is the secret you have been waiting for."),
    (20.0, 23.7, "A final glide over the ridge as the light fades."),
]


def build_transcript() -> dict:
    segments = []
    for start, end, text in SENTENCES:
        toks = text.split()
        step = (end - start) / len(toks)
        words = [{"text": t, "start": round(start + i * step, 3),
                  "end": round(start + (i + 1) * step - 0.02, 3)}
                 for i, t in enumerate(toks)]
        segments.append({"start": start, "end": end, "text": text, "words": words})
    return {"language": "en", "segments": segments}


def main() -> None:
    os.makedirs(os.path.join(HERE, "fixtures"), exist_ok=True)
    src = os.path.join(HERE, "fixtures", "source.mp4")
    tj = os.path.join(HERE, "fixtures", "sample_transcript.json")

    with open(tj, "w") as f:
        json.dump(build_transcript(), f, indent=2)
    print(f"wrote {tj}")

    sz = "size=1280x720:rate=30:duration=4"
    inputs = [
        f"testsrc={sz}",                 # 0 HERO sharp
        f"testsrc2={sz}",                # 1 -> blur (WEAK)
        f"color=c=0x0B0B0B:{sz}",        # 2 DARK
        f"testsrc2={sz}",                # 3 HERO sharp
        f"testsrc={sz}",                 # 4 -> light blur (MED)
        f"testsrc2={sz}",                # 5 -> saturated (HERO)
    ]
    fc = (
        "[0:v]setsar=1,format=yuv420p[v0];"
        "[1:v]gblur=sigma=16,setsar=1,format=yuv420p[v1];"
        "[2:v]setsar=1,format=yuv420p[v2];"
        "[3:v]setsar=1,format=yuv420p[v3];"
        "[4:v]gblur=sigma=6,setsar=1,format=yuv420p[v4];"
        "[5:v]eq=saturation=1.8,setsar=1,format=yuv420p[v5];"
        "[v0][v1][v2][v3][v4][v5]concat=n=6:v=1:a=0[vout]"
    )
    cmd = ["ffmpeg", "-y"]
    for src_def in inputs:
        cmd += ["-f", "lavfi", "-i", src_def]
    cmd += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={DUR}"]
    cmd += ["-filter_complex", fc, "-map", "[vout]", "-map", "6:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", src]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print(f"wrote {src}")


if __name__ == "__main__":
    main()
