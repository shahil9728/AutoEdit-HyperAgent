#!/usr/bin/env python3
"""autoedit CLI — the 'single click'.

    python3 cli.py <video> --formats short square --transcript fixtures/sample.json

With no --transcript and faster-whisper installed, real ASR runs automatically
(see README). Outputs land in --outdir (default: out/).
"""

import argparse
import sys

from autoedit import run_pipeline


def main() -> int:
    p = argparse.ArgumentParser(description="Autonomous short-form video editor (MVP)")
    p.add_argument("video", help="path to the source video")
    p.add_argument("--formats", nargs="+", default=["short"],
                   help="target platforms: short reel tiktok square vlog cinematic")
    p.add_argument("--transcript", default=None,
                   help="word-timed transcript JSON (skip to use real ASR)")
    p.add_argument("--outdir", default="out", help="output directory")
    p.add_argument("--budget", type=float, default=None,
                   help="target cut length in seconds (smaller = more aggressive selection)")
    p.add_argument("--no-visual", action="store_true",
                   help="skip visual analysis; rank on transcript only (talking-head mode)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    asr = None
    if not args.transcript:
        import importlib.util
        if importlib.util.find_spec("faster_whisper") is not None:
            from adapters.whisper_asr import transcribe as asr  # optional, real ASR
        else:
            print("[note] No --transcript and faster-whisper not installed -> "
                  "running VISUAL-ONLY (no speech-aware selection or captions). "
                  "Pass --transcript or install faster-whisper for speech.",
                  file=sys.stderr)

    results = run_pipeline(
        source=args.video,
        formats=args.formats,
        transcript_path=args.transcript,
        outdir=args.outdir,
        asr=asr,
        budget=args.budget,
        use_visual=not args.no_visual,
        verbose=not args.quiet,
    )
    print("Done. Outputs:")
    for r in results:
        print(f"  [{r['format']}] {r['video']}  ({r['duration']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
