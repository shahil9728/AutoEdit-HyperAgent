"""The orchestrator — wires the agents together around one EDL per target.

Flow: analyze the footage ONCE (technical probe + transcript + visual metrics),
then for each requested platform build a tailored EDL (blended selection with
that platform's visual/speech weighting) and render it.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

from . import agents, visual
from .edl import EDL
from .presets import get_preset


def run_pipeline(
    source: str,
    formats: List[str],
    transcript_path: Optional[str] = None,
    outdir: str = "out",
    asr: Optional[Callable[[str], dict]] = None,
    llm: Optional[Callable[[str], str]] = None,
    budget: Optional[float] = None,
    use_visual: bool = True,
    verbose: bool = True,
) -> List[dict]:
    os.makedirs(outdir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(source))[0]

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # 1. Analysis Agent --------------------------------------------------- #
    log("[1/6] Analysis     : probe + transcribe + visual metrics")
    meta = agents.probe_source(source)
    transcript = agents.transcribe(source, transcript_path, asr)

    shots = []
    if use_visual:
        shots = visual.analyze(source, meta.duration)
    log(f"        {meta.width}x{meta.height} @ {meta.fps}fps, {meta.duration:.1f}s, "
        f"audio={meta.has_audio}; {len(transcript.segments)} speech segs; {len(shots)} shots")

    if shots:
        log("        ┌─ shot ──────┬─ sharp ─┬─ expo ─┬─ sat ─┬─ motion ┬─ VISUAL")
        for s in shots:
            log(f"        │ {s['start']:5.1f}-{s['end']:4.1f}s │ {s['sharpness']:6.2f} │"
                f" {s['exposure']:.2f}  │ {s['saturation']:5.1f} │ {s['motion']:6.2f} │"
                f"  {s['visual_score']:.2f}")

    results: List[dict] = []
    from .render import render  # local import keeps ffmpeg dep lazy

    for fmt in formats:
        target = get_preset(fmt)
        clip_budget = budget if budget else min(target["max_duration"], 30.0)
        vw, sw = target["visual_weight"], target["speech_weight"]

        # 2. Selection Agent (blended visual + speech, weighted per format) -- #
        if shots:
            clips = agents.select_blended(shots, transcript, vw, sw, clip_budget)
        else:  # fallback: transcript-only (talking-head path)
            clips = agents.select_highlights(transcript, clip_budget, llm=llm)

        # 3. Story Agent ------------------------------------------------------ #
        timeline = agents.build_story(clips, hook_first=target["aspect"] == "9:16")
        # 4. Caption Agent ---------------------------------------------------- #
        captions = agents.build_captions(timeline, transcript)

        edl = EDL(source=meta, target=target, timeline=timeline, captions=captions,
                  analysis={"shots": shots, "visual_weight": vw, "speech_weight": sw})
        edl_path = os.path.join(outdir, f"{basename}_{target['name']}.edl.json")
        edl.save(edl_path)

        log(f"\n[+] {target['platform']:<16} {target['width']}x{target['height']}  "
            f"(visual×{vw} / speech×{sw}) | {len(timeline)} clips, {edl.total_duration():.1f}s")
        for c in timeline:
            log(f"        {c.id:<12} {c.src_in:5.1f}->{c.src_out:4.1f}s  {c.reason}")

        # 5 + 6. Editing + Platform export ----------------------------------- #
        out_path = render(edl, outdir, basename)
        results.append({"format": target["name"], "video": out_path, "edl": edl_path,
                        "duration": round(edl.total_duration(), 2),
                        "clips": [c.id for c in timeline]})
        log(f"        rendered -> {out_path}")

    return results
