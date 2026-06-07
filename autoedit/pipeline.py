"""The orchestrator — wires the agents together around one EDL per target.

Flow: analyze the footage ONCE (technical probe + transcript + visual metrics),
then for each requested platform build a tailored EDL (blended selection with
that platform's visual/speech weighting) and render it.

`progress(stage)` (optional) is called as work advances so a caller (the web
backend) can surface live sub-stages. `shots_hint` lets the caller pass known
shot boundaries (e.g. clip cut points) to skip the costly scene-detect pass.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

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
    progress: Optional[Callable[[str], None]] = None,
    shots_hint: Optional[List[Tuple[float, float]]] = None,
) -> List[dict]:
    os.makedirs(outdir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(source))[0]

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    def note(stage: str) -> None:
        if progress:
            progress(stage)

    # 1. Analysis Agent --------------------------------------------------- #
    log("[1/6] Analysis     : probe + transcribe + visual metrics")
    meta = agents.probe_source(source)
    transcript = agents.transcribe(source, transcript_path, asr)

    shots = []
    if use_visual:
        note("analyzing footage (scoring shots)")
        shots = visual.analyze(source, meta.duration, shots=shots_hint)
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
        # Budget = how many seconds of footage Selection keeps.
        #   * explicit budget (user picked a target length) -> honour it.
        #   * auto (budget falsy) + known clip boundaries (a multi-clip upload)
        #     -> keep EVERY uploaded clip, capped only at the platform max. The
        #     user chose those clips on purpose, so trimming to a tiny default
        #     and silently dropping one is the wrong call.
        #   * auto + single video -> a sensible platform-length default.
        max_dur = float(target["max_duration"])
        if budget and budget > 0:
            clip_budget = float(budget)
        elif shots_hint:
            clip_budget = min(meta.duration + 0.5, max_dur)
        else:
            clip_budget = min(max_dur, 60.0)
        vw, sw = target["visual_weight"], target["speech_weight"]

        # 2. Selection Agent (blended visual + speech, weighted per format) -- #
        if shots:
            clips = agents.select_blended(shots, transcript, vw, sw, clip_budget)
        else:  # fallback: transcript-only (talking-head path)
            clips = agents.select_highlights(transcript, clip_budget, llm=llm)

        # 3. Story Agent ------------------------------------------------------ #
        timeline = agents.build_story(clips, hook_first=target["aspect"] == "9:16")
        # 3b. Effects Agent — pick a per-clip effect from its analysis --------- #
        timeline = agents.assign_effects(timeline, target)
        # 4. Caption Agent ---------------------------------------------------- #
        captions = agents.build_captions(timeline, transcript)

        edl = EDL(source=meta, target=target, timeline=timeline, captions=captions,
                  analysis={"shots": shots, "visual_weight": vw, "speech_weight": sw})
        edl_path = os.path.join(outdir, f"{basename}_{target['name']}.edl.json")
        edl.save(edl_path)

        log(f"\n[+] {target['platform']:<16} {target['width']}x{target['height']}  "
            f"(visual×{vw} / speech×{sw}) | {len(timeline)} clips, {edl.total_duration():.1f}s")
        for c in timeline:
            log(f"        {c.id:<12} {c.src_in:5.1f}->{c.src_out:4.1f}s  "
                f"motion={c.motion:.2f} fx={c.effect:<9} {c.reason}")

        # 5 + 6. Editing + Platform export ----------------------------------- #
        note(f"rendering {target['name']}")
        out_path = render(edl, outdir, basename)
        results.append({"format": target["name"], "video": out_path, "edl": edl_path,
                        "duration": round(edl.total_duration(), 2),
                        "clips": [c.id for c in timeline]})
        log(f"        rendered -> {out_path}")

    return results
