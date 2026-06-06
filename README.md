# autoedit — autonomous video editor (MVP)

A small, **runnable** slice of the "AI video production team" vision: drop in
raw footage, get back finished, captioned, platform-specific videos —
automatically. It now judges the **picture** (sharpness, exposure, colour,
motion), not just the speech, so it works for cinematic / b-roll footage as well
as talking-head content.

---

## The six agents, in miniature

| Product vision agent | This MVP | Status |
|---|---|---|
| **Analysis** | `probe_source` (ffprobe) + `transcribe` (Whisper/fixture) + **`visual.analyze`** (shot detection + sharpness/exposure/saturation/motion) | metadata + visual real; ASR pluggable |
| **Clip Selection** | `select_blended` — rank shots by a **visual+speech blend, weighted per format**; drop weak/dark/blurry | real; LLM-pluggable |
| **Storytelling** | `build_story` — hook-first then chronological | rule-based |
| **Editing & Style** | `render.py` pass 1 — cut + aspect reframe + concat | **real ffmpeg** |
| **Music / Captions / FX** | `generate_ass` word-pop captions + music hook | captions real; music/VO planned |
| **Platform Optimization** | `presets.py` + per-format render loop | real (multi-version output) |

Everything talks through one shared artifact — the **Edit Decision List (EDL)**
(`edl.py`). Agents never call each other; they read and mutate the EDL.

---

## Visual analysis — judging the picture (the new part)

`visual.py` runs entirely on ffmpeg (no numpy / OpenCV), so it works anywhere:

| Metric | How |
|---|---|
| **shots** | scene-cut detection (`select=gt(scene,T)`) → shot ranges (long takes subdivided) |
| **sharpness / focus** | Laplacian via `convolution` kernel, then mean energy (`signalstats`) — blur scores ~0 |
| **exposure** | mean luma (`YAVG`); penalise dark and blown-out |
| **saturation** | `SATAVG` — colour punch |
| **motion** | inter-frame difference (`tblend=difference`) energy |

These are normalised across the clip into a 0..1 `visual_score`. The Selection
agent blends it with the speech score using each format's weighting
(`presets.py`):

```
cinematic 0.85 / 0.15      travel 0.80 / 0.20      (visual-dominant)
short/reel/tiktok 0.45 / 0.55   square 0.50 / 0.50  (balanced)
vlog 0.20 / 0.80                                    (speech-dominant)
```

**Why it matters:** from the *same* footage, cinematic mode keeps the sharp,
well-exposed shots and drops the blurry/dark ones; vlog mode keeps the shots
where the hook is *spoken*, even if they're visually plain. Use `--no-visual`
to force the old transcript-only behaviour.

---

## Quickstart (no credentials, no GPU)

```bash
ffmpeg -version                 # the only hard dependency for the demo

python3 make_sample.py          # 6-shot source (varying quality) + transcript

# same footage, three intents — watch the selection differ
python3 cli.py fixtures/source.mp4 \
    --formats cinematic vlog reel \
    --budget 12 --transcript fixtures/sample_transcript.json --outdir out
```

The console prints the per-shot visual metrics table, then which shots each
format kept and why. Outputs (+ inspectable `*.edl.json`) land in `out/`.

---

## How the render spine works

**Pass 1 — cut, reframe, concat** (one ffmpeg call): per shot, `trim` → `crop`
to the target aspect anchored at `focus_x` → `scale` → `concat`.
**Pass 2 — burn captions** (one ffmpeg call): a libass `.ass` file with a
TikTok-style active-word highlight, burned with the `subtitles` filter.

Transitions today are **hard cuts** (`concat`). Upgrading to ffmpeg's built-in
`xfade`/`acrossfade` (slow dissolves for cinematic, quick slides for energetic)
is a queued change. Reframe is a static centre/anchor crop; subject-tracking
(MediaPipe) to drive `focus_x` is queued too.

---

## Wiring the real ML steps

1. **Whisper** — `adapters/whisper_asr.py` already returns the fixture schema;
   omit `--transcript` to use it.
2. **LLM selection** — pass `llm=` to swap the heuristic for real editorial judgment.
3. **Aesthetic scoring** — add a NIMA/CLIP-aesthetic or VLM score per shot inside
   `visual.analyze` and fold it into `visual_score` (the cleanest next upgrade).
4. **Subject-tracking reframe** — set `focus_x` per shot from face/saliency.

## Backend & one-click deploy

`server.py` is a dependency-free HTTP backend:

```
GET  /                                  health check
POST /process?format=reel&budget=12     body = raw video bytes -> rendered MP4
```

Run it locally with `make serve`, or containerise it exactly like production:

```bash
make docker-build && make docker-run     # -> http://localhost:8000
```

Deploy with **auto-redeploy on push** using the included `Dockerfile` +
`render.yaml` (Render) or `fly.toml` (Fly.io). Full steps in **DEPLOY.md**.

## Layout

```
autoedit/
  edl.py          shared Edit Decision List (now carries visual/speech scores + analysis)
  visual.py       NEW: ffmpeg-only shot detection + visual quality scoring
  agents.py       Analysis / Selection (blended) / Story / Caption
  render.py       Editing + Platform agents (ffmpeg: cut, reframe, caption burn)
  presets.py      per-platform target specs + visual/speech weights
  pipeline.py     orchestrator ("the single click")
cli.py            command-line entrypoint (--formats, --budget, --no-visual)
server.py         HTTP backend (upload -> render -> download)
make_sample.py    synthetic 6-shot source + transcript generator
adapters/         optional ML adapters (real Whisper, etc.)
Dockerfile        container image (python + ffmpeg)
render.yaml       Render blueprint (auto-deploy on push)
fly.toml          Fly.io app config
Makefile          common tasks: sample, demo, serve, docker-build/run
DEPLOY.md         step-by-step deploy guide
fixtures/  out/   generated demo inputs / outputs
```
