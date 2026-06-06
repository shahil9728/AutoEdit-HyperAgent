#!/usr/bin/env python3
"""HTTP backend for autoedit — async job API so long renders don't time out.

  GET  /                      the upload UI (landing.html)        [also /app]
  GET  /health                health check (200 ok / 503)         [also /healthz]
  POST /process?formats=reel,short&budget=12
        multipart `files` (>=1). Returns 202 {job_id} immediately and renders
        in a background thread. All clips are combined into one edit.
  GET  /status/<job_id>       JSON: status, current stage, per-format state
  GET  /result/<job_id>?format=reel   the rendered MP4 for that format
  OPTIONS *                   CORS preflight

Why async: rendering can take minutes; holding one HTTP request open that long
trips Cloudflare/Render timeouts (the 520 you saw). Short poll requests don't.
"""

import cgi
import datetime
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from autoedit import __version__, run_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
START_TIME = time.time()
JOBS = {}
LOCK = threading.Lock()
JOB_TTL = 1800  # seconds


def log(msg):
    print(f"[autoedit {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# ffmpeg helpers (ultrafast presets — free-tier CPU is slow)
# --------------------------------------------------------------------------- #
def _run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-8:])
        raise RuntimeError(f"ffmpeg failed ({p.returncode}): {tail}")


def _has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "a",
                          "-show_entries", "stream=index", "-of", "csv=p=0", path],
                         stdout=subprocess.PIPE, text=True)
    return bool(out.stdout.strip())


def _normalize(src, out):
    vf = ("scale=1280:720:force_original_aspect_ratio=decrease,"
          "pad=1280:720:-1:-1:color=black,setsar=1,fps=30,format=yuv420p")
    if _has_audio(src):
        fc = f"[0:v]{vf}[v];[0:a]aresample=44100,aformat=channel_layouts=stereo[a]"
        _run(["ffmpeg", "-y", "-i", src, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
              "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
              "-c:a", "aac", "-ar", "44100", "-ac", "2", out])
    else:
        fc = f"[0:v]{vf}[v]"
        _run(["ffmpeg", "-y", "-i", src,
              "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
              "-filter_complex", fc, "-map", "[v]", "-map", "1:a", "-shortest",
              "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
              "-c:a", "aac", "-ar", "44100", "-ac", "2", out])


def _prepare_source(paths, work, stage):
    if len(paths) == 1:
        return paths[0]
    norm = []
    for i, p in enumerate(paths):
        stage(f"normalizing clip {i + 1}/{len(paths)}")
        o = os.path.join(work, f"norm{i}.mp4")
        _normalize(p, o)
        norm.append(o)
    stage("combining clips")
    combined = os.path.join(work, "combined.mp4")
    inputs = []
    for p in norm:
        inputs += ["-i", p]
    streams = "".join(f"[{i}:v][{i}:a]" for i in range(len(norm)))
    fc = f"{streams}concat=n={len(norm)}:v=1:a=1[v][a]"
    _run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
          "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-c:a", "aac", combined])
    return combined


def _worker(job_id, paths, fmts, budget, work):
    job = JOBS[job_id]

    def stage(s):
        job["stage"] = s
        log(f"[{job_id[:6]}] {s}")

    try:
        source = _prepare_source(paths, work, stage)
        for fmt in fmts:
            stage(f"rendering {fmt}")
            try:
                outdir = os.path.join(work, "out_" + fmt)
                res = run_pipeline(source, [fmt], transcript_path=None, outdir=outdir,
                                   budget=budget, use_visual=True, verbose=False)
                j = res[0]
                job["formats"][fmt].update(status="done", path=j["video"],
                                           clips=j["clips"], duration=j["duration"])
                log(f"[{job_id[:6]}] done {fmt} — {j['duration']}s, {len(j['clips'])} clips")
            except Exception as e:  # noqa
                job["formats"][fmt].update(status="error", error=str(e))
                log(f"[{job_id[:6]}] ERROR {fmt}: {e}")
                traceback.print_exc()
        job["status"] = "done"
        stage("done")
    except Exception as e:  # noqa
        job["status"] = "error"
        job["error"] = str(e)
        stage("error")
        traceback.print_exc()


def _sweep():
    now = time.time()
    with LOCK:
        for jid in [k for k, v in JOBS.items() if now - v["created"] > JOB_TTL]:
            shutil.rmtree(JOBS[jid].get("work", ""), ignore_errors=True)
            JOBS.pop(jid, None)


def health_payload():
    ffmpeg = shutil.which("ffmpeg")
    ok = ffmpeg is not None
    return ok, {"status": "ok" if ok else "degraded", "service": "autoedit",
                "version": __version__, "ffmpeg": bool(ffmpeg),
                "uptime_seconds": round(time.time() - START_TIME, 1),
                "active_jobs": len(JOBS),
                "time": datetime.datetime.now(datetime.timezone.utc).isoformat()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "X-Clips, X-Duration")

    def _send(self, code, ctype, body, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, "application/json", json.dumps(obj).encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _serve_app(self):
        try:
            with open(os.path.join(HERE, "landing.html"), "rb") as f:
                self._send(200, "text/html; charset=utf-8", f.read(),
                           {"Cache-Control": "no-store"})
        except FileNotFoundError:
            self._json(200, {"status": "ok", "note": "landing.html missing"})

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path in ("", "/app"):
            self._serve_app()
        elif path in ("/health", "/healthz"):
            ok, payload = health_payload()
            self._json(200 if ok else 503, payload)
        elif path.startswith("/status/"):
            self._status(path[len("/status/"):])
        elif path.startswith("/result/"):
            self._result(path[len("/result/"):], urllib.parse.parse_qs(parsed.query))
        else:
            self._json(404, {"error": "not found"})

    def _status(self, job_id):
        job = JOBS.get(job_id)
        if not job:
            self._json(404, {"error": "unknown job"})
            return
        self._json(200, {
            "status": job["status"], "stage": job["stage"],
            "elapsed": round(time.time() - job["created"], 1),
            "formats": {k: {"status": v["status"], "clips": v.get("clips"),
                            "duration": v.get("duration"), "error": v.get("error")}
                        for k, v in job["formats"].items()},
        })

    def _result(self, job_id, q):
        job = JOBS.get(job_id)
        if not job:
            self._json(404, {"error": "unknown job"})
            return
        fmt = (q.get("format") or [None])[0]
        entry = job["formats"].get(fmt) if fmt else None
        if not entry or entry.get("status") != "done":
            self._json(409, {"error": "not ready", "state": entry.get("status") if entry else None})
            return
        with open(entry["path"], "rb") as f:
            vid = f.read()
        self._send(200, "video/mp4", vid, {
            "X-Clips": ",".join(entry.get("clips") or []),
            "X-Duration": str(entry.get("duration", "")),
            "Content-Disposition": f'attachment; filename="{fmt}.mp4"',
        })

    def _read_uploads(self, work):
        ctype = self.headers.get("Content-Type", "")
        paths = []
        if "multipart/form-data" in ctype:
            fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                  environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype,
                                           "CONTENT_LENGTH": self.headers.get("Content-Length", "")})
            field = fs["files"] if "files" in fs else None
            items = field if isinstance(field, list) else ([field] if field else [])
            for i, it in enumerate(items):
                if not getattr(it, "filename", None):
                    continue
                p = os.path.join(work, f"input{i}.mp4")
                with open(p, "wb") as f:
                    if getattr(it, "file", None):
                        shutil.copyfileobj(it.file, f, 1 << 16)   # stream, low memory
                    else:
                        v = it.value
                        f.write(v if isinstance(v, (bytes, bytearray)) else str(v).encode("latin-1"))
                paths.append(p)
        else:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 0:
                p = os.path.join(work, "input0.mp4")
                with open(p, "wb") as f:
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 16, left))
                        if not chunk:
                            break
                        f.write(chunk)
                        left -= len(chunk)
                paths.append(p)
        return paths

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.rstrip("/") != "/process":
            self._json(404, {"error": "not found"})
            return
        q = urllib.parse.parse_qs(parsed.query)
        if "formats" in q:
            fmts = [s for s in q["formats"][0].split(",") if s]
        elif "format" in q:
            fmts = [q["format"][0]]
        else:
            fmts = ["reel"]
        budget = float((q.get("budget") or ["12"])[0])

        work = tempfile.mkdtemp(prefix="job_")
        try:
            paths = self._read_uploads(work)
        except Exception as e:  # noqa
            shutil.rmtree(work, ignore_errors=True)
            self._json(400, {"error": f"upload parse failed: {e}"})
            return
        if not paths:
            shutil.rmtree(work, ignore_errors=True)
            self._json(400, {"error": "no files uploaded"})
            return

        total_mb = sum(os.path.getsize(p) for p in paths) / 1e6
        _sweep()
        job_id = uuid.uuid4().hex
        with LOCK:
            JOBS[job_id] = {"status": "running", "stage": "queued", "created": time.time(),
                            "work": work, "formats": {f: {"status": "pending"} for f in fmts}}
        log(f"job {job_id[:6]} queued: {len(paths)} files ({total_mb:.1f} MB), formats={fmts}, budget={budget}")
        threading.Thread(target=_worker, args=(job_id, paths, fmts, budget, work), daemon=True).start()
        self._json(202, {"job_id": job_id, "formats": fmts, "files": len(paths)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    log(f"autoedit v{__version__} on :{port}  (app:/  health:/health  api:POST /process)")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
