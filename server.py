#!/usr/bin/env python3
"""HTTP backend for autoedit — serves the web app AND the render API.

  GET  /            the upload UI (landing.html)            [also /app]
  GET  /health      health check: 200 ok, 503 if ffmpeg missing   [also /healthz]
  POST /process?format=reel&budget=12
        multipart/form-data with one or more `files` (all combined into one edit)
        -> rendered MP4
  OPTIONS *         CORS preflight

Logs every step to stdout, so on Render you can watch progress in the Logs tab.
Python stdlib only (uses cgi for multipart — present on Python 3.11).
"""

import cgi
import datetime
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from autoedit import __version__, run_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
START_TIME = time.time()


def log(msg):
    print(f"[autoedit {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# ffmpeg helpers: normalize each clip to a common spec, then concatenate
# --------------------------------------------------------------------------- #
def _run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-8:])
        raise RuntimeError(f"ffmpeg failed ({p.returncode}): {tail}")


def _has_audio(path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, text=True)
    return bool(out.stdout.strip())


def _normalize(src, out):
    """Scale/pad to 1280x720@30, stereo 44.1k AAC (silent track if none)."""
    vf = ("scale=1280:720:force_original_aspect_ratio=decrease,"
          "pad=1280:720:-1:-1:color=black,setsar=1,fps=30,format=yuv420p")
    if _has_audio(src):
        fc = f"[0:v]{vf}[v];[0:a]aresample=44100,aformat=channel_layouts=stereo[a]"
        _run(["ffmpeg", "-y", "-i", src, "-filter_complex", fc,
              "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
              "-crf", "23", "-c:a", "aac", "-ar", "44100", "-ac", "2", out])
    else:
        _run(["ffmpeg", "-y", "-i", src,
              "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
              "-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]", "-map", "1:a",
              "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
              "-c:a", "aac", "-ar", "44100", "-ac", "2", out])


def _prepare_source(paths, work):
    if len(paths) == 1:
        log(f"1 clip — no concat needed")
        return paths[0]
    log(f"normalizing {len(paths)} clips to 1280x720@30…")
    norm = []
    for i, p in enumerate(paths):
        o = os.path.join(work, f"norm{i}.mp4")
        _normalize(p, o)
        norm.append(o)
        log(f"  normalized clip {i + 1}/{len(paths)}")
    combined = os.path.join(work, "combined.mp4")
    inputs = []
    for p in norm:
        inputs += ["-i", p]
    streams = "".join(f"[{i}:v][{i}:a]" for i in range(len(norm)))
    fc = f"{streams}concat=n={len(norm)}:v=1:a=1[v][a]"
    _run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", combined])
    log(f"combined {len(norm)} clips -> combined.mp4")
    return combined


def health_payload():
    ffmpeg = shutil.which("ffmpeg")
    ok = ffmpeg is not None
    return ok, {
        "status": "ok" if ok else "degraded", "service": "autoedit",
        "version": __version__, "ffmpeg": bool(ffmpeg),
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # we do our own logging
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "X-Clips, X-Duration, X-Sources")

    def _send(self, code, ctype, body, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

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
                self._send(200, "text/html; charset=utf-8", f.read())
        except FileNotFoundError:
            self._send(200, "application/json",
                       json.dumps({"status": "ok", "note": "landing.html missing"}).encode())

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path.rstrip("/")
        if path in ("", "/app"):
            self._serve_app()
        elif path in ("/health", "/healthz"):
            ok, payload = health_payload()
            self._send(200 if ok else 503, "application/json", json.dumps(payload).encode())
        else:
            self._send(404, "application/json", b'{"error":"not found"}')

    def _read_uploads(self, work):
        """Return list of saved input paths from the request (multipart or raw)."""
        ctype = self.headers.get("Content-Type", "")
        paths = []
        if "multipart/form-data" in ctype:
            fs = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype,
                         "CONTENT_LENGTH": self.headers.get("Content-Length", "")})
            field = fs["files"] if "files" in fs else None
            items = field if isinstance(field, list) else ([field] if field else [])
            for i, it in enumerate(items):
                if not getattr(it, "filename", None):
                    continue
                raw = it.file.read() if getattr(it, "file", None) else it.value
                p = os.path.join(work, f"input{i}.mp4")
                with open(p, "wb") as f:
                    f.write(raw if isinstance(raw, (bytes, bytearray)) else raw.encode("latin-1"))
                paths.append(p)
        else:  # raw single-file body (curl --data-binary)
            n = int(self.headers.get("Content-Length", "0"))
            if n > 0:
                p = os.path.join(work, "input0.mp4")
                with open(p, "wb") as f:
                    f.write(self.rfile.read(n))
                paths.append(p)
        return paths

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.rstrip("/") != "/process":
            self._send(404, "application/json", b'{"error":"not found"}')
            return
        q = urllib.parse.parse_qs(parsed.query)
        fmt = q.get("format", ["reel"])[0]
        budget = float(q.get("budget", ["12"])[0])
        work = tempfile.mkdtemp(prefix="job_")
        t0 = time.time()
        try:
            paths = self._read_uploads(work)
            total_mb = sum(os.path.getsize(p) for p in paths) / 1e6
            log(f"/process format={fmt} budget={budget} files={len(paths)} ({total_mb:.1f} MB)")
            if not paths:
                self._send(400, "application/json", b'{"error":"no files uploaded"}')
                return
            source = _prepare_source(paths, work)
            log(f"analyzing + rendering '{fmt}'…")
            res = run_pipeline(source, [fmt], transcript_path=None, outdir=work,
                               budget=budget, use_visual=True, verbose=False)
            job = res[0]
            with open(job["video"], "rb") as f:
                vid = f.read()
            log(f"done '{fmt}' in {time.time() - t0:.1f}s — {len(job['clips'])} clips, "
                f"{job['duration']}s, {len(vid)/1e6:.1f} MB out")
            self._send(200, "video/mp4", vid, {
                "X-Clips": ",".join(job["clips"]),
                "X-Duration": str(job["duration"]),
                "X-Sources": str(len(paths)),
                "Content-Disposition": f'attachment; filename="{fmt}.mp4"',
            })
        except Exception as e:  # noqa
            log(f"ERROR on '{fmt}': {e}")
            traceback.print_exc()
            self._send(500, "application/json", json.dumps({"error": str(e)}).encode())
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    log(f"autoedit v{__version__} on :{port}  (app: /  health: /health  api: POST /process)")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
