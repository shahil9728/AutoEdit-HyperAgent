#!/usr/bin/env python3
"""HTTP backend for autoedit — serves the web app AND the render API.

  GET  /            the upload UI (landing.html)            [also /app]
  GET  /health      health check: 200 ok, 503 if ffmpeg missing   [also /healthz]
  POST /process?format=reel&budget=12   body = raw video bytes -> rendered MP4
  OPTIONS *         CORS preflight

Serving the page from the same origin as the API means the browser talks to
/process with no cross-origin friction. CORS headers are still sent so the page
also works when hosted elsewhere. Python stdlib only.
"""

import datetime
import json
import os
import shutil
import tempfile
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from autoedit import __version__, run_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
START_TIME = time.time()


def health_payload():
    """Liveness + confirmation the critical dependency (ffmpeg) is present."""
    ffmpeg = shutil.which("ffmpeg")
    ok = ffmpeg is not None
    return ok, {
        "status": "ok" if ok else "degraded",
        "service": "autoedit",
        "version": __version__,
        "ffmpeg": bool(ffmpeg),
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
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
                       json.dumps({"status": "ok",
                                   "note": "landing.html missing; API at POST /process"}).encode())

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path.rstrip("/")
        if path in ("", "/app"):
            self._serve_app()
        elif path in ("/health", "/healthz"):
            ok, payload = health_payload()
            self._send(200 if ok else 503, "application/json",
                       json.dumps(payload).encode())
        else:
            self._send(404, "application/json", b'{"error":"not found"}')

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.rstrip("/") != "/process":
            self._send(404, "application/json", b'{"error":"not found"}')
            return
        q = urllib.parse.parse_qs(parsed.query)
        fmt = q.get("format", ["reel"])[0]
        budget = float(q.get("budget", ["12"])[0])
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            self._send(400, "application/json", b'{"error":"empty body"}')
            return

        data = self.rfile.read(n)
        work = tempfile.mkdtemp(prefix="job_")
        inp = os.path.join(work, "input.mp4")
        with open(inp, "wb") as f:
            f.write(data)
        try:
            res = run_pipeline(inp, [fmt], transcript_path=None, outdir=work,
                               budget=budget, use_visual=True, verbose=False)
        except Exception as e:  # noqa
            self._send(500, "application/json", json.dumps({"error": str(e)}).encode())
            return

        job = res[0]
        with open(job["video"], "rb") as f:
            vid = f.read()
        self._send(200, "video/mp4", vid, {
            "X-Clips": ",".join(job["clips"]),
            "X-Duration": str(job["duration"]),
            "Content-Disposition": f'attachment; filename="{fmt}.mp4"',
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"autoedit on :{port}  (app: /  health: /health  api: POST /process)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
