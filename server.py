#!/usr/bin/env python3
"""Minimal HTTP backend for autoedit — the upload -> render -> download loop.

  GET  /                          health check (JSON)
  POST /process?format=reel&budget=12   body = raw video bytes
                                  -> runs the pipeline, returns the rendered MP4

Python stdlib only (no Flask needed). This is essentially the service you'd
deploy behind the landing page. In the agent sandbox it's driven by curl on
localhost, because the sandbox has no public inbound URL to expose it on.
"""

import json
import os
import tempfile
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from autoedit import run_pipeline


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, ctype, body, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/"):
            self._send(200, "application/json",
                       json.dumps({"status": "ok", "service": "autoedit",
                                   "endpoint": "POST /process?format=reel"}).encode())
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
    print(f"autoedit backend listening on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
