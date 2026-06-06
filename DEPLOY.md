# Deploying the autoedit backend

The goal: push code to GitHub once, and from then on **every fix you push
auto-redeploys** — no hand-editing the running server.

```
  you push to GitHub  ─►  host rebuilds the Docker image  ─►  live in ~1–2 min
```

The backend is `server.py` (stdlib + ffmpeg):

```
GET  /                                  liveness + info (JSON)
GET  /health  (or /healthz)             health check: 200 ok, 503 if ffmpeg missing
POST /process?format=reel&budget=12     body = raw video bytes -> rendered MP4
```

---

## 0. One-time: put it on GitHub

```bash
git init && git add -A && git commit -m "autoedit backend"
git branch -M main
git remote add origin https://github.com/<you>/autoedit.git
git push -u origin main
```

(If you downloaded the repo zip, the `git init` + first commit is already done —
just add your remote and push.)

---

## 1A. Deploy on Render (uses `render.yaml`, simplest)

1. Render dashboard → **New + → Blueprint**.
2. Connect your GitHub repo. Render reads `render.yaml`, builds the `Dockerfile`,
   and deploys a free web service.
3. `autoDeploy: true` is already set → **every push to `main` redeploys.**

Your backend lives at `https://autoedit-backend.onrender.com`.
Test it:

```bash
curl https://autoedit-backend.onrender.com/
curl -X POST --data-binary @clip.mp4 \
  "https://autoedit-backend.onrender.com/process?format=reel&budget=12" -o reel.mp4
```

> Free Render web services sleep after idle and have limited RAM/CPU — fine for
> demos and short clips. Bump the plan for longer footage.

## 1B. Deploy on Fly.io (uses `fly.toml`)

```bash
# install flyctl: https://fly.io/docs/flyctl/install/
fly auth login
fly launch --no-deploy        # detects Dockerfile + fly.toml (keep the app name or change it)
fly deploy                    # first deploy
```

**Auto-deploy on push** — add `.github/workflows/fly.yml`:

```yaml
name: fly deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: superfly/flyctl-actions/setup-flyctl@master
      - run: flyctl deploy --remote-only
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

Then `fly tokens create deploy` and save it as the `FLY_API_TOKEN` GitHub secret.
Now every push redeploys.

---

## 2. Run / build locally first (optional sanity check)

```bash
make serve                       # python3 server.py  -> http://localhost:8000
# or in Docker, exactly like production:
make docker-build && make docker-run
```

---

## 3. Connecting the landing page

`landing.html`'s Execute button currently runs a front-end simulation. To make
it call your live backend, point it at your URL and `fetch()` the `/process`
endpoint:

```js
const BACKEND_URL = "https://autoedit-backend.onrender.com";
const res = await fetch(`${BACKEND_URL}/process?format=reel&budget=12`, {
  method: "POST", body: fileInput.files[0]   // raw video bytes
});
const blob = await res.blob();               // the rendered MP4
videoEl.src = URL.createObjectURL(blob);
```

Host `landing.html` on the same service (or any static host) and the demo is
fully live. (Ask me to wire this up — it's a small change to `landing.html`.)

---

## The dev loop you wanted

1. Something breaks → copy the error / `fly logs` / Render logs.
2. Paste it to me here → I commit the fix to the repo.
3. The host auto-redeploys. You never hand-edit the server.

## Notes

- **Speech-to-text** is off by default (visual-only). To enable it, uncomment the
  faster-whisper line in the `Dockerfile` (and expect a larger image / more RAM).
- **Resources**: ffmpeg rendering is CPU/RAM-bound. Short clips run on free tiers;
  for long footage, raise the plan or move rendering to a GPU worker + queue.
