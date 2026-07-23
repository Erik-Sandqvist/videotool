#!/usr/bin/env python3
"""Webbgranssnitt for AI Clipper.

Startar en lokal webbserver dar man klistrar in en YouTube-lank, foljer
korningen live och forhandsgranskar/laddar ner de fardiga klippen.

    python webapp.py            # oppna sedan http://127.0.0.1:8765

Pipelinen kors som subprocess (ai_clipper.py), sa serverns processer
paverkas inte av krascher i nedladdning/transkribering, och jobb kan
avbrytas. Ett jobb i taget - whisper ater garna hela CPU:n.
"""

import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
SCRIPT = BASE_DIR / "ai_clipper.py"
JOBS_DIR = BASE_DIR / "webjobs"

HOST, PORT = "127.0.0.1", 8765

app = Flask(__name__)
jobs = {}
jobs_lock = threading.Lock()

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")
DEVICES = ("cpu", "cuda")
OUT_FORMATS = ("9:16", "4:5", "1:1", "16:9", "original")
JOB_ID_RE = re.compile(r"^job-[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")


class Job:
    def __init__(self, job_id, params):
        self.id = job_id
        self.params = params
        self.status = "running"          # running | done | error | cancelled
        self.log = []
        self.proc = None
        self.out_dir = JOBS_DIR / job_id
        self.created = datetime.now().strftime("%Y-%m-%d %H:%M")

    def clips(self):
        if not self.out_dir.is_dir():
            return []
        return sorted(p.name for p in self.out_dir.glob("[0-9][0-9]_*.mp4"))

    def append_log(self, line):
        # yt-dlp spammar progress-rader - ersatt istallet for att stapla
        if (line.startswith("[download]") and self.log
                and self.log[-1].startswith("[download]")):
            self.log[-1] = line
            return
        self.log.append(line)
        if len(self.log) > 2000:
            del self.log[:1000]


def run_job(job):
    p = job.params
    cmd = [sys.executable, "-u", str(SCRIPT), p["url"],
           "--clips", str(p["clips"]),
           "--min-len", str(p["min_len"]),
           "--max-len", str(p["max_len"]),
           "--whisper-model", p["whisper_model"],
           "--device", p["device"],
           "--compute-type", p["compute_type"],
           "--model", p["model"],
           "--format", p["format"],
           "--out", str(job.out_dir)]
    for flag in ("heuristic", "no_captions", "no_face_track"):
        if p[flag]:
            cmd.append("--" + flag.replace("_", "-"))

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        job.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR), env=env, text=True,
            encoding="utf-8", errors="replace", bufsize=1)
        for raw in job.proc.stdout:
            for part in raw.replace("\r", "\n").split("\n"):
                part = part.rstrip()
                if part:
                    job.append_log(part)
        code = job.proc.wait()
    except Exception as e:
        job.append_log(f"FEL: kunde inte kora pipelinen: {e}")
        job.status = "error"
        return
    if job.status == "cancelled":
        return
    job.status = "done" if code == 0 and job.clips() else "error"


@app.get("/")
def index():
    return PAGE


@app.get("/api/info")
def api_info():
    import shutil as _shutil
    return jsonify({
        "api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "ffmpeg": _shutil.which("ffmpeg") is not None,
    })


@app.post("/api/start")
def api_start():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not url.lower().startswith(("http://", "https://")):
        return jsonify({"error": "Ange en giltig video-URL (http/https)."}), 400

    try:
        clips = max(1, min(20, int(data.get("clips", 3))))
        min_len = max(3.0, float(data.get("min_len", 20)))
        max_len = float(data.get("max_len", 60))
    except (TypeError, ValueError):
        return jsonify({"error": "Ogiltiga siffervarden."}), 400
    if max_len <= min_len:
        return jsonify({"error": "Max-langd maste vara storre an min-langd."}), 400

    whisper_model = str(data.get("whisper_model", "small"))
    device = str(data.get("device", "cpu"))
    out_format = str(data.get("format", "9:16"))
    if whisper_model not in WHISPER_MODELS or device not in DEVICES:
        return jsonify({"error": "Ogiltig whisper-modell eller device."}), 400
    if out_format not in OUT_FORMATS:
        return jsonify({"error": "Ogiltigt utformat."}), 400
    compute_type = str(data.get("compute_type", "int8"))
    model = str(data.get("model", "claude-sonnet-5"))
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", compute_type):
        return jsonify({"error": "Ogiltig compute-type."}), 400
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", model):
        return jsonify({"error": "Ogiltigt modellnamn."}), 400

    with jobs_lock:
        if any(j.status == "running" for j in jobs.values()):
            return jsonify({"error": "Ett jobb kor redan - vanta tills det ar "
                                     "klart eller avbryt det forst."}), 409
        job_id = "job-{}-{}".format(
            datetime.now().strftime("%Y%m%d-%H%M%S"), os.urandom(2).hex())
        job = Job(job_id, {
            "url": url, "clips": clips, "min_len": min_len, "max_len": max_len,
            "whisper_model": whisper_model, "device": device,
            "compute_type": compute_type, "model": model,
            "format": out_format,
            "heuristic": bool(data.get("heuristic")),
            "no_captions": bool(data.get("no_captions")),
            "no_face_track": bool(data.get("no_face_track")),
        })
        jobs[job_id] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return jsonify({"id": job_id})


@app.get("/api/status/<job_id>")
def api_status(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "okant jobb"}), 404
    since = request.args.get("since", 0, type=int)
    log = job.log
    return jsonify({
        "status": job.status,
        "log": log[max(0, since):],
        "log_total": len(log),
        "clips": job.clips(),
    })


@app.post("/api/cancel/<job_id>")
def api_cancel(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "okant jobb"}), 404
    if job.status != "running" or job.proc is None:
        return jsonify({"error": "jobbet kor inte"}), 400
    job.status = "cancelled"
    job.append_log("Avbrutet av anvandaren.")
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(job.proc.pid), "/T", "/F"],
                       capture_output=True)
    else:
        job.proc.terminate()
    return jsonify({"ok": True})


@app.get("/api/jobs")
def api_jobs():
    items = [{"id": j.id, "status": j.status, "created": j.created,
              "url": j.params["url"], "n_clips": len(j.clips())}
             for j in jobs.values()]
    items.sort(key=lambda x: x["id"], reverse=True)
    return jsonify(items)


@app.get("/clips/<job_id>/<path:filename>")
def serve_clip(job_id, filename):
    if not JOB_ID_RE.fullmatch(job_id):
        return jsonify({"error": "ogiltigt jobb-id"}), 400
    return send_from_directory(JOBS_DIR / job_id, filename)


PAGE = """<!doctype html>
<html lang="sv">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Clipper</title>
<style>
  :root {
    --bg: #0e1116; --panel: #171c24; --panel2: #1e2530; --border: #2a3342;
    --text: #e8edf4; --muted: #8b98ab; --accent: #7c5cff; --accent2: #00d4a6;
    --err: #ff6470;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.5 "Segoe UI", system-ui, sans-serif;
  }
  header {
    padding: 28px 32px 20px;
    background: linear-gradient(120deg, rgba(124,92,255,.18), rgba(0,212,166,.10) 60%, transparent);
    border-bottom: 1px solid var(--border);
  }
  header h1 { margin: 0; font-size: 26px; letter-spacing: .3px; }
  header h1 span { color: var(--accent); }
  header p { margin: 6px 0 0; color: var(--muted); }
  .badge {
    display: inline-block; margin-top: 10px; padding: 3px 12px;
    border-radius: 999px; font-size: 12.5px; border: 1px solid var(--border);
    background: var(--panel);
  }
  .badge.ok { color: var(--accent2); border-color: rgba(0,212,166,.4); }
  .badge.warn { color: #ffc46b; border-color: rgba(255,196,107,.4); }
  main {
    display: grid; grid-template-columns: 380px 1fr; gap: 22px;
    padding: 22px 32px 40px; max-width: 1400px; margin: 0 auto;
  }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px;
  }
  .card h2 { margin: 0 0 14px; font-size: 15px; text-transform: uppercase;
             letter-spacing: 1.2px; color: var(--muted); }
  label { display: block; font-size: 13px; color: var(--muted); margin: 12px 0 4px; }
  input[type=text], input[type=number], select {
    width: 100%; padding: 9px 11px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--panel2);
    color: var(--text); font-size: 14px; outline: none;
  }
  input:focus, select:focus { border-color: var(--accent); }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
  .checks { margin-top: 14px; display: grid; gap: 7px; }
  .checks label { display: flex; align-items: center; gap: 8px; margin: 0;
                  font-size: 13.5px; color: var(--text); cursor: pointer; }
  button {
    width: 100%; margin-top: 18px; padding: 12px; border: 0; border-radius: 10px;
    background: linear-gradient(90deg, var(--accent), #a06bff);
    color: #fff; font-size: 15px; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: .45; cursor: not-allowed; }
  button.cancel { background: transparent; border: 1px solid var(--err);
                  color: var(--err); margin-top: 10px; display: none; }
  #log {
    background: #0a0d12; border: 1px solid var(--border); border-radius: 10px;
    padding: 14px; height: 320px; overflow-y: auto; white-space: pre-wrap;
    font: 12.5px/1.55 Consolas, monospace; color: #b9c4d4;
  }
  #status-pill {
    float: right; font-size: 12.5px; padding: 3px 12px; border-radius: 999px;
    border: 1px solid var(--border); color: var(--muted);
  }
  #status-pill.running { color: var(--accent); border-color: rgba(124,92,255,.5); }
  #status-pill.done { color: var(--accent2); border-color: rgba(0,212,166,.5); }
  #status-pill.error, #status-pill.cancelled { color: var(--err);
      border-color: rgba(255,100,112,.5); }
  #clips { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
           gap: 16px; margin-top: 16px; }
  .clip { background: var(--panel2); border: 1px solid var(--border);
          border-radius: 12px; overflow: hidden; }
  .clip video { width: 100%; max-height: 420px; background: #000; display: block; }
  .clip .meta { padding: 10px 12px; }
  .clip .meta div { font-size: 13px; overflow: hidden; text-overflow: ellipsis;
                    white-space: nowrap; }
  .clip a { display: inline-block; margin-top: 6px; font-size: 12.5px;
            color: var(--accent2); text-decoration: none; }
  .empty { color: var(--muted); font-size: 13.5px; margin-top: 14px; }
</style>
</head>
<body>
<header>
  <h1>AI <span>Clipper</span></h1>
  <p>Fran YouTube-lank till fardiga vertikala klipp med undertexter och ansiktstracking.</p>
  <span class="badge" id="ai-badge">kontrollerar ...</span>
  <span class="badge" id="ffmpeg-badge" style="display:none">ffmpeg saknas!</span>
</header>
<main>
  <section class="card">
    <h2>Nytt jobb</h2>
    <form id="f" onsubmit="startJob(event)">
      <label>YouTube-URL</label>
      <input type="text" name="url" placeholder="https://www.youtube.com/watch?v=..." required>
      <div class="row3">
        <div><label>Antal klipp</label>
          <input type="number" name="clips" value="3" min="1" max="20"></div>
        <div><label>Min (s)</label>
          <input type="number" name="min_len" value="20" min="3"></div>
        <div><label>Max (s)</label>
          <input type="number" name="max_len" value="60" min="5"></div>
      </div>
      <div class="row3">
        <div><label>Format</label>
          <select name="format">
            <option selected>9:16</option><option>4:5</option>
            <option>1:1</option><option>16:9</option>
            <option value="original">original</option>
          </select></div>
        <div><label>Whisper-modell</label>
          <select name="whisper_model">
            <option>tiny</option><option>base</option>
            <option selected>small</option><option>medium</option>
            <option>large-v3</option>
          </select></div>
        <div><label>Device</label>
          <select name="device"><option selected>cpu</option><option>cuda</option></select></div>
      </div>
      <div class="row">
        <div><label>Compute type</label>
          <input type="text" name="compute_type" value="int8"></div>
        <div><label>Claude-modell</label>
          <input type="text" name="model" value="claude-sonnet-5"></div>
      </div>
      <div class="checks">
        <label><input type="checkbox" name="heuristic"> Hoppa over AI (heuristik)</label>
        <label><input type="checkbox" name="no_captions"> Inga undertexter</label>
        <label><input type="checkbox" name="no_face_track"> Ingen ansiktstracking</label>
      </div>
      <button id="start-btn" type="submit">Skapa klipp</button>
      <button id="cancel-btn" type="button" class="cancel" onclick="cancelJob()">Avbryt jobbet</button>
    </form>
  </section>
  <section>
    <div class="card">
      <h2>Korning <span id="status-pill">inget jobb</span></h2>
      <div id="log">Har dyker loggen upp nar du startar ett jobb.</div>
    </div>
    <div class="card" style="margin-top:22px">
      <h2>Klipp</h2>
      <div id="clips"></div>
      <div class="empty" id="clips-empty">Inga klipp an.</div>
    </div>
  </section>
</main>
<script>
let currentJob = null, logCount = 0, shownClips = [];

async function init() {
  try {
    const info = await (await fetch('/api/info')).json();
    const b = document.getElementById('ai-badge');
    if (info.api_key) { b.textContent = 'Segmentval: Claude (API-nyckel hittad)'; b.className = 'badge ok'; }
    else { b.textContent = 'Segmentval: heuristik (ingen ANTHROPIC_API_KEY)'; b.className = 'badge warn'; }
    if (!info.ffmpeg) document.getElementById('ffmpeg-badge').style.display = 'inline-block';
    const jobsList = await (await fetch('/api/jobs')).json();
    const latest = jobsList.find(j => j.status === 'running') || jobsList.find(j => j.n_clips > 0);
    if (latest) { currentJob = latest.id; logCount = 0; setRunning(latest.status === 'running'); poll(); }
  } catch (e) { /* servern svarar inte annu */ }
}

function setRunning(running) {
  document.getElementById('start-btn').disabled = running;
  document.getElementById('cancel-btn').style.display = running ? 'block' : 'none';
}

function setPill(status) {
  const pill = document.getElementById('status-pill');
  const names = { running: 'kor ...', done: 'klart', error: 'fel', cancelled: 'avbrutet' };
  pill.textContent = names[status] || status;
  pill.className = status;
}

async function startJob(ev) {
  ev.preventDefault();
  const f = document.getElementById('f');
  const body = {
    url: f.url.value.trim(), clips: +f.clips.value,
    min_len: +f.min_len.value, max_len: +f.max_len.value,
    whisper_model: f.whisper_model.value, device: f.device.value,
    compute_type: f.compute_type.value.trim(), model: f.model.value.trim(),
    format: f.format.value, heuristic: f.heuristic.checked,
    no_captions: f.no_captions.checked, no_face_track: f.no_face_track.checked,
  };
  const r = await fetch('/api/start', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await r.json();
  if (!r.ok) { alert(data.error || 'Kunde inte starta jobbet.'); return; }
  currentJob = data.id; logCount = 0; shownClips = [];
  document.getElementById('log').textContent = '';
  document.getElementById('clips').innerHTML = '';
  document.getElementById('clips-empty').style.display = 'block';
  setRunning(true); setPill('running');
  poll();
}

async function cancelJob() {
  if (!currentJob) return;
  await fetch('/api/cancel/' + currentJob, { method: 'POST' });
}

async function poll() {
  if (!currentJob) return;
  let s;
  try {
    s = await (await fetch(`/api/status/${currentJob}?since=${logCount}`)).json();
  } catch (e) { setTimeout(poll, 3000); return; }
  if (s.error) { setPill('error'); setRunning(false); return; }
  if (s.log && s.log.length) {
    const el = document.getElementById('log');
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 40;
    el.textContent += (el.textContent ? '\\n' : '') + s.log.join('\\n');
    if (atBottom) el.scrollTop = el.scrollHeight;
  }
  logCount = s.log_total;
  renderClips(s.clips || []);
  setPill(s.status);
  if (s.status === 'running') setTimeout(poll, 1500);
  else setRunning(false);
}

function renderClips(clips) {
  if (!clips.length) return;
  document.getElementById('clips-empty').style.display = 'none';
  const grid = document.getElementById('clips');
  for (const name of clips) {
    if (shownClips.includes(name)) continue;
    shownClips.push(name);
    const src = `/clips/${currentJob}/${encodeURIComponent(name)}`;
    const div = document.createElement('div');
    div.className = 'clip';
    div.innerHTML = `<video controls preload="metadata" src="${src}"></video>
      <div class="meta"><div title="${name}">${name}</div>
      <a href="${src}" download>Ladda ner</a></div>`;
    grid.appendChild(div);
  }
}

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    JOBS_DIR.mkdir(exist_ok=True)
    if not SCRIPT.exists():
        print(f"FEL: hittar inte {SCRIPT}")
        sys.exit(1)
    print(f"AI Clipper webbgranssnitt: http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
