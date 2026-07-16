#!/usr/bin/env python3
"""AI Clipper - skapar korta, vertikala, textade "viral clips" fran en YouTube-video.

Flode:
  1. Ladda ner videon med yt-dlp
  2. Transkribera lokalt med faster-whisper (ord-tidsstamplar)
  3. Valj segment med Claude (om ANTHROPIC_API_KEY finns) eller en lokal heuristik
  4. Klipp ut, beskär till 9:16 och bränn in TikTok-stil-undertexter med ffmpeg

Kraver: Python 3.9+, ffmpeg/ffprobe i PATH, paketen i requirements.txt.
"""

import argparse
import bisect
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"


# ---------------------------------------------------------------------------
# Hjälpfunktioner
# ---------------------------------------------------------------------------

def check_ffmpeg():
    """Avbryt med tydligt felmeddelande om ffmpeg/ffprobe saknas i PATH."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        print("FEL: {} hittades inte i PATH.".format(" och ".join(missing)))
        print("Installera ffmpeg (t.ex. 'winget install Gyan.FFmpeg' pa Windows,")
        print("'brew install ffmpeg' pa macOS eller 'sudo apt install ffmpeg' pa Linux)")
        print("och se till att ffmpeg och ffprobe ligger i PATH.")
        sys.exit(1)


def probe_video(path):
    """Returnerar (längd i sekunder, bredd, höjd) för en videofil via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe misslyckades: {proc.stderr.strip()[-400:]}")
    data = json.loads(proc.stdout)
    duration = float(data["format"]["duration"])
    width, height = 1920, 1080
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width") or width)
            height = int(stream.get("height") or height)
            break
    return duration, width, height


def sanitize_title(title, fallback="klipp"):
    """Gör en titel säker att använda som filnamn (ASCII, understreck)."""
    trans = str.maketrans({
        "å": "a", "ä": "a", "ö": "o", "Å": "a", "Ä": "a", "Ö": "o",
        "é": "e", "è": "e", "ü": "u", "ß": "ss",
    })
    text = title.lower().translate(trans)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return (text[:40].rstrip("_")) or fallback


# ---------------------------------------------------------------------------
# Steg 1: Nedladdning
# ---------------------------------------------------------------------------

def download_video(url, out_dir):
    """Laddar ner bästa mp4-video + m4a-ljud och slår ihop till source.mp4."""
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "source.mp4"
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        print(f"   Titel: {info.get('title', '(okand)')}")

    if target.exists():
        return target
    # Fallback om sammanslagningen fick en annan ändelse
    for candidate in out_dir.glob("source.*"):
        if candidate.suffix.lower() in (".mp4", ".mkv", ".webm"):
            return candidate
    raise RuntimeError("Kunde inte hitta den nedladdade videofilen.")


# ---------------------------------------------------------------------------
# Steg 2: Transkribering
# ---------------------------------------------------------------------------

def transcribe(path, model_size, device, compute_type):
    """Transkriberar med faster-whisper. Returnerar (segmentlista, ordlista)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_gen, info = model.transcribe(str(path), word_timestamps=True, vad_filter=True)
    print(f"   Sprak: {info.language} (sannolikhet {info.language_probability:.0%})")

    segments, words = [], []
    for seg in segments_gen:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({"start": seg.start, "end": seg.end, "text": text})
        for w in seg.words or []:
            token = w.word.strip()
            if token:
                words.append({"start": w.start, "end": w.end, "word": token})
        print(f"   Transkriberat t.o.m. {seg.end:6.1f} s ...", end="\r", flush=True)
    print()
    return segments, words


# ---------------------------------------------------------------------------
# Steg 3: Segmentval
# ---------------------------------------------------------------------------

def select_segments_ai(segments, duration, n_clips, min_len, max_len, model):
    """Ber Claude välja de bästa klippen. Kastar exception vid fel."""
    import anthropic

    lines = [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments]
    transcript = "\n".join(lines)
    if len(transcript) > 300_000:
        transcript = transcript[:300_000] + "\n[... transkriptionen avkortad ...]"

    prompt = f"""Du är expert på viralt kortformat-innehåll (TikTok, Instagram Reels, YouTube Shorts).

Nedan finns en tidsstämplad transkription av en video som är {duration:.0f} sekunder lång.
Varje rad har formatet [start-slut] text, med tider i sekunder.

Välj de {n_clips} bästa icke-överlappande segmenten som fungerar som fristående virala klipp. Leta efter:
- starka hooks som fångar tittaren under de första sekunderna
- tydliga poänger, punchlines eller överraskningar
- känslomässiga toppar
- konkreta, användbara tips

Krav:
- Varje segment ska vara mellan {min_len:.0f} och {max_len:.0f} sekunder långt.
- Segmenten får inte överlappa varandra.
- Börja och sluta vid naturliga meningsgränser.
- Utnyttja längdintervallet: föredra klipp i den övre delen av intervallet när innehållet bär, hellre än att klippa för kort.
- "start" och "end" anges i sekunder och måste ligga inom videons längd.

Svara med ENBART en ren JSON-array utan någon annan text, i exakt detta format:
[{{"start": 12.5, "end": 45.0, "title": "kort slagkraftig titel", "reason": "kort motivering"}}]

Transkription:
{transcript}"""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    lo, hi = text.find("["), text.rfind("]")
    if lo == -1 or hi <= lo:
        raise ValueError("hittade ingen JSON-array i modellens svar")
    result = json.loads(text[lo:hi + 1])
    if not isinstance(result, list):
        raise ValueError("modellens svar var inte en JSON-array")
    return result


def select_segments_heuristic(words, duration, n_clips, min_len, max_len):
    """Utan API: rangordna tidsfönster efter taltäthet (ord/sekund) och välj
    de bästa icke-överlappande fönstren utspridda över videon."""
    if not words:
        return []

    # Sikta pa ovre delen av langdintervallet - for korta klipp kanns snopna
    target = min(max_len, max(min_len, min_len + 0.75 * (max_len - min_len)))
    target = min(target, duration)
    starts = [w["start"] for w in words]

    candidates = []
    t, step = 0.0, 2.0
    last_start = max(0.0, duration - target)
    while t <= last_start + 1e-6:
        lo = bisect.bisect_left(starts, t)
        hi = bisect.bisect_right(starts, t + target)
        count = hi - lo
        if count > 0:
            candidates.append({"start": t, "end": t + target, "score": count / target})
        t += step
    if not candidates:
        return []

    def overlaps(cand, chosen_list):
        return any(cand["start"] < c["end"] and c["start"] < cand["end"] for c in chosen_list)

    # Välj bästa fönstret i varje region av videon => sprids ut över hela videon
    chosen = []
    region = duration / max(1, n_clips)
    for i in range(n_clips):
        r0, r1 = i * region, (i + 1) * region
        best = None
        for c in candidates:
            if r0 <= c["start"] < r1 and not overlaps(c, chosen):
                if best is None or c["score"] > best["score"]:
                    best = c
        if best is not None:
            chosen.append(best)
    # Fyll upp globalt om nagon region var tom
    for c in sorted(candidates, key=lambda c: -c["score"]):
        if len(chosen) >= n_clips:
            break
        if not overlaps(c, chosen):
            chosen.append(c)
    chosen.sort(key=lambda c: c["start"])

    # Snäpp fönstren till ordgränser, förläng mot mållängden och ge dem titlar
    result = []
    for i, c in enumerate(chosen):
        limit = chosen[i + 1]["start"] if i + 1 < len(chosen) else duration
        lo_i = bisect.bisect_left(starts, c["start"] - 0.01)
        hi_i = bisect.bisect_right(starts, c["end"]) - 1
        if hi_i < lo_i or lo_i >= len(words):
            continue
        start = max(0.0, words[lo_i]["start"] - 0.3)
        k = hi_i
        # Förläng klippet framåt ord för ord tills mållängden nås (utan att
        # passera nästa valda klipp eller max-längden)
        while (k + 1 < len(words)
               and words[k]["end"] + 0.4 - start < target
               and words[k + 1]["end"] + 0.4 - start <= max_len
               and words[k + 1]["start"] < limit - 0.01):
            k += 1
        end = min(duration, words[k]["end"] + 0.4, limit)
        if end - start > max_len:
            end = start + max_len
        title = " ".join(w["word"] for w in words[lo_i:lo_i + 4])
        result.append({
            "start": start,
            "end": end,
            "title": title or "klipp",
            "reason": f"hog taltathet ({c['score']:.1f} ord/s)",
        })
    return result


def normalize_segments(raw, duration, n_clips, min_len, max_len):
    """Klampa, längdjustera, sortera och ta bort överlapp. Max n_clips segment."""
    eff_min = min(min_len, duration)
    cleaned = []
    for i, s in enumerate(raw):
        try:
            start = float(s["start"])
            end = float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end <= start:
            continue
        if end - start > max_len:
            end = start + max_len
        if end - start < eff_min:
            end = min(duration, start + eff_min)
            if end - start < eff_min:
                start = max(0.0, end - eff_min)
        if end - start < eff_min - 0.5:
            continue
        title = str(s.get("title") or "").strip() or f"klipp {i + 1}"
        reason = str(s.get("reason") or "").strip()
        cleaned.append({"start": start, "end": end, "title": title, "reason": reason})

    cleaned.sort(key=lambda x: x["start"])
    result = []
    for s in cleaned:
        if result and s["start"] < result[-1]["end"]:
            s = dict(s, start=result[-1]["end"])
            if s["end"] - s["start"] < eff_min * 0.5:
                continue
        result.append(s)
    return result[:n_clips]


# ---------------------------------------------------------------------------
# Ansiktstracking för smart 9:16-crop
# ---------------------------------------------------------------------------

_YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
              "face_detection_yunet/face_detection_yunet_2023mar.onnx")
_DETECTOR_CACHE = {}


def _get_face_detector(cv2, det_w, det_h):
    """Skapar en ansiktsdetektor: YuNet (DNN, OpenCV 4.5.4+/5.x) i första hand,
    Haar-kaskad som fallback. Returnerar (typ, detektor) eller None."""
    if hasattr(cv2, "FaceDetectorYN_create"):
        model = (Path.home() / ".cache" / "ai_clipper"
                 / "face_detection_yunet_2023mar.onnx")
        if not model.exists():
            try:
                print("   Laddar ner ansiktsmodell (YuNet, ~230 kB) ...")
                import urllib.request
                data = urllib.request.urlopen(_YUNET_URL, timeout=30).read()
                if len(data) < 100_000:
                    raise ValueError("ofullstandig nedladdning")
                model.parent.mkdir(parents=True, exist_ok=True)
                model.write_bytes(data)
            except Exception as e:
                print(f"   (kunde inte ladda ner ansiktsmodellen: {e})")
        if model.exists():
            try:
                det = cv2.FaceDetectorYN_create(str(model), "", (det_w, det_h),
                                                score_threshold=0.6)
                return ("yunet", det)
            except cv2.error as e:
                print(f"   (kunde inte initiera YuNet: {e})")
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not cascade.empty():
            return ("haar", cascade)
    return None


def _detect_faces(kind, det, cv2, frame):
    """Detekterar ansikten i en bildruta. Returnerar lista av (x-centrum, area)."""
    if kind == "yunet":
        det.setInputSize((frame.shape[1], frame.shape[0]))
        _ret, faces = det.detect(frame)
        if faces is None:
            return []
        return [(float(f[0]) + float(f[2]) / 2.0, float(f[2]) * float(f[3]))
                for f in faces]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_face = max(24, int(frame.shape[0] * 0.08))
    rects = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                 minSize=(min_face, min_face))
    return [(x + w / 2.0, float(w) * float(h)) for (x, _y, w, h) in rects]


def analyze_faces(source, seg, src_w, src_h):
    """Samplar bildrutor i klippet och detekterar ansikten med OpenCV.
    Returnerar en lista av (tid relativt klippstart, ansiktets x-centrum i
    källpixlar), eller None om OpenCV saknas eller för få ansikten hittas.

    Vid flera ansikten följs det som ligger närmast föregående position
    (annars det största) - en stabil approximation av den som pratar."""
    try:
        import cv2
    except ImportError:
        print("   (opencv-python saknas - hoppar over ansiktstracking; "
              "installera med: pip install opencv-python)")
        return None

    shrink = min(1.0, 480.0 / src_h)     # detektera i nedskalad bild for fart
    det_w, det_h = int(round(src_w * shrink)), int(round(src_h * shrink))
    cache_key = (det_w, det_h)
    if cache_key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[cache_key] = _get_face_detector(cv2, det_w, det_h)
    detector = _DETECTOR_CACHE[cache_key]
    if detector is None:
        print("   (ingen ansiktsdetektor tillganglig - centrerad crop)")
        return None
    kind, det = detector

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return None

    duration = seg["end"] - seg["start"]
    sample_dt = 0.34                     # ~3 samplingar/sekund räcker för panorering
    # En enda seek till klippstarten, sedan sekventiell avkodning - att seeka
    # per sampel tvingar fram omkodning fran narmaste keyframe och ar mycket
    # langsammare.
    cap.set(cv2.CAP_PROP_POS_MSEC, seg["start"] * 1000.0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not fps or fps <= 1 or fps > 240:
        fps = 30.0
    frame_step = max(1, int(round(fps * sample_dt)))

    samples, prev_cx = [], None
    frame_idx = 0
    while frame_idx / fps < duration:
        if not cap.grab():
            break
        if frame_idx % frame_step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if shrink < 1.0:
                frame = cv2.resize(frame, (det_w, det_h))
            found = _detect_faces(kind, det, cv2, frame)
            cx = None
            if found:
                cands = [(c / shrink, a) for c, a in found]
                biggest = max(a for _c, a in cands)
                cands = [(c, a) for c, a in cands if a >= 0.4 * biggest]
                if prev_cx is not None:
                    cx = min(cands, key=lambda ca: abs(ca[0] - prev_cx))[0]
                else:
                    cx = max(cands, key=lambda ca: ca[1])[0]
                prev_cx = cx
            samples.append([frame_idx / fps, cx])
        frame_idx += 1
    cap.release()

    detected = sum(1 for _t, c in samples if c is not None)
    if not samples or detected < max(2, len(samples) * 0.15):
        return None  # för få träffar för att trackingen ska vara meningsfull

    # Fyll luckor: framåt-fyll, och fyll inledande luckor med första träffen
    last = None
    for s in samples:
        if s[1] is None:
            s[1] = last
        else:
            last = s[1]
    first = next(c for _t, c in samples if c is not None)
    for s in samples:
        if s[1] is None:
            s[1] = first
        else:
            break

    # Medianfilter (fönster 3) mot enstaka felaktiga detektioner
    smoothed = []
    for i in range(len(samples)):
        window = sorted(samples[j][1]
                        for j in range(max(0, i - 1), min(len(samples), i + 2)))
        smoothed.append((samples[i][0], window[len(window) // 2]))
    return smoothed


def _crop_positions(samples, crop_w, src_w):
    """Översätter ansiktspositioner till en lugn crop-bana: kameran ligger
    stilla så länge ansiktet är kvar i en dödzon, och glider mjukt
    (smoothstep, ~0.3 s) när ansiktet flyttar sig."""
    max_x = src_w - crop_w

    def clamp_x(cx):
        return min(float(max_x), max(0.0, cx - crop_w / 2.0))

    dead = crop_w * 0.12
    glide = 0.3        # kortare än samplingsintervallet => tider förblir stigande
    steps = 8
    cur = clamp_x(samples[0][1])
    positions = [(0.0, cur)]
    for t, cx in samples[1:]:
        want = clamp_x(cx)
        if abs(want - cur) <= dead:
            continue
        for k in range(1, steps + 1):
            u = k / steps
            ease = u * u * (3 - 2 * u)
            positions.append((t + glide * u, cur + (want - cur) * ease))
        cur = want
    return positions


def write_sendcmd(positions, path):
    """Skriver en kommandofil för ffmpegs sendcmd-filter som flyttar croppen."""
    lines = [f"{t:.3f} crop x {int(round(x))};" for t, x in positions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Steg 4: Klippning med undertexter
# ---------------------------------------------------------------------------

def _ass_time(t):
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _group_words(words, max_words=4, max_gap=0.8, max_dur=3.5):
    """Grupperar ord i korta 3-4-ordsbitar för TikTok-stil-captions."""
    chunks, current = [], []
    for w in words:
        if current and (
            len(current) >= max_words
            or w["start"] - current[-1]["end"] > max_gap
            or w["end"] - current[0]["start"] > max_dur
        ):
            chunks.append(current)
            current = []
        current.append(w)
    if current:
        chunks.append(current)
    return chunks


def build_ass(clip_words, clip_start, clip_dur, play_w, play_h):
    """Bygger en .ass-fil: stor fet vit text med svart kontur, versaler,
    grupperad i korta bitar synkade mot whisper-tidsstamplarna."""
    fontsize = max(24, int(play_h * 0.055))
    outline = max(2, int(fontsize * 0.09))
    margin_v = int(play_h * 0.30)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Viral,Arial Black,{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, Effect, Text
"""
    chunks = _group_words(clip_words)
    events = []
    for i, chunk in enumerate(chunks):
        start = max(0.0, chunk[0]["start"] - clip_start)
        end = chunk[-1]["end"] - clip_start + 0.15
        if i + 1 < len(chunks):
            end = min(end, chunks[i + 1][0]["start"] - clip_start - 0.02)
        end = min(end, clip_dur)
        if end - start < 0.05:
            continue
        text = " ".join(w["word"] for w in chunk).upper()
        text = text.replace("{", "(").replace("}", ")").replace("\n", " ")
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Viral,,0,0,0,,{text}"
        )
    return header + "\n".join(events) + "\n"


def cut_clip(source, seg, index, out_dir, vertical, captions, words, src_w, src_h,
             face_track=True):
    """Klipper ut ett segment med ffmpeg. Kör med cwd=out_dir så att
    subtitles/sendcmd-filtren kan använda relativa sökvägar (undviker problem
    med enhetsbokstav/kolon i ffmpeg-filtersyntax på Windows)."""
    base = f"{index:02d}_{sanitize_title(seg['title'])}"
    out_name = base + ".mp4"
    duration = seg["end"] - seg["start"]
    temp_files = []

    ass_name = None
    if captions:
        clip_words = [w for w in words
                      if seg["start"] - 0.05 <= w["start"] < seg["end"]]
        if clip_words:
            ass_name = base + ".ass"
            play_w, play_h = (1080, 1920) if vertical else (src_w, src_h)
            content = build_ass(clip_words, seg["start"], duration, play_w, play_h)
            (out_dir / ass_name).write_text(content, encoding="utf-8")
            temp_files.append(ass_name)

    filters = []
    if vertical:
        crop_w = int(src_h * 9 / 16) // 2 * 2
        if crop_w < src_w:
            # Bredare an 9:16: beskär i sidled - följ ansikten om möjligt
            x_pos = (src_w - crop_w) // 2
            if face_track:
                samples = analyze_faces(source, seg, src_w, src_h)
                if samples:
                    positions = _crop_positions(samples, crop_w, src_w)
                    cmd_name = base + ".cmd"
                    write_sendcmd(positions, out_dir / cmd_name)
                    temp_files.append(cmd_name)
                    filters.append(f"sendcmd=f={cmd_name}")
                    x_pos = int(round(positions[0][1]))
                else:
                    print(f"   (inga ansikten i klipp {index} - centrerad crop)")
            filters.append(f"crop=w={crop_w}:h={src_h}:x={x_pos}:y=0")
        elif src_w * 16 < src_h * 9:
            # Smalare an 9:16: beskär höjden istället (centrerat)
            crop_h = int(src_w * 16 / 9) // 2 * 2
            filters.append(f"crop=w={src_w}:h={crop_h}:x=0:y={(src_h - crop_h) // 2}")
        filters.append("scale=1080:1920")
    if ass_name:
        filters.append(f"subtitles={ass_name}")

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{seg['start']:.3f}",
        "-i", str(source.resolve()),
        "-t", f"{duration:.3f}",
    ]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_name,
    ]

    proc = subprocess.run(cmd, cwd=str(out_dir), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"   FEL: ffmpeg misslyckades for {out_name}:")
        print("   " + "\n   ".join(proc.stderr.strip().splitlines()[-6:]))
        return None
    for name in temp_files:
        try:
            (out_dir / name).unlink()
        except OSError:
            pass
    return out_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="ai_clipper",
        description="AI Clipper - genererar korta, vertikala, textade viral clips "
                    "fran en YouTube-video (gratis, lokal motsvarighet till "
                    "Klap.app/Opus Clip).",
    )
    parser.add_argument("url", help="YouTube-URL till videon")
    parser.add_argument("--clips", type=int, default=3,
                        help="antal klipp att generera (default: 3)")
    parser.add_argument("--min-len", type=float, default=20,
                        help="minsta klipplangd i sekunder (default: 20)")
    parser.add_argument("--max-len", type=float, default=60,
                        help="storsta klipplangd i sekunder (default: 60)")
    parser.add_argument("--out", default="clips",
                        help="outputmapp (default: clips)")
    parser.add_argument("--whisper-model", default="small",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="whisper-modellstorlek (default: small)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="enhet for whisper (default: cpu)")
    parser.add_argument("--compute-type", default="int8",
                        help="compute_type for faster-whisper, t.ex. int8, float16 "
                             "(default: int8)")
    parser.add_argument("--model", default=DEFAULT_CLAUDE_MODEL,
                        help=f"Claude-modell for segmentval (default: {DEFAULT_CLAUDE_MODEL})")
    parser.add_argument("--heuristic", action="store_true",
                        help="hoppa over AI och anvand alltid heuristiken")
    parser.add_argument("--no-vertical", action="store_true",
                        help="behall originalformatet istallet for 9:16")
    parser.add_argument("--no-captions", action="store_true",
                        help="branna inte in undertexter")
    parser.add_argument("--no-face-track", action="store_true",
                        help="stang av ansiktstracking och anvand fast "
                             "centrerad 9:16-crop")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    check_ffmpeg()

    if args.min_len <= 0 or args.max_len <= args.min_len:
        print("FEL: --max-len maste vara storre an --min-len (och bada positiva).")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== AI Clipper ==")

    # Steg 1: nedladdning
    print(f"[1/4] Laddar ner video fran {args.url} ...")
    try:
        source = download_video(args.url, out_dir)
    except Exception as e:
        print(f"FEL: nedladdningen misslyckades: {e}")
        sys.exit(1)
    duration, src_w, src_h = probe_video(source)
    print(f"   Sparad som {source} ({duration:.0f} s, {src_w}x{src_h})")

    # Steg 2: transkribering
    print(f"[2/4] Transkriberar med faster-whisper "
          f"({args.whisper_model}, {args.device}, {args.compute_type}) ...")
    segments, words = transcribe(source, args.whisper_model, args.device,
                                 args.compute_type)
    if not words:
        print("FEL: inget tal hittades i videon - kan inte valja klipp.")
        sys.exit(1)
    print(f"   {len(segments)} segment, {len(words)} ord.")

    # Steg 3: segmentval
    clips = None
    use_ai = not args.heuristic and os.environ.get("ANTHROPIC_API_KEY")
    if use_ai:
        print(f"[3/4] AI-analys: ber Claude ({args.model}) valja de basta klippen ...")
        try:
            raw = select_segments_ai(segments, duration, args.clips,
                                     args.min_len, args.max_len, args.model)
            clips = normalize_segments(raw, duration, args.clips,
                                       args.min_len, args.max_len)
            if not clips:
                raise ValueError("AI:n returnerade inga anvandbara segment")
        except Exception as e:
            print(f"   AI-analysen misslyckades ({type(e).__name__}: {e})")
            print("   Fortsatter med heuristiken istallet.")
            clips = None
    else:
        reason = "--heuristic angavs" if args.heuristic else "ANTHROPIC_API_KEY saknas"
        print(f"[3/4] Heuristiskt segmentval ({reason}) ...")

    if clips is None:
        raw = select_segments_heuristic(words, duration, args.clips,
                                        args.min_len, args.max_len)
        clips = normalize_segments(raw, duration, args.clips,
                                   args.min_len, args.max_len)

    if not clips:
        print("FEL: kunde inte hitta nagra lampliga segment.")
        sys.exit(1)

    print(f"   Valde {len(clips)} segment:")
    for i, seg in enumerate(clips, 1):
        extra = f" - {seg['reason']}" if seg.get("reason") else ""
        print(f"   {i}. {seg['start']:7.1f}-{seg['end']:7.1f} s  "
              f"\"{seg['title']}\"{extra}")

    # Steg 4: klippning
    if args.no_vertical:
        fmt = "originalformat"
    elif args.no_face_track:
        fmt = "vertikalt 9:16 (centrerad crop)"
    else:
        fmt = "vertikalt 9:16 med ansiktstracking"
    cap = "utan undertexter" if args.no_captions else "med inbrända undertexter"
    print(f"[4/4] Klipper {len(clips)} klipp, {fmt}, {cap} ...")
    created = []
    for i, seg in enumerate(clips, 1):
        name = cut_clip(source, seg, i, out_dir,
                        vertical=not args.no_vertical,
                        captions=not args.no_captions,
                        words=words, src_w=src_w, src_h=src_h,
                        face_track=not args.no_face_track)
        if name:
            print(f"   {name} klart ({seg['end'] - seg['start']:.0f} s)")
            created.append(name)

    if not created:
        print("FEL: inga klipp kunde skapas.")
        sys.exit(1)
    print(f"Klart! {len(created)} klipp sparade i {out_dir.resolve()}")


if __name__ == "__main__":
    main()
