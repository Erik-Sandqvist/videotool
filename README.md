# AI Clipper

Ett CLI-verktyg i Python som tar en YouTube-länk och genererar korta, vertikala,
textade "viral clips" — en gratis, lokal motsvarighet till Klap.app/Opus Clip.

## Så funkar det

1. **Nedladdning** — videon hämtas med `yt-dlp` (bästa mp4-video + m4a-ljud, ihopslagen till en mp4).
2. **Transkribering** — ljudet transkriberas lokalt med `faster-whisper`, med tidsstämplar per ord.
3. **Segmentval** — om miljövariabeln `ANTHROPIC_API_KEY` finns skickas den tidsstämplade
   transkriptionen till Claude, som väljer ut de segment som passar bäst som fristående
   virala klipp (hooks, poänger, känslomässiga toppar, konkreta tips). Utan API-nyckel
   (eller om AI-anropet failar) används en gratis heuristik som rangordnar tidsfönster
   efter taltäthet och sprider klippen över videon.
4. **Klippning** — varje segment klipps ut med `ffmpeg`, beskärs till valt format
   (default vertikalt 9:16, 1080x1920 — även 4:5, 1:1, 16:9 och original stöds) och
   får ordvisa undertexter i TikTok/Shorts-stil inbrända (stor fet vit text med
   svart kontur, versaler, 3–4 ord i taget).
5. **Ansiktstracking** — vid 9:16-beskärningen detekteras ansikten med OpenCV och
   croppen centreras på det största/mest stabila ansiktet (en bra approximation av
   den som pratar). Kameran ligger stilla så länge ansiktet är kvar i en dödzon och
   panorerar mjukt när det flyttar sig. Hittas inga ansikten används centrerad crop.

## Installation

### 1. Krav

- **Python 3.9+**
- **ffmpeg och ffprobe i PATH** — verktyget vägrar starta utan dem.

Installera ffmpeg:

```powershell
# Windows (välj en):
winget install Gyan.FFmpeg
choco install ffmpeg
scoop install ffmpeg
```

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

Verifiera med `ffmpeg -version` i en ny terminal.

### 2. Python-paket

```powershell
pip install -r requirements.txt
```

Första körningen laddar faster-whisper automatiskt ner vald modell (t.ex. `small`, ~500 MB).

### 3. (Valfritt) API-nyckel för AI-segmentval

Utan nyckel funkar allt ändå — verktyget faller då tillbaka på heuristiken.
Med en Anthropic-nyckel väljer Claude ut de bästa klippen.

I PowerShell, för aktuell session:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Permanent (gäller nya terminalfönster):

```powershell
setx ANTHROPIC_API_KEY "sk-ant-..."
```

I bash/zsh:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Webbgränssnitt

```powershell
python webapp.py
```

Öppna sedan **http://127.0.0.1:8765** i webbläsaren. Där kan du:

- klistra in en YouTube-länk och ställa in alla alternativ i ett formulär
- följa nedladdning/transkribering/klippning i en live-logg
- förhandsgranska klippen direkt i sidan och ladda ner dem
- avbryta ett pågående jobb

Sidan visar också om `ANTHROPIC_API_KEY` och ffmpeg hittas i serverns miljö
(sätt nyckeln i terminalen *innan* du startar `webapp.py`). Ett jobb körs i
taget, och varje jobb får en egen mapp under `webjobs/`.

## Användning (CLI)

```
python ai_clipper.py <url> [--clips N] [--min-len S] [--max-len S] [--out MAPP]
                      [--whisper-model tiny|base|small|medium|large-v3]
                      [--device cpu|cuda] [--compute-type TYP]
                      [--model CLAUDE_MODELL] [--heuristic]
                      [--format 9:16|4:5|1:1|16:9|original]
                      [--no-captions] [--no-face-track]
```

### Exempel

```powershell
# Standard: 3 klipp à 20-60 sekunder till mappen clips/
python ai_clipper.py "https://www.youtube.com/watch?v=VIDEO_ID"

# 5 kortare klipp (15-30 s) till en egen mapp
python ai_clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" --clips 5 --min-len 15 --max-len 30 --out mina_klipp

# Snabbare transkribering med GPU och större modell
python ai_clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" --whisper-model medium --device cuda --compute-type float16

# Kvadratiska klipp för Instagram-flödet
python ai_clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" --format 1:1

# Hoppa över AI:n (ingen API-kostnad) och behåll originalformatet utan text
python ai_clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" --heuristic --format original --no-captions
```

### Flaggor

| Flagga | Default | Beskrivning |
|---|---|---|
| `--clips N` | `3` | Antal klipp att generera |
| `--min-len S` | `20` | Minsta klipplängd i sekunder |
| `--max-len S` | `60` | Största klipplängd i sekunder |
| `--out MAPP` | `clips` | Outputmapp (skapas vid behov) |
| `--whisper-model` | `small` | Whisper-modell: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--device` | `cpu` | `cpu` eller `cuda` |
| `--compute-type` | `int8` | faster-whisper compute type, t.ex. `int8`, `float16` |
| `--model` | `claude-sonnet-5` | Claude-modell för segmentvalet |
| `--format` | `9:16` | Utformat: `9:16`, `4:5`, `1:1`, `16:9` eller `original` |
| `--heuristic` | av | Använd alltid heuristiken, även om API-nyckel finns |
| `--no-vertical` | av | (deprecerad) samma som `--format original` |
| `--no-captions` | av | Bränn inte in undertexter |
| `--no-face-track` | av | Stäng av ansiktstracking (fast centrerad crop) |

## Output

I outputmappen hamnar:

- `source.mp4` — den nedladdade källvideon (återanvänds vid nästa körning av samma video)
- `01_titel.mp4`, `02_titel.mp4`, ... — de färdiga klippen, namngivna efter
  klippets titel (från Claude eller de första orden i klippet)

## Tips

- `--whisper-model small` är en bra balans på CPU. Har du NVIDIA-GPU: kör
  `--device cuda --compute-type float16` med `medium` eller `large-v3` för bäst kvalitet.
- Heuristiken är helt gratis och funkar förvånansvärt bra på pratiga videor
  (poddar, föreläsningar), men Claude är klart bättre på att hitta hooks och poänger.
- Undertexterna genereras som en `.ass`-fil per klipp och bränns in med ffmpegs
  `subtitles`-filter; filen tas bort automatiskt när klippet är klart.
- Ansiktstrackingen styr croppen via en `.cmd`-fil till ffmpegs `sendcmd`-filter
  (tas också bort automatiskt). Saknas `opencv-python` hoppar verktyget över
  trackingen med en varning och kör centrerad crop istället.
- Utan API-nyckel siktar heuristiken på den övre delen av längdintervallet
  (~50 s med default 20–60) — vill du ha kortare klipp, sänk `--max-len`.
