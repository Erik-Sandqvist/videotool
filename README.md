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
4. **Klippning** — varje segment klipps ut med `ffmpeg`, beskärs till vertikalt
   9:16-format (1080x1920) och får ordvisa undertexter i TikTok/Shorts-stil inbrända
   (stor fet vit text med svart kontur, versaler, 3–4 ord i taget).

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

## Användning

```
python ai_clipper.py <url> [--clips N] [--min-len S] [--max-len S] [--out MAPP]
                      [--whisper-model tiny|base|small|medium|large-v3]
                      [--device cpu|cuda] [--compute-type TYP]
                      [--model CLAUDE_MODELL] [--heuristic]
                      [--no-vertical] [--no-captions]
```

### Exempel

```powershell
# Standard: 3 klipp à 20-60 sekunder till mappen clips/
python ai_clipper.py "https://www.youtube.com/watch?v=VIDEO_ID"

# 5 kortare klipp (15-30 s) till en egen mapp
python ai_clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" --clips 5 --min-len 15 --max-len 30 --out mina_klipp

# Snabbare transkribering med GPU och större modell
python ai_clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" --whisper-model medium --device cuda --compute-type float16

# Hoppa över AI:n (ingen API-kostnad) och behåll originalformatet utan text
python ai_clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" --heuristic --no-vertical --no-captions
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
| `--heuristic` | av | Använd alltid heuristiken, även om API-nyckel finns |
| `--no-vertical` | av | Behåll originalformatet istället för 9:16-crop |
| `--no-captions` | av | Bränn inte in undertexter |

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
