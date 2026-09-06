#!/usr/bin/env python3
"""Транскрибация видео с YouTube или локального аудиофайла.

Два источника текста:
  --mode subs    субтитры YouTube (быстро, но это чужое ASR среднего качества)
  --mode whisper распознавание звука моделью Whisper (медленно, точнее)

Примеры:
    python3 tools/yt_transcribe.py https://youtu.be/ID --mode subs -l ru
    python3 tools/yt_transcribe.py https://youtu.be/ID --mode whisper --cookies cookies.txt
    python3 tools/yt_transcribe.py lecture.m4a --mode whisper -l ru
    python3 tools/yt_transcribe.py lecture.m4a --mode whisper --speed 0.75 --from 21:00 --to 26:00

На выходе в <out>/<имя>/: transcript_raw.txt, transcript_timed.txt, transcript.srt.
"""
import argparse
import json
import os
import re
import subprocess
import sys

# Клиенты плеера перебираются по очереди: web почти всегда упирается в
# бот-проверку, ios обычно отдаёт хотя бы субтитры.
CLIENTS = ["ios", "web_embedded", "android_vr", "tv", "mweb", "web"]
NODE = "/opt/node22/bin/node"  # yt-dlp нужен Node >= 22 для JS-челленджей
AUDIO_EXT = {".m4a", ".mp3", ".wav", ".opus", ".ogg", ".webm", ".mp4", ".mkv", ".aac", ".flac"}


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def ytdlp(args, cookies=None):
    cmd = [sys.executable, "-m", "yt_dlp"]
    if os.path.exists(NODE):
        cmd += ["--js-runtimes", f"node:{NODE}"]
    if cookies:
        cmd += ["--cookies", cookies]
    return run(cmd + args)


def video_id(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url.strip()


def parse_ts(value):
    """'21:30' или '1:02:15' или '95' -> секунды."""
    if value is None:
        return None
    parts = [float(p) for p in str(value).split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


# --- источник 1: субтитры -------------------------------------------------

def fetch_subs(url, lang, outdir, cookies=None):
    base = os.path.join(outdir, "subs")
    for client in CLIENTS:
        for flag in ("--write-subs", "--write-auto-subs"):
            ytdlp([
                "--extractor-args", f"youtube:player_client={client}",
                "--ignore-no-formats-error", "--skip-download",
                flag, "--sub-langs", f"{lang}-orig,{lang}",
                "--sub-format", "json3", "-o", base + ".%(ext)s", url,
            ], cookies)
            for suffix in (f"{lang}-orig", lang):
                path = f"{base}.{suffix}.json3"
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    print(f"субтитры: client={client} {flag} lang={suffix}")
                    return path
    return None


def parse_json3(path):
    data = json.load(open(path, encoding="utf-8"))
    lines = []
    for ev in data.get("events", []):
        if "segs" not in ev:
            continue
        text = "".join(s.get("utf8", "") for s in ev["segs"]).replace("\n", " ").strip()
        if text:
            start = ev.get("tStartMs", 0) / 1000
            lines.append((start, start + ev.get("dDurationMs", 0) / 1000, text))
    return lines


# --- источник 2: Whisper --------------------------------------------------

def download_audio(url, outdir, cookies=None):
    """Скачивает лучшую аудиодорожку. Без cookies YouTube часто отдаёт
    'Sign in to confirm you're not a bot' — тогда нужен --cookies."""
    target = os.path.join(outdir, "audio.m4a")
    if os.path.exists(target):
        return target
    last = ""
    for client in CLIENTS:
        r = ytdlp([
            "--extractor-args", f"youtube:player_client={client}",
            "-f", "bestaudio/best", "-o", target, url,
        ], cookies)
        if os.path.exists(target):
            print(f"аудио скачано через client={client}")
            return target
        last = (r.stderr or r.stdout)[-600:]
    raise SystemExit("не удалось скачать аудио. Последняя ошибка:\n" + last +
                     "\nПередайте cookies: --cookies cookies.txt")


def prepare_audio(src, outdir, speed=1.0, start=None, end=None):
    """Приводит звук к 16 кГц моно; при необходимости режет и замедляет."""
    dst = os.path.join(outdir, "audio16k.wav")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-i", src]
    filters = ["highpass=f=60", "loudnorm=I=-18:TP=-2:LRA=11"]
    if speed != 1.0:
        # atempo надёжно работает в диапазоне 0.5-2.0, вне его — каскадом
        tempo, chain = speed, []
        while tempo < 0.5:
            chain.append("atempo=0.5")
            tempo /= 0.5
        chain.append(f"atempo={tempo:.4f}")
        filters += chain
    cmd += ["-af", ",".join(filters), "-ar", "16000", "-ac", "1", dst]
    r = run(cmd)
    if not os.path.exists(dst):
        raise SystemExit("ffmpeg не смог подготовить звук:\n" + r.stderr[-800:])
    return dst


def transcribe(audio, lang, model_name, speed=1.0, offset=0.0, prompt=None):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         cpu_threads=os.cpu_count() or 4)
    segments, info = model.transcribe(
        audio,
        language=None if lang in ("auto", "") else lang,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,  # иначе модель зацикливается на распевах
        initial_prompt=prompt,
    )
    print(f"язык: {info.language} (p={info.language_probability:.2f})")
    lines = []
    for s in segments:
        # если звук замедляли, возвращаем таймкоды к исходному темпу
        a = offset + s.start * speed
        b = offset + s.end * speed
        text = s.text.strip()
        if text:
            lines.append((a, b, text))
            print(f"[{int(a)//60:02d}:{int(a)%60:02d}] {text[:90]}", flush=True)
    return lines


# --- вывод ----------------------------------------------------------------

def hms(sec, sep=":", ms=False):
    s = int(sec)
    base = f"{s // 3600:02d}{sep}{s % 3600 // 60:02d}{sep}{s % 60:02d}"
    return base + (f",{int((sec - s) * 1000):03d}" if ms else "")


def write_outputs(lines, outdir):
    raw = re.sub(r"\s+", " ", " ".join(t for _, _, t in lines)).strip()
    open(os.path.join(outdir, "transcript_raw.txt"), "w", encoding="utf-8").write(raw + "\n")

    blocks, buf, cur = [], [], None
    for start, _, text in lines:
        bucket = int(start) // 30
        if cur is None:
            cur = bucket
        if bucket != cur:
            blocks.append(f"[{hms(cur * 30)}] " + " ".join(buf))
            buf, cur = [], bucket
        buf.append(text)
    if buf:
        blocks.append(f"[{hms(cur * 30)}] " + " ".join(buf))
    open(os.path.join(outdir, "transcript_timed.txt"), "w", encoding="utf-8").write(
        "\n\n".join(blocks) + "\n")

    with open(os.path.join(outdir, "transcript.srt"), "w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(lines, 1):
            f.write(f"{i}\n{hms(a, ms=True)} --> {hms(b, ms=True)}\n{text}\n\n")

    print(f"\nсегментов: {len(lines)}, символов: {len(raw)}, "
          f"конец: {hms(lines[-1][1]) if lines else '-'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="URL YouTube, ID видео или путь к аудиофайлу")
    ap.add_argument("--mode", choices=["subs", "whisper"], default="subs")
    ap.add_argument("-l", "--lang", default="ru", help="код языка или auto")
    ap.add_argument("-o", "--out", default="transcripts")
    ap.add_argument("-m", "--model", default="deepdml/faster-whisper-large-v3-turbo-ct2")
    ap.add_argument("--cookies", help="cookies.txt для обхода бот-проверки YouTube")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="замедление звука перед распознаванием, напр. 0.75")
    ap.add_argument("--from", dest="start", help="начало фрагмента, напр. 21:00")
    ap.add_argument("--to", dest="end", help="конец фрагмента, напр. 26:00")
    ap.add_argument("--prompt", help="подсказка модели: имена, термины, стиль")
    args = ap.parse_args()

    is_file = os.path.exists(args.source) and os.path.splitext(args.source)[1].lower() in AUDIO_EXT
    name = os.path.splitext(os.path.basename(args.source))[0] if is_file else video_id(args.source)
    url = None if is_file else (args.source if args.source.startswith("http")
                                else f"https://youtu.be/{name}")
    outdir = os.path.join(args.out, name)
    os.makedirs(outdir, exist_ok=True)

    if args.mode == "subs":
        if is_file:
            raise SystemExit("режим subs работает только с URL")
        path = fetch_subs(url, args.lang, outdir, args.cookies)
        if not path:
            raise SystemExit("субтитров нет — используйте --mode whisper")
        write_outputs(parse_json3(path), outdir)
        return

    src = args.source if is_file else download_audio(url, outdir, args.cookies)
    start = parse_ts(args.start)
    audio = prepare_audio(src, outdir, args.speed, start, parse_ts(args.end))
    lines = transcribe(audio, args.lang, args.model, args.speed, start or 0.0, args.prompt)
    write_outputs(lines, outdir)


if __name__ == "__main__":
    main()
