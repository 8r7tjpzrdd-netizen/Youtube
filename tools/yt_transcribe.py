#!/usr/bin/env python3
"""Достаёт текст видео с YouTube: сначала пробует субтитры (ручные, потом авто),
при их отсутствии — распознаёт звук локально через faster-whisper.

Использование:
    python3 tools/yt_transcribe.py <URL или ID> [-l ru] [-o transcripts]

На выходе в каталоге <out>/<video_id>/:
    transcript_raw.txt    — сплошной текст без таймкодов
    transcript_timed.txt  — тот же текст блоками по 30 секунд с таймкодами
    subs.<lang>.json3     — исходные субтитры (если брались из YouTube)
"""
import argparse
import json
import os
import re
import subprocess
import sys

# Клиенты YouTube перебираются по очереди: web чаще всего отдаёт
# "Sign in to confirm you're not a bot", ios и web_embedded обычно проходят.
CLIENTS = ["ios", "web_embedded", "android_vr", "tv", "mweb", "web"]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def video_id(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url.strip()


def fetch_subs(url, lang, outdir):
    """Пробует скачать субтитры на языке lang. Возвращает путь к json3 или None."""
    base = os.path.join(outdir, "subs")
    for client in CLIENTS:
        for flag in ("--write-subs", "--write-auto-subs"):
            run([
                sys.executable, "-m", "yt_dlp",
                "--extractor-args", f"youtube:player_client={client}",
                "--ignore-no-formats-error", "--skip-download",
                flag, "--sub-langs", f"{lang}-orig,{lang}",
                "--sub-format", "json3", "-o", base + ".%(ext)s", url,
            ])
            for suffix in (f"{lang}-orig", lang):
                path = f"{base}.{suffix}.json3"
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    print(f"субтитры: client={client} {flag} lang={suffix}")
                    return path
    return None


def parse_json3(path):
    """[(начало в мс, текст)] из формата json3."""
    data = json.load(open(path, encoding="utf-8"))
    lines = []
    for ev in data.get("events", []):
        if "segs" not in ev:
            continue
        text = "".join(s.get("utf8", "") for s in ev["segs"]).replace("\n", " ").strip()
        if text:
            lines.append((ev.get("tStartMs", 0), text))
    return lines


def whisper_fallback(url, lang, outdir):
    """Скачивает звук и распознаёт его локально. Требует ffmpeg и faster-whisper."""
    audio = os.path.join(outdir, "audio.m4a")
    for client in CLIENTS:
        r = run([
            sys.executable, "-m", "yt_dlp",
            "--extractor-args", f"youtube:player_client={client}",
            "-f", "bestaudio", "-o", audio, url,
        ])
        if os.path.exists(audio):
            break
    else:
        raise SystemExit("не удалось скачать аудио: " + r.stderr[-800:])

    from faster_whisper import WhisperModel  # ставится отдельно, см. README

    model = WhisperModel(os.environ.get("WHISPER_MODEL", "medium"), compute_type="int8")
    segments, _ = model.transcribe(audio, language=lang, vad_filter=True)
    return [(int(s.start * 1000), s.text.strip()) for s in segments]


def write_outputs(lines, outdir):
    raw = re.sub(r"\s+", " ", " ".join(t for _, t in lines)).strip()
    with open(os.path.join(outdir, "transcript_raw.txt"), "w", encoding="utf-8") as f:
        f.write(raw + "\n")

    def hms(ms):
        s = ms // 1000
        return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

    blocks, buf, cur = [], [], None
    for ms, text in lines:
        bucket = ms // 30000
        if cur is None:
            cur = bucket
        if bucket != cur:
            blocks.append(f"[{hms(cur * 30000)}] " + " ".join(buf))
            buf, cur = [], bucket
        buf.append(text)
    if buf:
        blocks.append(f"[{hms(cur * 30000)}] " + " ".join(buf))
    with open(os.path.join(outdir, "transcript_timed.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")

    print(f"сегментов: {len(lines)}, символов: {len(raw)}, "
          f"длительность: {lines[-1][0] // 60000} мин")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("-l", "--lang", default="ru")
    ap.add_argument("-o", "--out", default="transcripts")
    ap.add_argument("--whisper", action="store_true", help="сразу распознавать звук, не трогая субтитры")
    args = ap.parse_args()

    vid = video_id(args.url)
    url = args.url if args.url.startswith("http") else f"https://youtu.be/{vid}"
    outdir = os.path.join(args.out, vid)
    os.makedirs(outdir, exist_ok=True)

    lines = None
    if not args.whisper:
        subs = fetch_subs(url, args.lang, outdir)
        if subs:
            lines = parse_json3(subs)
    if not lines:
        print("субтитров нет — распознаю звук")
        lines = whisper_fallback(url, args.lang, outdir)
    write_outputs(lines, outdir)


if __name__ == "__main__":
    main()
