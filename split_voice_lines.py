#!/usr/bin/env python3
"""
Скачивает аудио с YouTube, режет на фразы (сегменты Whisper),
сохраняет в папку с понятными именами (локальная LLM через Ollama — опционально).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WhisperProfile:
    key: str
    model: str
    compute_cuda: str
    compute_cpu: str
    beam_size: int
    best_of: int
    patience: float
    label: str


# Доп. секунды к концу таймкода фразы (до экспорта), чтобы не резать слово по Whisper.
PHRASE_END_PAD_SEC = 0.15

# Подсказка Whisper (initial_prompt): слова/стиль, которые модель чаще угадывает верно.
# Пустая строка = без подсказки. Переопределяется флагом --whisper-prompt.
WHISPER_INITIAL_PROMPT = ""

WHISPER_PROFILES: dict[str, WhisperProfile] = {
    "quality": WhisperProfile(
        key="quality",
        model="large-v3",
        compute_cuda="float16",
        compute_cpu="int8",
        beam_size=5,
        best_of=5,
        patience=1.0,
        label="large-v3 + float16 + beam 5 (быстрее, хорошая точность)",
    ),
    "max": WhisperProfile(
        key="max",
        model="large-v3",
        compute_cuda="float32",
        compute_cpu="int8_float16",
        beam_size=10,
        best_of=5,
        patience=1.5,
        label="large-v3 + float32 + beam 10 (макс. точность, сильная нагрузка на GPU)",
    ),
}


def setup_cuda_dlls() -> None:
    """Пути к nvidia-cublas/cudnn DLL (до import faster_whisper)."""
    if sys.platform != "win32":
        return
    extra_path: list[str] = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        spec = importlib.util.find_spec(pkg)
        if not spec or not spec.submodule_search_locations:
            continue
        bin_dir = Path(spec.submodule_search_locations[0]) / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            extra_path.append(str(bin_dir))
    if extra_path:
        os.environ["PATH"] = ";".join(extra_path) + ";" + os.environ.get("PATH", "")


setup_cuda_dlls()

import imageio_ffmpeg
import requests
import yt_dlp
from faster_whisper import WhisperModel
from tqdm import tqdm

_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
import pydub

pydub.AudioSegment.converter = _FFMPEG_EXE
from pydub import AudioSegment


def configure_ffmpeg() -> str:
    """Встроенный ffmpeg — не нужно ставить отдельно на Windows."""
    AudioSegment.converter = _FFMPEG_EXE
    return _FFMPEG_EXE


def detect_device(requested: str) -> tuple[str, str]:
    """auto → cuda если доступен, иначе cpu."""
    if requested != "auto":
        compute = "float16" if requested == "cuda" else "int8"
        return requested, compute

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def resolve_whisper_setup(
    device: str,
    profile_key: str,
    model_override: str | None,
) -> tuple[WhisperProfile, str, str]:
    if profile_key not in WHISPER_PROFILES:
        profile_key = "quality"
    profile = WHISPER_PROFILES[profile_key]
    model = model_override or profile.model
    if device == "cpu" and model.startswith("large"):
        model = "medium"
    compute = profile.compute_cuda if device == "cuda" else profile.compute_cpu
    return profile, model, compute


class DownloadProgress:
    def __init__(self) -> None:
        self.bar: tqdm | None = None

    def hook(self, data: dict) -> None:
        if data["status"] != "downloading":
            return
        total = data.get("total_bytes") or data.get("total_bytes_estimate")
        downloaded = data.get("downloaded_bytes", 0)
        if total and self.bar is None:
            self.bar = tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Скачивание",
                leave=True,
            )
        if self.bar and total:
            self.bar.n = downloaded
            self.bar.refresh()
        elif self.bar is None:
            speed = data.get("speed")
            spd = f" {speed / 1024 / 1024:.1f} MB/s" if speed else ""
            tqdm.write(f"  загружено {downloaded / 1024 / 1024:.1f} MB{spd}")

    def close(self) -> None:
        if self.bar:
            self.bar.close()


def get_node_executable() -> str:
    import shutil

    node = shutil.which("node")
    if node:
        return node
    cursor = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs/cursor/resources/app/resources/helpers/node.exe"
    )
    if cursor.exists():
        return str(cursor)
    raise RuntimeError(
        "Node.js не найден (нужен для YouTube). Установите: https://nodejs.org"
    )


def cookies_help_message() -> str:
    return (
        "Нужен файл cookies.txt в корне проекта.\n"
        "1. Расширение Chrome: Get cookies.txt LOCALLY\n"
        "   https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc\n"
        "2. Откройте видео на youtube.com (подтвердите возраст)\n"
        "3. Export -> сохраните как cookies.txt рядом со скриптом"
    )


def save_last_job(url: str, work: Path, title: str) -> None:
    Path(".last-job.json").write_text(
        json.dumps(
            {
                "url": url,
                "id": extract_youtube_id(url),
                "title": title,
                "output": str(work),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def download_audio(url: str, out_dir: Path, ffmpeg: str) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = DownloadProgress()

    cookies_file = Path("cookies.txt")
    if not cookies_file.exists():
        raise RuntimeError(cookies_help_message())

    node = get_node_executable()
    tqdm.write("Cookies: cookies.txt")
    tqdm.write(f"Node (YouTube EJS): {node}")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "ffmpeg_location": ffmpeg,
        "cookiefile": str(cookies_file),
        "js_runtimes": {"node": node},
        "remote_components": ["ejs:github"],
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [progress.hook],
    }

    try:
        tqdm.write("Скачивание аудио…")
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        title = info.get("title", "source")
        vid = info.get("id", "")
        dur = info.get("duration")
        if dur:
            tqdm.write(f"  id={vid}, длительность={dur / 60:.1f} мин")
    except Exception as e:
        raise RuntimeError(
            f"Скачивание не удалось: {e}\n\n{cookies_help_message()}"
        ) from e
    finally:
        progress.close()

    wav = out_dir / "source.wav"
    if not wav.exists():
        candidates = list(out_dir.glob("source.*"))
        wav = next((p for p in candidates if p.suffix.lower() == ".wav"), candidates[0])

    try:
        from pydub.utils import mediainfo

        info = mediainfo(str(wav))
        length_ms = int(float(info.get("duration", 0)) * 1000)
        if length_ms:
            tqdm.write(f"  WAV: {length_ms / 60000:.1f} мин ({wav.stat().st_size / 1024 / 1024:.0f} MB)")
    except Exception:
        pass

    return wav, title


def slugify(text: str, max_len: int = 60) -> str:
    """Текст реплики -> slug для имени файла (транслит, латиница)."""
    from unidecode import unidecode

    text = unidecode(text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "_", text).strip("_")
    if not text:
        text = "phrase"
    return text[:max_len].rstrip("_")


def ollama_slug(phrase: str, model: str, host: str) -> str | None:
    prompt = f"""Ты помогаешь именовать короткие голосовые реплики (русский).
Дана фраза:
«{phrase}»

Верни ТОЛЬКО одно короткое имя файла (2-6 слов, snake_case, латиница или транслит).
Без кавычек, без пояснений. Примеры: hold_the_line, need_help, call_back_later, order_ready."""

    try:
        r = requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
        raw = raw.split("\n")[0].strip("`\"' ")
        return slugify(raw, max_len=50) or None
    except Exception:
        return None


def transcribe(
    wav: Path,
    model_size: str,
    language: str,
    device: str,
    compute_type: str,
    profile: WhisperProfile,
    *,
    initial_prompt: str | None = None,
) -> list[dict]:
    tqdm.write(
        f"Загрузка Whisper «{model_size}» на {device} ({compute_type}), "
        f"профиль «{profile.key}»: beam={profile.beam_size}…"
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    transcribe_kw: dict = dict(
        language=language,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=400,
            speech_pad_ms=320 if profile.key == "max" else 260,
        ),
        word_timestamps=True,
        beam_size=profile.beam_size,
        best_of=profile.best_of,
        patience=profile.patience,
        condition_on_previous_text=False,
        repetition_penalty=1.08 if profile.key == "max" else 1.0,
        temperature=0.0 if profile.key == "max" else [0.0, 0.2, 0.4],
    )
    if initial_prompt:
        transcribe_kw["initial_prompt"] = initial_prompt
        tqdm.write(f"Whisper initial_prompt: {initial_prompt[:80]}{'…' if len(initial_prompt) > 80 else ''}")

    segments_iter, info = model.transcribe(str(wav), **transcribe_kw)

    duration = info.duration or 0.0
    bar = tqdm(
        total=duration,
        unit="s",
        desc="Транскрипция",
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]",
    )

    out: list[dict] = []
    last_end = 0.0
    for seg in segments_iter:
        t = (seg.text or "").strip()
        if len(t) >= 2:
            words = []
            if seg.words:
                for w in seg.words:
                    wt = (w.word or "").strip()
                    if wt:
                        words.append({"word": wt, "start": w.start, "end": w.end})
            out.append({"start": seg.start, "end": seg.end, "text": t, "words": words})
        if seg.end > last_end:
            bar.update(seg.end - last_end)
            last_end = seg.end
    bar.n = duration
    bar.refresh()
    bar.close()
    return clean_whisper_segments(out)


def simplify_stutter_text(text: str) -> str:
    """«Наш ранен! Наш ранен! …» от галлюцинаций Whisper → одна фраза."""
    parts = split_text_to_phrases(text.strip(), min_len=2)
    if len(parts) <= 1:
        return text.strip()
    norm = [_norm_token(p) for p in parts]
    if norm and len(set(norm)) == 1:
        return parts[0]
    return text.strip()


def _segment_text_key(text: str) -> str:
    return simplify_stutter_text(text).casefold()


def collapse_consecutive_duplicate_segments(
    segments: list[dict],
    *,
    max_run_keep: int = 1,
) -> list[dict]:
    """Схлопывает подряд идущие одинаковые сегменты (петли Whisper)."""
    if not segments:
        return []
    out: list[dict] = []
    i = 0
    while i < len(segments):
        text = segments[i]["text"].strip()
        j = i + 1
        while j < len(segments) and segments[j]["text"].strip() == text:
            j += 1
        run = segments[i:j]
        if len(run) <= 1:
            out.append(run[0])
        elif len(run) == 2:
            out.extend(run)
        else:
            keep = sorted(
                run, key=lambda s: s["end"] - s["start"], reverse=True
            )[:max_run_keep]
            keep.sort(key=lambda s: s["start"])
            out.extend(keep)
        i = j
    return out


def dedupe_text_bursts(
    segments: list[dict],
    *,
    burst_gap_sec: float = 5.0,
) -> list[dict]:
    """
    Один сегмент на «вспышку» одинакового текста (галлюцинации Whisper).
    Повторы той же реплики через десятки секунд в ролике сохраняются.
    """
    ordered = sorted(segments, key=lambda s: s["start"])
    kept: list[dict] = []
    last_end_by_key: dict[str, float] = {}

    for seg in ordered:
        key = _segment_text_key(seg["text"])
        start = seg["start"]
        if key in last_end_by_key and start < last_end_by_key[key] + burst_gap_sec:
            if kept and _segment_text_key(kept[-1]["text"]) == key:
                prev = kept[-1]
                if (seg["end"] - seg["start"]) > (prev["end"] - prev["start"]):
                    kept[-1] = seg.copy()
                    last_end_by_key[key] = seg["end"]
            continue
        item = seg.copy()
        kept.append(item)
        last_end_by_key[key] = seg["end"]

    return kept


def clean_whisper_segments(
    segments: list[dict],
    *,
    burst_gap_sec: float = 5.0,
) -> list[dict]:
    cleaned: list[dict] = []
    for seg in segments:
        text = simplify_stutter_text(seg.get("text", ""))
        if len(text) < 2:
            continue
        if seg["end"] - seg["start"] < 0.12:
            continue
        item = seg.copy()
        item["text"] = text
        item.pop("words", None)
        cleaned.append(item)
    cleaned = collapse_consecutive_duplicate_segments(cleaned)
    return dedupe_text_bursts(cleaned, burst_gap_sec=burst_gap_sec)


def prepare_whisper_segments(
    segments: list[dict],
    *,
    merge_gap: float,
    burst_gap: float,
) -> list[dict]:
    """Нормализация сегментов Whisper перед нарезкой."""
    before = len(segments)
    out = clean_whisper_segments(segments, burst_gap_sec=burst_gap)
    out = merge_tiny_gaps(out, merge_gap)
    if before != len(out):
        tqdm.write(f"Очистка сегментов Whisper: {before} -> {len(out)}")
    return out


def _norm_token(tok: str) -> str:
    return re.sub(r"[^\wёЁа-яА-Я0-9]+", "", tok.lower())


def split_text_to_phrases(text: str, min_len: int = 4) -> list[str]:
    """Делит по ! и ? — типичные границы игровых реплик."""
    parts = re.split(r"(?<=[!?])\s+", text.strip())
    out = [p.strip() for p in parts if len(p.strip()) >= min_len]
    return out if out else [text.strip()]


def _proportional_phrase_times(
    phrases: list[str], t0: float, t1: float
) -> list[dict]:
    weights = [max(len(p), 1) for p in phrases]
    total = sum(weights) or 1
    dur = t1 - t0
    cur = t0
    result: list[dict] = []
    for i, phrase in enumerate(phrases):
        end = t1 if i == len(phrases) - 1 else cur + dur * weights[i] / total
        result.append({"start": cur, "end": end, "text": phrase})
        cur = end
    return result


def align_phrases_with_words(
    phrases: list[str],
    words: list[dict],
    seg_start: float,
    seg_end: float,
) -> list[dict]:
    if len(phrases) <= 1:
        return [{"start": seg_start, "end": seg_end, "text": phrases[0]}]

    if not words:
        return _proportional_phrase_times(phrases, seg_start, seg_end)

    wi = 0
    aligned: list[dict] = []
    for phrase in phrases:
        ptoks = [_norm_token(w) for w in phrase.split() if _norm_token(w)]
        if not ptoks:
            continue
        start_wi = wi
        matched = 0
        while wi < len(words) and matched < len(ptoks):
            wnorm = _norm_token(words[wi]["word"])
            if wnorm and (
                wnorm == ptoks[matched]
                or ptoks[matched].startswith(wnorm)
                or wnorm.startswith(ptoks[matched])
            ):
                matched += 1
            wi += 1
        if matched == 0:
            continue
        end_wi = min(wi - 1, len(words) - 1)
        aligned.append(
            {
                "start": words[start_wi]["start"],
                "end": words[end_wi]["end"],
                "text": phrase,
            }
        )

    if len(aligned) != len(phrases):
        return _proportional_phrase_times(phrases, seg_start, seg_end)

    aligned[0]["start"] = max(seg_start, aligned[0]["start"] - 0.05)
    for a in aligned:
        a["end"] = min(seg_end, a["end"] + PHRASE_END_PAD_SEC)
    return aligned


def split_by_word_pauses(words: list[dict], pause_sec: float) -> list[dict]:
    """Если нет !/? — режем по паузам между словами."""
    if len(words) < 2:
        return []
    groups: list[list[dict]] = [[words[0]]]
    for w in words[1:]:
        if w["start"] - groups[-1][-1]["end"] >= pause_sec:
            groups.append([w])
        else:
            groups[-1].append(w)
    if len(groups) <= 1:
        return []
    return [
        {
            "start": g[0]["start"],
            "end": g[-1]["end"],
            "text": " ".join(x["word"] for x in g),
        }
        for g in groups
        if len(" ".join(x["word"] for x in g).strip()) >= 4
    ]


def split_segment_into_phrases(
    seg: dict,
    *,
    word_pause_sec: float,
) -> list[dict]:
    text = seg["text"]
    words = seg.get("words") or []
    phrases = split_text_to_phrases(text)

    if len(phrases) <= 1 and words:
        pause_parts = split_by_word_pauses(words, word_pause_sec)
        if len(pause_parts) > 1:
            return pause_parts

    if len(phrases) <= 1:
        return [{"start": seg["start"], "end": seg["end"], "text": text}]

    parts = align_phrases_with_words(phrases, words, seg["start"], seg["end"])
    for p in parts:
        p.pop("words", None)
    return parts


def split_segments_by_phrases(
    segments: list[dict],
    *,
    word_pause_sec: float,
) -> list[dict]:
    out: list[dict] = []
    for seg in segments:
        for part in split_segment_into_phrases(seg, word_pause_sec=word_pause_sec):
            part.pop("words", None)
            out.append(part)
    return out


def extract_youtube_id(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[&?]|$)", url)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None


def load_last_job() -> dict | None:
    path = Path(".last-job.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_output_dir(url: str | None, override: Path | None) -> Path:
    if override is not None:
        return override
    job = load_last_job()
    if job and job.get("output"):
        return Path(job["output"])
    vid = extract_youtube_id(url)
    if vid:
        return Path("output") / vid
    return Path("output") / "untitled"


def clean_previous_exports(work: Path) -> int:
    """Удаляет phrases/ и manifest.json перед новым экспортом."""
    removed = 0
    phrases_dir = work / "phrases"
    if phrases_dir.exists():
        for path in list(phrases_dir.rglob("*")):
            if path.is_file():
                path.unlink(missing_ok=True)
                removed += 1
        shutil.rmtree(phrases_dir, ignore_errors=True)
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink(missing_ok=True)
        removed += 1
    return removed


def extend_phrase_ends(segments: list[dict], pad_sec: float = PHRASE_END_PAD_SEC) -> list[dict]:
    """Сдвигает конец каждой фразы — меньше обрывов на последнем слоге."""
    out: list[dict] = []
    for s in segments:
        item = s.copy()
        item["end"] = item["end"] + pad_sec
        out.append(item)
    return out


def merge_tiny_gaps(segments: list[dict], gap_sec: float = 0.35) -> list[dict]:
    if not segments:
        return segments
    merged = [segments[0].copy()]
    for s in segments[1:]:
        prev = merged[-1]
        if s["start"] - prev["end"] < gap_sec:
            prev["end"] = s["end"]
            prev["text"] = (prev["text"] + " " + s["text"]).strip()
        else:
            merged.append(s.copy())
    return merged


def export_clips(
    wav: Path,
    segments: list[dict],
    out_dir: Path,
    *,
    use_ollama: bool,
    ollama_model: str,
    ollama_host: str,
    pad_start_ms: int,
    pad_end_ms: int,
    fmt: str,
) -> list[dict]:
    audio = AudioSegment.from_wav(str(wav))
    manifest: list[dict] = []
    used_names: dict[str, int] = {}

    desc = "Экспорт + Ollama" if use_ollama else "Экспорт фраз"
    for i, seg in enumerate(
        tqdm(segments, desc=desc, unit="фраза", dynamic_ncols=True),
        start=1,
    ):
        start_ms = max(0, int(seg["start"] * 1000) - pad_start_ms)
        end_ms = min(len(audio), int(seg["end"] * 1000) + pad_end_ms)
        clip = audio[start_ms:end_ms]

        base = slugify(seg["text"])
        if use_ollama:
            llm_name = ollama_slug(seg["text"], ollama_model, ollama_host)
            if llm_name:
                base = llm_name

        name = base
        if name in used_names:
            used_names[name] += 1
            name = f"{base}_{used_names[name]}"
        else:
            used_names[name] = 0

        path = out_dir / f"{name}.{fmt}"
        if fmt == "mp3":
            clip.export(path, format="mp3", bitrate="192k")
        else:
            clip.export(path, format="wav")

        manifest.append(
            {
                "index": i,
                "file": path.name,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "slug": base,
            }
        )
        tqdm.write(f"  {path.name} <- {seg['text'][:72]}")

    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Разбивка YouTube на голосовые фразы")
    p.add_argument(
        "url",
        nargs="?",
        default=None,
        help="любая ссылка YouTube (не нужна с --skip-download, если есть source.wav)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="папка вывода (по умолчанию: output/VIDEO_ID или из .last-job.json)",
    )
    p.add_argument("--whisper-model", default=None, help="переопределить имя модели (редко нужно)")
    p.add_argument(
        "--profile",
        choices=list(WHISPER_PROFILES.keys()),
        default="quality",
        help="quality: быстрее | max: макс. точность и нагрузка GPU (5090)",
    )
    p.add_argument("--language", default="ru")
    p.add_argument(
        "--whisper-prompt",
        default=None,
        help="подсказка Whisper (initial_prompt); по умолчанию WHISPER_INITIAL_PROMPT в скрипте",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto: CUDA если доступна (рекомендуется для RTX)",
    )
    p.add_argument("--merge-gap", type=float, default=0.35)
    p.add_argument(
        "--dedupe-burst-gap",
        type=float,
        default=5.0,
        help="сек: одинаковый текст чаще этого интервала = одна «вспышка» (анти-петли Whisper)",
    )
    p.add_argument(
        "--split-phrases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="делить сегмент на несколько реплик по !/? и паузам между словами",
    )
    p.add_argument(
        "--word-pause",
        type=float,
        default=0.55,
        help="пауза между словами (сек) для разбиения без !/?",
    )
    p.add_argument(
        "--retranscribe",
        action="store_true",
        help="заново вызвать Whisper (по умолчанию берётся segments_raw.json если есть)",
    )
    p.add_argument(
        "--pad-ms",
        type=int,
        default=60,
        help="отступ в начале клипа при экспорте (мс)",
    )
    p.add_argument(
        "--tail-pad-ms",
        type=int,
        default=300,
        help="отступ в конце клипа (мс) — чтобы фраза не обрывалась",
    )
    p.add_argument("--format", default="mp3", choices=["mp3", "wav"])
    p.add_argument("--ollama", action="store_true")
    p.add_argument("--ollama-model", default="qwen2.5:7b")
    p.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    job = load_last_job()
    url = args.url or (job.get("url") if job else None)
    work = resolve_output_dir(url, args.output)

    if not args.skip_download and not url:
        tqdm.write("Ошибка: укажите URL YouTube")
        tqdm.write("  uv run python split_voice_lines.py \"https://www.youtube.com/watch?v=...\"")
        return 2

    retranscribe = args.retranscribe or args.profile == "max"
    whisper_prompt = (args.whisper_prompt if args.whisper_prompt is not None else WHISPER_INITIAL_PROMPT).strip() or None

    ffmpeg = configure_ffmpeg()
    device, _ = detect_device(args.device)
    profile, whisper_model, compute_type = resolve_whisper_setup(
        device, args.profile, args.whisper_model
    )

    tqdm.write(f"FFmpeg: {ffmpeg}")
    tqdm.write(f"Устройство: {device} ({compute_type})")
    tqdm.write(f"Профиль: {profile.key} — {profile.label}")
    tqdm.write(f"Модель: {whisper_model}")
    tqdm.write(f"Папка: {work.resolve()}")
    tqdm.write(f"Экспорт: {args.format.upper()} -> phrases/*.{args.format}")
    tqdm.write(
        f"Отступы: начало {args.pad_ms} мс, конец {args.tail_pad_ms} мс "
        f"(+{PHRASE_END_PAD_SEC:.2f} с к таймкоду фразы)"
    )

    work.mkdir(parents=True, exist_ok=True)

    if args.skip_download and (work / "source.wav").exists():
        wav = work / "source.wav"
        title = (job or {}).get("title", "cached")
        tqdm.write(f"Используем кэш: {wav}")
    elif args.skip_download:
        tqdm.write(f"Ошибка: нет {work / 'source.wav'} — сначала запустите с URL")
        return 2
    else:
        wav, title = download_audio(url, work, ffmpeg)
        tqdm.write(f"Скачано: {title}")
        tqdm.write(f"  -> {wav}")
        save_last_job(url, work, title)

    segments_cache = work / "segments_raw.json"
    use_cache = not retranscribe and segments_cache.exists()
    if use_cache:
        tqdm.write(f"Кэш сегментов (без Whisper): {segments_cache}")
        raw_segments = json.loads(segments_cache.read_text(encoding="utf-8"))
    else:
        tqdm.write("Транскрипция Whisper…")
        raw_segments = transcribe(
            wav,
            whisper_model,
            args.language,
            device,
            compute_type,
            profile,
            initial_prompt=whisper_prompt,
        )

    segments = prepare_whisper_segments(
        raw_segments,
        merge_gap=args.merge_gap,
        burst_gap=args.dedupe_burst_gap,
    )
    segments_cache.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if use_cache:
        tqdm.write(f"Кэш обновлён (очищенные сегменты): {segments_cache}")
    else:
        tqdm.write(f"Сегментов после Whisper: {len(segments)} -> {segments_cache}")

    before = len(segments)
    if args.split_phrases:
        segments = split_segments_by_phrases(segments, word_pause_sec=args.word_pause)
        tqdm.write(f"После разбиения по смыслу: {before} -> {len(segments)} фраз")
    else:
        for s in segments:
            s.pop("words", None)
        tqdm.write(f"Фраз: {len(segments)}")

    segments = extend_phrase_ends(segments)

    removed = clean_previous_exports(work)
    if removed:
        tqdm.write(f"Удалены старые экспорты: {removed} файл(ов)")

    clips_dir = work / "phrases"
    clips_dir.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Папка фраз: {clips_dir.resolve()} (*.{args.format})")
    manifest = export_clips(
        wav,
        segments,
        clips_dir,
        use_ollama=args.ollama,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        pad_start_ms=args.pad_ms,
        pad_end_ms=args.tail_pad_ms,
        fmt=args.format,
    )

    meta = {
        "source_url": url,
        "title": title,
        "device": device,
        "compute_type": compute_type,
        "whisper_model": whisper_model,
        "whisper_profile": profile.key,
        "pad_start_ms": args.pad_ms,
        "tail_pad_ms": args.tail_pad_ms,
        "phrase_end_pad_sec": PHRASE_END_PAD_SEC,
        "merge_gap": args.merge_gap,
        "split_phrases": args.split_phrases,
        "word_pause": args.word_pause,
        "ollama": args.ollama,
        "phrases": manifest,
    }
    manifest_path = work / "manifest.json"
    manifest_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    tqdm.write("")
    tqdm.write(f"Готово: {clips_dir} ({len(manifest)} файлов)")
    tqdm.write(f"Метаданные: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
