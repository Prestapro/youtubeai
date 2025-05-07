import os
import argparse
#
# Attempt to load speaker diarization pipeline
try:
    from pyannote.audio import Pipeline
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    if not HF_TOKEN:
        print("⚠️ HUGGINGFACE_TOKEN not set; skipping speaker diarization.")
        diarizer = None
    else:
        try:
            diarizer = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=HF_TOKEN
            )
        except Exception as e:
            print(f"⚠️ Could not load 'pyannote/speaker-diarization' pipeline: {e}")
            print("   Ensure your Hugging Face token has access and retry.")
            diarizer = None
except ImportError:
    print("⚠️ pyannote.audio not installed; skipping speaker diarization.")
    diarizer = None
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pymorphy2
import json
import subprocess
import re
import time
from collections import namedtuple
import ffmpeg

from tqdm import tqdm

# --- Проверка наличия video.json и создание, если его нет ---
if not os.path.exists('video.json'):
    with open('video.json', 'w', encoding='utf-8') as f:
        json.dump({"transcription": []}, f, ensure_ascii=False, indent=4)

# --- Вспомогательные функции для обработки временных интервалов и объединения слов ---
from datetime import datetime

# Функция для конвертации строки в формат времени
def str_to_time(time_str):
    return datetime.strptime(time_str, "%H:%M:%S,%f")

# Функция для объединения слов и временных фреймов
def merge_words(word_data):
    """
    Объединяет слова с пересекающимися временными интервалами.
    Если интервалы пересекаются или соприкасаются, объединяет слова и расширяет интервал.
    """
    if not word_data:
        return []
    merged = []
    # Сортируем по времени начала
    word_data_sorted = sorted(word_data, key=lambda w: str_to_time(w["from"]))
    cur = word_data_sorted[0].copy()
    for word_info in word_data_sorted[1:]:
        prev_to = str_to_time(cur["to"])
        cur_from = str_to_time(cur["from"])
        next_from = str_to_time(word_info["from"])
        next_to = str_to_time(word_info["to"])
        # Если пересекаются или соприкасаются (prev_to >= next_from)
        if prev_to >= next_from:
            # Объединяем слова через пробел, расширяем интервал
            cur["word"] = cur["word"] + " " + word_info["word"]
            cur["to"] = max(cur["to"], word_info["to"], key=lambda t: str_to_time(t))
        else:
            merged.append(cur)
            cur = word_info.copy()
    merged.append(cur)
    return merged

# Функция для создания файла с временными метками
def generate_timestamp_file(input_file="video.json", output_file="timestamps.json"):
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден. Пропуск генерации timestamps.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=== Содержимое video.json ===")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    timestamps = []
    # Новый способ: итерируем по data['segments'][*]['words']
    for segment in data.get('segments', []):
        for item in segment.get('words', []):
            word = item['word'].strip()
            if word.isalpha() and len(word) > 1:
                normal_form = morph.parse(word)[0].normal_form
                time_from = format_time(item['start'])
                time_to = format_time(item['end'])
                timestamps.append({
                    'word': normal_form,
                    'from': time_from,
                    'to': time_to
                })

    # Объединяем фрагменты слов с перекрывающимися временными интервалами
    merged_timestamps = merge_words(timestamps)

    # Удаляем дубликаты
    unique_timestamps = []
    seen_keys = set()
    for item in merged_timestamps:
        key = f"{item['word']}_{item['from']}_{item['to']}"
        if key not in seen_keys:
            seen_keys.add(key)
            unique_timestamps.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_timestamps, f, ensure_ascii=False, indent=4)

    print(f"Таймкоды и слова успешно сохранены в {output_file}")
    print(f"Объединено {len(timestamps)} фрагментов в {len(unique_timestamps)} слов")

# Функция для создания временных меток по словам
def adjust_timestamps_for_words(data):
    for i in range(len(data)):
        start = data[i]['timestamps']['from']
        end = data[i]['timestamps']['to']
        
        # Применяем логику для слов
        word = data[i]['text'].strip()
        if word:
            data[i]['timestamps']['from'] = start
            data[i]['timestamps']['to'] = end
    return data


# --- Функция для извлечения целых предложений ---
def extract_full_sentences(transcript):
    sentence_endings = r'([.!?])'
    sentences = re.split(sentence_endings, transcript)
    full_sentences = []

    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip() + sentences[i+1].strip()
        if sentence:
            full_sentences.append(sentence)

    return full_sentences

# --- Функция для сохранения транскрипта и метаданных в video.json ---
def save_to_json(data, filename="video.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
#
# --- Python 3.11 compat: restore inspect.getargspec for pymorphy2 ---
import inspect
if not hasattr(inspect, "getargspec"):
    def _getargspec(func):
        sig = inspect.signature(func)
        args, defaults = [], []
        varargs = varkw = None
        for p in sig.parameters.values():
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                args.append(p.name)
                if p.default is not inspect.Parameter.empty:
                    defaults.append(p.default)
            elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                varargs = p.name
            elif p.kind == inspect.Parameter.VAR_KEYWORD:
                varkw = p.name
        ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
        return ArgSpec(args, varargs, varkw,
                       tuple(defaults) if defaults else None)
    inspect.getargspec = _getargspec
# -------------------------------------------------------------------

#
# Ensure nltk punkt is downloaded and stemmer is available
nltk.download('punkt')
try:
    stemmer_ru = SnowballStemmer("russian")
    stemmer_en = SnowballStemmer("english")
except:
    nltk.download('punkt')
    stemmer_ru = SnowballStemmer("russian")
    stemmer_en = SnowballStemmer("english")

morph = pymorphy2.MorphAnalyzer()

# whisper.cpp config
WHISPER_CPP_BIN = "./whisper.cpp/main"
WHISPER_CPP_MODEL = "./whisper.cpp/models/ggml-large-v3.bin"
VIDEO_FILE = "audio.mp4"

# Функция для конвертации времени в формат SRT
def format_time(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s - int(s)) * 1000):03}"

# Файл для сохранения видео
VIDEO_FILE = "audio.mp4"

def download_audio_from_youtube(url):
    start_time = time.time()
    subprocess.run([
        "yt-dlp", "--cookies-from-browser", "chrome",
        "-f", "bestvideo+bestaudio",
        "-o", VIDEO_FILE,
        url
    ])

# --- Download audio only for fast transcription ---
def download_audio_only(youtube_url, output_path):
    """
    Скачать только аудио-дорожку из YouTube для ускоренной транскрипции в WAV.
    """
    cmd = [
        "yt-dlp",
        "--cookies-from-browser", "chrome",
        "-x",
        "--audio-format", "wav",
        "-f", "bestaudio",
        "-o", output_path,
        youtube_url
    ]
    subprocess.run(cmd, check=True)

def tokenize_words_only(data):
    result = []
    for item in data:
        # Разбиваем на слова
        tokens = word_tokenize(item["text"], language="russian")
        # Нормализуем слова, получая их леммы
        words = [morph.parse(token)[0].normal_form for token in tokens if token.isalpha() and len(token) > 1]
        result.extend(words)
    return result

# Функция для обработки видео и создания транскрипта с помощью whisper.cpp, сохранение результата в video.json

from faster_whisper import WhisperModel

def download_audio(youtube_url, output_path):
    # Скачиваем полный видеофайл (видео+аудио) для нарезки
    cmd = [
        "yt-dlp",
        "--cookies-from-browser", "chrome",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]",
        "--merge-output-format", "mp4",
        "-o", output_path,
        youtube_url
    ]
    subprocess.run(cmd, check=True)

def transcribe_audio(audio_path):
    print("📥 Загружаем модель Faster-Whisper...")
    # Подбор устройства: MPS на macOS (Apple Silicon), иначе CPU
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"Используем устройство для транскрипции: {device}")
    model = WhisperModel("medium", device=device, compute_type="int8")
    print("🧠 Транскрибируем аудио (c word_timestamps)...")
    segments, info = model.transcribe(audio_path, beam_size=1, word_timestamps=True)
    segments = list(segments)
    print(f"✅ Язык: {info.language}, продолжительность: {info.duration:.2f} сек")

    # Выполняем диаризацию
    if diarizer:
        print("🎙️ Выполняем диаризацию спикеров...")
        diarization = diarizer(audio_path)
    else:
        diarization = []

    result = {"text": "", "segments": [], "language": info.language}
    for i, segment in enumerate(segments):
        # Определяем спикера по началу сегмента
        speaker_label = None
        if diarizer:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if segment.start >= turn.start and segment.start < turn.end:
                    speaker_label = speaker
                    break
        else:
            # Fallback: alternate speakers if no diarization available
            speaker_label = f"speaker_{i % 2}"

        seg_data = {
            "id": i,
            "speaker": speaker_label or "Unknown",
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in (segment.words or [])
            ]
        }
        result["segments"].append(seg_data)
        # Append each speaker segment on its own line
        result["text"] += f"{speaker_label or 'Unknown'}: {segment.text}\n"

    return result

def print_word_timestamps(transcription):
    for segment in tqdm(transcription['segments'], desc="⏱ Сегменты"):
        for word_info in segment.get('words', []):
            word = word_info['word']
            start_time = word_info['start']
            print(f"{word} [{start_time:.2f}s]")


# Функция для нарезки клипов (видео+аудио)
def extract_clips(audio_file, segments, video_file="audio.mp4"):
    start_time_total = time.time()
    start_time_cut = time.time()
    for idx, (start, end) in enumerate(tqdm(segments, desc="🎞 Нарезка клипов"), 1):
        safe_start = format_time(start).replace(",", "_").replace(":", "_")
        safe_end = format_time(end).replace(",", "_").replace(":", "_")
        output_filename = f"clip_{idx}_{safe_start}_{safe_end}.mp4"
        try:
            # Добавляем drawtext фильтр с таймкодом в правом нижнем углу
            timestamp = f"{int(start // 60):02}\\:{int(start % 60):02}"
            (
                ffmpeg
                .input(video_file, ss=start)
                .output(
                    output_filename,
                    vf=f"drawtext=fontfile=/System/Library/Fonts/SFNSDisplay.ttf:text='{timestamp}':x=w-tw-10:y=h-th-10:fontsize=24:fontcolor=white:borderw=2",
                    vcodec="libx264",
                    acodec="aac",
                    audio_bitrate="192k",
                    format="mp4",
                    t=end - start,
                    **{'metadata': 'timecode=00:00:00:00'}
                )
                .overwrite_output()
                .run(quiet=False)
            )
            print(f"Создан файл: {output_filename}")
        except ffmpeg.Error as e:
            if e.stderr is not None:
                try:
                    error_message = e.stderr.decode()
                except Exception:
                    error_message = str(e)
            else:
                error_message = str(e)
            print(f"Ошибка при нарезке файла {output_filename}: {error_message}")
    cut_time = time.time() - start_time_cut
    print(f"⏱ Нарезка заняла {cut_time:.2f} секунд")
    log_time("Нарезка клипов", start_time_cut)

    # Склеиваем все клипы в один файл с помощью ffmpeg concat и -c copy
    # Список всех названий фрагментов
    final_clips = []
    for idx in range(1, len(segments) + 1):
        start_fmt = format_time(segments[idx - 1][0]).replace(",", "_").replace(":", "_")
        end_fmt = format_time(segments[idx - 1][1]).replace(",", "_").replace(":", "_")
        clip_name = f"clip_{idx}_{start_fmt}_{end_fmt}.mp4"
        final_clips.append(clip_name)

    # Фильтруем непустые файлы (размер больше 2КБ)
    final_clips = [f for f in final_clips if os.path.exists(f) and os.path.getsize(f) > 2048]

    # Генерируем временный файл list.txt со списком фрагментов
    with open("list.txt", "w") as f:
        for filename in final_clips:
            f.write(f"file '{filename}'\n")

    # Склейка с помощью ffmpeg-python
    try:
        (
            ffmpeg
            .input('list.txt', format='concat', safe=0)
            .output('final.mp4', c='copy')
            .run(overwrite_output=True)
        )
        print("✅ Склеенный файл сохранён как final.mp4")
        log_time("Склейка финального файла", start_time_total)
    except ffmpeg.Error as e:
        if e.stderr is not None:
            try:
                error_message = e.stderr.decode()
            except Exception:
                error_message = str(e)
        else:
            error_message = str(e)
        print(f"❌ Ошибка при склейке клипов: {error_message}")
# Утилита для логирования времени этапов

def log_time(stage, start_time):
    elapsed_time = time.time() - start_time
    print(f"⏱ {stage} заняло {elapsed_time:.2f} секунд")

#
# --- Add extract_by_keyword function ---
def extract_by_keyword(
    keyword,
    video_json="video.json",
    audio_file="audio.mp3",
    video_file="audio.mp3",
    allow_partial=False,
    padding_before=1.0,
    padding_after=1.0
):
    # now supports custom padding before/after each segment
    with open(video_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Определяем язык и нормализацию
    lang = data.get("language", "en")
    if lang == "ru":
        normalize = lambda w: morph.parse(w)[0].normal_form
    else:
        normalize = lambda w: stemmer_en.stem(w.lower())

    target_segments = []

    # Multi-word phrase normalization
    phrase_tokens = [normalize(re.sub(r"[^\wёЁа-яА-Яa-zA-Z]", "", w)) for w in keyword.split() if w.strip()]
    phrase_length = len(phrase_tokens)

    print(f"[DEBUG] phrase_tokens: {phrase_tokens}")

    for segment in tqdm(data.get("segments", []), desc="🔍 Поиск фразы"):
        words = segment.get("words", [])
        print(f"📋 Сегмент: {segment.get('text', '')[:50]}...")
        norm_words = []
        if words:
            for word_info in words:
                raw = word_info.get("word", "").strip()
                if not raw:
                    continue
                cleaned = re.sub(r"[^\wёЁа-яА-Яa-zA-Z]", "", raw.lower())
                if not cleaned:
                    continue
                try:
                    norm = normalize(cleaned)
                except Exception as e:
                    print(f"[WARN] Can't normalize '{cleaned}': {e}")
                    continue
                norm_words.append((norm, word_info["start"], word_info["end"]))
        else:
            print("[DEBUG] Пропущен сегмент без words")
            continue
        print(f"[DEBUG] norm_words: {norm_words}")

        for i in range(len(norm_words) - phrase_length + 1):
            window = norm_words[i:i+phrase_length]
            print(f"[DEBUG] Checking window: {[w[0] for w in window]} vs {phrase_tokens}")
            if all(
                (p == w[0]) or (
                    allow_partial
                    and len(p) >= 3
                    and (w[0].startswith(p) or w[0].endswith(p))
                )
                for p, w in zip(phrase_tokens, window)
            ):
                start = max(0, window[0][1] - padding_before)
                end = window[-1][2] + padding_after
                print(f"🔎 Найдена фраза на {start:.2f}s – {end:.2f}s")
                target_segments.append((start, end))

    # Merge overlapping or adjacent segments to avoid duplicates
    target_segments.sort(key=lambda x: x[0])
    merged_segments = []
    for start, end in target_segments:
        if not merged_segments:
            merged_segments.append((start, end))
        else:
            prev_start, prev_end = merged_segments[-1]
            if start <= prev_end:
                # Overlaps or adjacent: extend the previous segment
                merged_segments[-1] = (prev_start, max(prev_end, end))
            else:
                merged_segments.append((start, end))
    target_segments = merged_segments

    # Merge overlapping or adjacent segments to avoid duplicates
    target_segments.sort(key=lambda x: x[0])
    merged_segments = []
    for start, end in target_segments:
        if not merged_segments:
            merged_segments.append((start, end))
        else:
            prev_start, prev_end = merged_segments[-1]
            if start <= prev_end:
                # Overlaps or adjacent: extend the previous segment
                merged_segments[-1] = (prev_start, max(prev_end, end))
            else:
                merged_segments.append((start, end))
    target_segments = merged_segments

    if not target_segments:
        print(f"Фраза '{keyword}' не найдена.")
        return

    print(
        f"Найдено {len(target_segments)} вхождений фразы '{keyword}'"
        + (" (частичное совпадение)" if allow_partial else "")
    )
    extract_clips(audio_file, target_segments, video_file=video_file)

# --- Основная функция: скачивание, обработка, транскрипция ---
def generate_timestamp_file(input_file="video.json", output_file="timestamps.json"):
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден. Пропуск генерации timestamps.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    timestamps = []
    # Сохраняем только слова из букв, длиной больше 1 символа, и нормализуем
    for item in data['transcription']:
        word = item['text'].strip()
        if word.isalpha() and len(word) > 1:
            normal_form = morph.parse(word)[0].normal_form
            time_from = item['timestamps']['from']
            time_to = item['timestamps']['to']
            timestamps.append({
                'word': normal_form,
                'from': time_from,
                'to': time_to
            })

    # Объединяем фрагменты слов с перекрывающимися временными интервалами
    merged_timestamps = merge_words(timestamps)

    # Удаляем дубликаты
    unique_timestamps = []
    seen_keys = set()
    for item in merged_timestamps:
        key = f"{item['word']}_{item['from']}_{item['to']}"
        if key not in seen_keys:
            seen_keys.add(key)
            unique_timestamps.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_timestamps, f, ensure_ascii=False, indent=4)

    print(f"Таймкоды и слова успешно сохранены в {output_file}")
    print(f"Объединено {len(timestamps)} фрагментов в {len(unique_timestamps)} слов")
    # (Repeated block removed.)

nltk.download('punkt')
try:
    stemmer = SnowballStemmer("russian")
except:
    nltk.download('punkt')
    stemmer = SnowballStemmer("russian")

morph = pymorphy2.MorphAnalyzer()

WHISPER_CPP_BIN = "./whisper.cpp/main"
WHISPER_CPP_MODEL = "./whisper.cpp/models/ggml-large-v3.bin"
VIDEO_FILE = "audio.mp4"

def format_time(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m):02}:{int(s):02},{int((s - int(s)) * 1000):03}"

def download_audio_from_youtube(url):
    start_time = time.time()
    subprocess.run([
        "yt-dlp", "--cookies-from-browser", "chrome",
        "-f", "bestvideo+bestaudio",
        "-o", VIDEO_FILE,
        url
    ])

def tokenize_words_only(data):
    result = []
    for item in data:
        tokens = word_tokenize(item["text"], language="russian")
        words = [morph.parse(token)[0].normal_form for token in tokens if token.isalpha() and len(token) > 1]
        result.extend(words)
    return result

def extract_clips(audio_file, segments, video_file="audio.mp4"):
    start_time_total = time.time()
    start_time_cut = time.time()
    total_segments = len(segments)
    print(f"🔍 Всего сегментов: {total_segments}")
    for idx, (start, end) in enumerate(segments, 1):
        if end - start < 0.3:
            print(f"⚠️ Пропущен сегмент {idx}: длительность слишком мала ({end - start:.2f} сек)")
            continue
        if not os.path.exists(video_file):
            print(f"🚫 Файл {video_file} не найден. Пропуск сегмента {idx}.")
            continue
        print(f"[DEBUG] Сегмент {idx}: start={start:.2f}, end={end:.2f}, длительность={end - start:.2f}")
        safe_start = format_time(start).replace(",", "_").replace(":", "_")
        safe_end = format_time(end).replace(",", "_").replace(":", "_")
        output_filename = f"clip_{idx}_{safe_start}_{safe_end}.mp4"
        print(f"[DEBUG] Клип {idx}: {start:.2f}s – {end:.2f}s ({end - start:.2f}s)")
        try:
            # Human-readable timestamp HH:MM:SS
            hours = int(start // 3600)
            mins = int((start % 3600) // 60)
            secs = int(start % 60)
            timestamp = f"{hours:02}\\:{mins:02}\\:{secs:02}"
            counter = f"{idx}/{total_segments}"
            (
                ffmpeg
                .input(video_file, ss=start)
                .output(
                    output_filename,
                    vf=f"drawtext=fontfile=/System/Library/Fonts/SFNSDisplay.ttf:text='{timestamp} {counter}':x=w-tw-10:y=h-th-10:fontsize=24:fontcolor=white:borderw=2",
                    vcodec="libx264",
                    acodec="aac",
                    audio_bitrate="192k",
                    format="mp4",
                    t=end - start,
                    **{'metadata': 'timecode=00:00:00:00'}
                )
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
            if os.path.exists(output_filename) and os.path.getsize(output_filename) < 2048:
                print(f"[WARNING] Клип {output_filename} слишком мал: {os.path.getsize(output_filename)} байт")
            print(f"Создан файл: {output_filename}")
        except ffmpeg.Error as e:
            if e.stderr is not None:
                print(f"[FFmpeg ERROR] Сегмент {idx}:")
                try:
                    error_message = e.stderr.decode()
                except Exception:
                    error_message = str(e)
            else:
                error_message = str(e)
            print(f"Ошибка при нарезке файла {output_filename}: {error_message}")
            print("🚫 FFmpeg stderr:")
            print(error_message)
    cut_time = time.time() - start_time_cut
    print(f"⏱ Нарезка заняла {cut_time:.2f} секунд")
    log_time("Нарезка клипов", start_time_cut)

    final_clips = []
    for idx, (start, end) in enumerate(segments, 1):
        if end - start < 0.3:
            continue
        start_fmt = format_time(start).replace(",", "_").replace(":", "_")
        end_fmt = format_time(end).replace(",", "_").replace(":", "_")
        clip_name = f"clip_{idx}_{start_fmt}_{end_fmt}.mp4"
        if os.path.exists(clip_name) and os.path.getsize(clip_name) > 2048:
            final_clips.append(clip_name)

    with open("list.txt", "w") as f:
        for filename in final_clips:
            f.write(f"file '{filename}'\n")

    if not final_clips:
        print("⚠️ Нет валидных клипов для склейки. Пропуск финального объединения.")
        if os.path.exists("list.txt"):
            os.remove("list.txt")
        return

    try:
        (
            ffmpeg
            .input('list.txt', format='concat', safe=0)
            .output('final.mp4', c='copy')
            .run(overwrite_output=True)
        )
        print("✅ Склеенный файл сохранён как final.mp4")
        log_time("Склейка финального файла", start_time_total)
    except ffmpeg.Error as e:
        if e.stderr is not None:
            try:
                error_message = e.stderr.decode()
            except Exception:
                error_message = str(e)
        else:
            error_message = str(e)
        print(f"❌ Ошибка при склейке клипов: {error_message}")
    # Удаление временного файла list.txt
    if os.path.exists("list.txt"):
        os.remove("list.txt")

def log_time(stage, start_time):
    elapsed_time = time.time() - start_time
    print(f"⏱ {stage} заняло {elapsed_time:.2f} секунд")

def main():
    parser = argparse.ArgumentParser(description="Extract clips from YouTube video by keywords.")
    parser.add_argument("--video", type=str, required=True, help="YouTube video URL")
    parser.add_argument("--keywords", type=str, required=True, help="Ключевые слова через запятую")
    parser.add_argument("--partial", action="store_true",
                        help="Разрешить частичное совпадение слов (пример = например), но не подстроки (мир ≠ мироед)")
    parser.add_argument(
        "--padding-before", type=float, default=1.0,
        help="Секунд для добавления перед найденным фрагментом (по умолчанию 1.0)"
    )
    parser.add_argument(
        "--padding-after", type=float, default=1.0,
        help="Секунд для добавления после найденного фрагмента (по умолчанию 1.0)"
    )
    parser.add_argument(
        "--keep-clips", action="store_true",
        help="Не удалять отдельные клипы после объединения в final.mp4 (по умолчанию клипы удаляются)"
    )
    parser.add_argument(
        "--download-video", action="store_true",
        help="Сохранять оригинальный видеофайл после обработки (по умолчанию удаляется)"
    )
    args = parser.parse_args()
    # Удаляем старый JSON, чтобы не использовать устаревшие данные
    if os.path.exists("video.json"):
        os.remove("video.json")

    youtube_spec = os.path.expanduser(args.video)
    keywords = [w.strip() for w in args.keywords.split(",") if w.strip()]
    if not keywords:
        print("Ключевые слова не введены. Завершение.")
        return

    youtube_spec = os.path.expanduser(args.video)
    if os.path.exists(youtube_spec):
        audio_file = video_file = youtube_spec
        downloaded = False
    else:
        # Скачиваем аудио для транскрипции
        audio_file = "audio.wav"
        print("🔻 Скачиваем аудио для транскрипции...")
        download_audio_only(args.video, audio_file)
        # Скачиваем видео+аудио, если нужен оригинал
        if args.download_video:
            video_file = "video.mp4"
            print("🔻 Скачиваем оригинал видео+аудио в video.mp4...")
            download_audio(args.video, video_file)
        else:
            video_file = audio_file
        downloaded = True

    print("🧠 Распознавание речи...")
    transcription = transcribe_audio(audio_file)

    print("🕒 Результаты с таймкодами:")
    print_word_timestamps(transcription)

    print("💾 Сохраняем результат в video.json...")
    with open("video.json", "w", encoding="utf-8") as f:
        json.dump(transcription, f, ensure_ascii=False, indent=4)

    # Сохраняем полный текст транскрипции с инкрементированием имени
    base_name = "transcript"
    ext = ".txt"
    file_name = base_name + ext
    counter = 1
    while os.path.exists(file_name):
        file_name = f"{base_name}_{counter}{ext}"
        counter += 1
    print(f"💾 Сохраняем полный текст транскрипции в {file_name}...")
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(transcription.get("text", "").strip())

    for keyword in keywords:
        print(f"🎯 Поиск ключевой фразы и нарезка клипов для '{keyword}'...")
        extract_by_keyword(
            keyword,
            audio_file=audio_file,
            video_file=video_file,
            allow_partial=args.partial,
            padding_before=args.padding_before,
            padding_after=args.padding_after
        )
        if not args.keep_clips:
            import glob
            for clip_file in glob.glob("clip_*.mp4"):
                if os.path.basename(clip_file) != "final.mp4":
                    os.remove(clip_file)
            print("🗑️ Удалены отдельные клипы, оставлен только final.mp4")

    # Clean up downloaded file if needed
    if downloaded:
        if not args.download_video:
            print("🧹 Удаление временного видеофайла...")
            os.remove(video_file)
        else:
            print(f"💾 Оригинальный видеофайл сохранён как {video_file}")

if __name__ == "__main__":
    main()
