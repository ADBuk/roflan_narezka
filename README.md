# Нарезка голосовых фраз с YouTube

Инструмент для **автоматической нарезки** длинных роликов с голосовыми репликами на отдельные аудиофайлы.

## Содержание

- [Зачем этот проект](#зачем-этот-проект)
- [Установка](#установка)
- [Скорость транскрипции](#скорость-транскрипции-ориентир)
- [Cookies](#cookies)
- [Команды запуска](#все-команды-запуска-uv)
- [Настройки точности](#настройки-точности-нарезки)
- [Промпт Whisper](#промпт-whisper-initial_prompt)
- [Пример с нуля](#пример-с-нуля)

## Зачем этот проект

Пайплайн:

1. **Скачивает** звук с YouTube (нужен `cookies.txt` для возрастных видео).
2. **Распознаёт речь** через Whisper (`faster-whisper`, желательно на GPU).
3. **Режет** на отдельные фразы по знакам препинания и паузам.
4. **Сохраняет MP3** с именами из текста реплики (транслит, без префикса `001_`).

Итог — папка `phrases/` с отдельным файлом на каждую распознанную реплику. Один скрипт, один `uv`-проект.

## Структура проекта

| Файл | Назначение |
|------|------------|
| `split_voice_lines.py` | скачивание (yt-dlp), транскрипция, нарезка, экспорт |
| `pyproject.toml` | зависимости [uv](https://docs.astral.sh/uv/) |
| `cookies.example.txt` | пример формата cookies |
| `cookies.txt` | локальные cookies YouTube (создаёте у себя на диске) |

Папка `output/` создаётся автоматически.

## Результат

После обработки ссылки `https://www.youtube.com/watch?v=VIDEO_ID` (или своей папки `-o output/...`) создаётся каталог:

```
output/<имя_папки>/
  source.wav              # полное аудио ролика
  segments_raw.json       # сегменты Whisper (кэш для повторного экспорта)
  manifest.json           # список фраз: файл, таймкоды, текст
  phrases/
    <slug>.mp3            # одна реплика = один файл
    <slug>_1.mp3          # суффикс, если та же фраза встречается ещё раз
    ...
```

| Что | Описание |
|-----|----------|
| Формат | MP3 192 kbps по умолчанию (`--format wav` — без сжатия) |
| Имена файлов | Транслит из распознанного текста; повторы получают `_1`, `_2`, … |
| Папка по умолчанию | `output/VIDEO_ID/`, если не задан `-o` |
| Обрезка конца | Параметры `--tail-pad-ms` и внутренний отступ, чтобы не резать слово |

## Требования

| Что | Зачем |
|-----|--------|
| **[uv](https://docs.astral.sh/uv/)** | venv и зависимости |
| **Node.js** | YouTube EJS; [nodejs.org](https://nodejs.org) |
| **`cookies.txt`** в корне | возрастные видео; из IDE DPAPI не работает |
| **NVIDIA GPU** (опционально) | быстрый Whisper |

FFmpeg — через `imageio-ffmpeg`, отдельно не ставить.

## Установка

### 1. Клонировать репозиторий

```bash
git clone <URL-репозитория> rofl
cd rofl
```

### 2. Установить [uv](https://docs.astral.sh/uv/)

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux / macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Зависимости проекта

```powershell
uv sync
```

Создаётся `.venv/` с Python, `faster-whisper`, `yt-dlp`, CUDA-библиотеками для NVIDIA.

### 4. Проверка GPU

```powershell
uv run python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
```

Ожидается `CUDA devices: 1` или больше. Если `0` — Whisper пойдёт на CPU (см. таблицу скорости ниже).

### 5. Cookies и первый запуск

См. раздел [Cookies](#cookies) и [пример с нуля](#пример-с-нуля).

---

## Скорость транскрипции (ориентир)

Оценки для профиля **`quality`** (модель **large-v3**, float16, beam 5).  
Исходник для оценки: **~32 минуты** русской речи (один длинный ролик).  
Скачивание с YouTube и экспорт MP3 **не входят** в таблицу (обычно +2–5 мин суммарно).

Профиль **`max`** (float32, beam 10) — примерно **в 3–4 раза дольше**, чем `quality`.

| GPU | VRAM | `quality` (~32 мин аудио) | Примечание |
|-----|------|---------------------------|------------|
| **Без GPU (CPU)** | — | **1.5–3 ч** | автоматически `medium`, не `large-v3` |
| GTX 1660 / RTX 2060 | 6 GB | 25–40 мин | мало VRAM; возможен `medium` вместо large |
| RTX 3060 / 3060 Ti | 8–12 GB | 12–20 мин | стабильный минимум для large-v3 |
| RTX 3070 / 3080 / 4070 | 8–12 GB | 10–16 мин | |
| RTX 4060 Ti 16GB | 16 GB | 8–12 мин | |
| RTX 4080 / 4070 Ti Super | 16 GB | 6–10 мин | |
| RTX 4090 | 24 GB | 4–7 мин | |
| RTX 5090 | 32 GB | **3–5 мин** | |

**Оценка для другой длины:** время транскрипции ≈ (длительность аудио) × (время из таблицы / 32 мин).  
Пример: RTX 4090, 60 мин аудио → около **9–12 мин** на `quality`.

| Длина ролика | RTX 3060 (`quality`) | RTX 4090 (`quality`) | RTX 5090 (`quality`) |
|--------------|----------------------|----------------------|----------------------|
| 10 мин | ~4–7 мин | ~1.5–2.5 мин | ~1–2 мин |
| 32 мин | ~12–20 мин | ~4–7 мин | ~3–5 мин |
| 60 мин | ~22–38 мин | ~8–13 мин | ~6–10 мин |
| 120 мин | ~45–75 мин | ~15–25 мин | ~12–20 мин |

Точное время зависит от версии драйвера, фона в аудио, числа реплик и загрузки GPU. Первый запуск дольше — качается модель Whisper (~3 GB).

---

## Cookies

1. Расширение **Get cookies.txt LOCALLY**:  
   https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc
2. Откройте видео на **youtube.com**, подтвердите возраст при необходимости.
3. Export → `cookies.txt` в корень проекта (шаблон: `cookies.example.txt`).

Обновляйте при `Sign in`, `age-restricted`, `403`, «format is not available».

---

## Все команды запуска (uv)

### 1. Первый раз: скачивание + нарезка

```powershell
uv run python split_voice_lines.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Своя папка:

```powershell
uv run python split_voice_lines.py "https://www.youtube.com/watch?v=VIDEO_ID" -o output/my_project
```

### 2. Быстрая перенарезка (аудио и Whisper уже есть)

Не трогает `segments_raw.json`, только заново режет MP3:

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --format mp3
```

С усиленным хвостом (если всё ещё режет конец):

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --format mp3 --tail-pad-ms 400
```

### 3. Точная версия (максимум качества) — рекомендуется для финала

Профиль **`max`**: large-v3, float32, beam 10, заново транскрибирует.  
Если `source.wav` уже есть:

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --profile max --format mp3 --tail-pad-ms 350 --pad-ms 60 --merge-gap 0.35 --word-pause 0.55
```

Полный цикл с нуля (URL + max):

```powershell
uv run python split_voice_lines.py "https://www.youtube.com/watch?v=VIDEO_ID" -o output/my_project --profile max --format mp3 --tail-pad-ms 350
```

### 4. Обычное качество (быстрее; см. [таблицу скорости](#скорость-транскрипции-ориентир))

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --profile quality --format mp3
```

### 5. Заново Whisper без смены профиля

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --retranscribe --format mp3
```

### 6. Только скачать аудио (без нарезки)

Скрипт всегда режет после скачивания; чтобы только WAV — прервите после строки «Скачано» или используйте отдельно yt-dlp. Практичнее: скачать и сразу нарезать командой из п. 1.

### 7. WAV вместо MP3

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --format wav
```

### 8. CPU (без GPU)

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --device cpu --whisper-model medium --format mp3
```

### 9. Имена файлов через Ollama

```powershell
ollama pull qwen2.5:7b
uv run python split_voice_lines.py --skip-download -o output/my_project --ollama --format mp3
```

### 10. Одна длинная реплика = один файл (без разбиения по `!`/`?`)

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --no-split-phrases --format mp3
```

### 11. Справка по флагам

```powershell
uv run python split_voice_lines.py --help
```

---

## Настройки точности нарезки

Нарезка идёт в три этапа: **Whisper** → **склейка/разбиение фраз** → **экспорт с отступами**. Ниже — что на что влияет.

### Профили Whisper

| Профиль | Модель | GPU | Beam | Когда |
|---------|--------|-----|------|--------|
| **quality** (по умолчанию) | large-v3, float16 | средняя | 5 | баланс скорости и точности |
| **max** | large-v3, float32 | высокая | 10 | финальная точная версия; сам вызывает `--retranscribe` |

В профиле `max` также выше `speech_pad_ms` у VAD. Включены `repetition_penalty` и фильтр петель Whisper (одинаковые фразы подряд), чтобы не плодить сотни копий одной реплики.

Если в кэше остались «петли» Whisper, перетранскрибируйте: `--retranscribe` или снова `--profile max`.

### Границы между репликами (после Whisper)

| Параметр | По умолчанию | Эффект |
|----------|--------------|--------|
| `--merge-gap` | `0.35` | Секунды: если две соседние реплики Whisper ближе — **склеиваются** в одну. Увеличьте (`0.5`), если одна фраза разбилась на два файла. Уменьшите (`0.25`), если в одном MP3 две реплики. |
| `--dedupe-burst-gap` | `5.0` | Одинаковый текст чаще чем раз в N секунд считается галлюцинацией Whisper и схлопывается в один сегмент. |
| `--split-phrases` | включено | Длинный сегмент режется по `!` / `?` и паузам между словами. |
| `--no-split-phrases` | — | Один сегмент Whisper = один файл. |
| `--word-pause` | `0.55` | Пауза между словами (сек): если нет `!`/`?`, режет по тишине между словами. Больше (`0.65`) — меньше мелких кусков; меньше (`0.45`) — больше отдельных фраз. |

### Чтобы конец фразы не обрывался

| Параметр | По умолчанию | Эффект |
|----------|--------------|--------|
| `--tail-pad-ms` | `300` | Добавляет тишину/хвост **после** конца таймкода при вырезке MP3. Главный рычаг против обрыва. Попробуйте `350`–`450`, если слышно обрезание. |
| `--pad-ms` | `60` | Отступ **в начале** клипа (не режет атаку слова). |
| (внутри скрипта) | `+0.15` с | К каждой фразе после разбиения добавляется `PHRASE_END_PAD_SEC` к полю `end` перед экспортом. |

Итого на конец фразы: ~**0.15 с** к таймкоду + **`--tail-pad-ms`** при нарезке WAV → MP3.

### Прочее

| Параметр | По умолчанию | Эффект |
|----------|--------------|--------|
| `--language` | `ru` | Язык Whisper |
| `--whisper-prompt` | из `WHISPER_INITIAL_PROMPT` | Подсказка распознаванию (см. раздел ниже) |
| `--device` | `auto` | `cuda` / `cpu` |
| `--retranscribe` | выкл | Игнорировать `segments_raw.json` |
| `-o` / `--output` | `output/VIDEO_ID` | Папка; можно `output/my_project` |
| `--format` | `mp3` | `mp3` или `wav` |

### Типичные сочетания

**Финал (точно + мягкий хвост):**

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --profile max --format mp3 --tail-pad-ms 350 --pad-ms 60
```

**Быстро поправить только обрезанные концы** (без Whisper):

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --format mp3 --tail-pad-ms 400
```

**Две реплики склеились в один MP3:**

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --merge-gap 0.25 --format mp3
```

**Одна реплика разбилась на два файла:**

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --merge-gap 0.5 --format mp3
```

---

## Типичные проблемы

| Симптом | Решение |
|---------|---------|
| Нет `cookies.txt` | экспорт cookies (раздел выше) |
| `n challenge solving failed` | Node.js, новый терминал |
| `cublas64_12.dll not found` | `uv sync`, новый терминал |
| Режет конец слова / интонацию | `--tail-pad-ms 400` или `450` |
| Плохой текст / границы | `--profile max` (перетранскрибирует) |
| В одном MP3 две фразы | `--merge-gap 0.25` или проверьте `--split-phrases` |
| Один MP3 на полстроки | `--merge-gap 0.5` |
| Сотни файлов `phrase_1` … `phrase_99` с одним текстом | петли Whisper в `segments_raw.json`; перезапустите нарезку (кэш очистится) или `--retranscribe` |

Перед экспортом удаляются `phrases/` и `manifest.json`; `segments_raw.json` сохраняется, пока нет `--retranscribe` или `--profile max`.

---

## Дубликаты имён (`phrase.mp3`, `phrase_1.mp3`, …)

| Причина | Что делать |
|---------|------------|
| **Галлюцинация Whisper** — одна фраза десятки раз подряд в транскрипте | Перезапуск нарезки (очистка кэша встроена) или `--retranscribe`; настройка `--dedupe-burst-gap` |
| **Нормальный повтор в ролике** — та же реплика в разных местах с паузой 30+ с | Ожидаемо: несколько файлов с суффиксами `_1`, `_2` |

---

## Промпт Whisper (`initial_prompt`)

Это **не** чат-промпт и **не** то же самое, что `--ollama` (Ollama только переименовывает уже готовые файлы).

**Whisper `initial_prompt`** — короткая текстовая подсказка в начале распознавания: модель чаще верно пишет слова и обороты, которые часто встречаются в вашем ролике (имена, жаргон, термины, аббревиатуры). Влияет на **текст в `segments_raw.json`** и на **имена MP3** (они строятся из текста).

Подсказка должна быть **короткой** — 1–2 предложения или список из 5–15 характерных слов из аудио. Длинный текст ухудшает распознавание.

После смены промпта нужен **`--retranscribe`** (или `--profile max`), иначе возьмётся старый `segments_raw.json`.

### Где менять

**Способ 1 — в коде (постоянно для всех запусков):**

В `split_voice_lines.py` найдите константу `WHISPER_INITIAL_PROMPT` (рядом с `WHISPER_PROFILES`):

```python
WHISPER_INITIAL_PROMPT = "Добрый день, спасибо, подтвердите, доставка, заказ."
```

**Способ 2 — из командной строки (разовый запуск):**

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --retranscribe --whisper-prompt "типичные слова и фразы из вашего ролика"
```

**Способ 3 — параметры `transcribe()`**

Тонкая настройка (VAD, beam, temperature) — в функции `transcribe()` в том же файле, вызов `model.transcribe(...)`.

### Имена файлов через LLM (отдельно)

Промпт для **Ollama** — функция `ollama_slug()` в `split_voice_lines.py`, флаг `--ollama`. Меняет только slug файла, не транскрипцию Whisper.

---

## Пример с нуля

После [установки](#установка):

```bash
uv sync
cp cookies.example.txt cookies.txt   # затем заменить на экспорт из браузера
uv run python split_voice_lines.py "https://www.youtube.com/watch?v=VIDEO_ID" -o output/my_project
```

Затем точная перенарезка:

```powershell
uv run python split_voice_lines.py --skip-download -o output/my_project --profile max --format mp3 --tail-pad-ms 350
```
