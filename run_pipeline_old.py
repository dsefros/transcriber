#!/usr/bin/env python3
"""
Единый pipeline для автоматической обработки встреч:
1. Транскрибация + диаризация через WhisperX
2. Анализ через локальную LLM (Ollama)
3. Генерация структурированного отчёта в Markdown

Поддерживает offline-режим без интернета и без Sber API.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import torch
import ollama
from datetime import datetime

# === Загрузка конфигурации ===
load_dotenv()
INPUT_DIR = os.getenv("INPUT_DIR", "input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
MODEL_NAME = os.getenv("MODEL_NAME", "phi3:medium-128k")
HF_TOKEN = os.getenv("HF_TOKEN")  # Для WhisperX (диаризация)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. Транскрибация и диаризация (на основе transcribe_v3.py) ===
def transcribe_and_diarize(audio_path: str, device: str = "cuda"):
    """Транскрибация + диаризация через WhisperX."""
    import whisperx
    from pydub import AudioSegment
    
    print(f"[DEBUG] Загрузка WhisperX (large-v3) на устройстве: {device}")
    model = whisperx.load_model("large-v3", device, compute_type="float16" if device == "cuda" else "int8")
    
    # Конвертация в WAV 16kHz моно
    print("[DEBUG] Конвертация аудио в WAV...")
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    wav_path = "temp_audio.wav"
    audio.export(wav_path, format="wav")
    
    # Транскрибация
    print("[DEBUG] Транскрибация...")
    audio_data = whisperx.load_audio(wav_path)
    result = model.transcribe(audio_data, language="ru")
    
    # Выравнивание
    model_a, metadata = whisperx.load_align_model(
        language_code="ru",
        device=device,
        model_name="facebook/wav2vec2-base-960h"
    )
    
    # Диаризация
    print("[DEBUG] Диаризация...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
    diarize_segments = diarize_model(audio_data)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # Формирование сегментов
    segments = []
    for seg in result["segments"]:
        segments.append({
            "text": seg.get("text", "").strip(),
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "speaker": seg.get("speaker", "SPEAKER_00")
        })
    
    # Очистка временных файлов
    if os.path.exists(wav_path):
        os.remove(wav_path)
    
    print(f"[DEBUG] WhisperX завершён. Сегментов: {len(segments)}")
    return segments

# === 2. Анализ через Ollama (локальная LLM) ===
def analyze_with_ollama(segments: list) -> str:
    """Анализ диалога через локальную модель Ollama."""
    print(f"[DEBUG] Анализ через Ollama ({MODEL_NAME})...")
    
    # Формирование диалога для промпта
    dialogue = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in segments if seg["text"]])
    
    # Системный промпт, заточенный под вашу предметную область
    system_prompt = """
Ты — эксперт по анализу технических встреч в компании Соммерс. Соммерс специализируется на IT-решениях для эквайринга, POS-терминалов и автоматизации бизнеса. Твои клиенты — банки, ритейл и частные мерчанты.

Проанализируй диалог и подготовь СТРУКТУРИРОВАННЫЙ ОТЧЁТ в формате Markdown со следующими разделами:

### 📝 Краткое содержание
- Общая тема встречи (1-2 предложения)
- Ключевые обсуждаемые направления (максимум 3 пункта)

### 👥 Участники и роли
- Для каждого спикера (SPEAKER_XX) определи роль на основе контекста:
  * "Менеджер Соммерс" — если обсуждает задачи, сроки, координацию
  * "Разработчик Соммерс" — если говорит о коде, багах, реализации
  * "Представитель клиента" — если описывает проблемы, требования, ожидания
  * "Неизвестно" — если невозможно определить
- Приведи 1-2 ключевые реплики для каждого спикера

### ⚠️ Проблемы и решения
Для каждой проблемы:
- Название проблемы (кратко)
- Кто озвучил проблему (спикер + роль)
- Суть проблемы (1 предложение)
- Предложенные решения (если есть)
- Кто предложил решение
- Статус решения (реализовано/в работе/требует согласования)

### ✅ Дальнейшие действия
Таблица с ЧЁТКИМИ обязательствами:
| Действие | Ответственный (спикер + роль) | Срок | Статус |
|----------|-------------------------------|------|--------|
| Пример: Протестировать прошивку ANFU | SPEAKER_07 (Разработчик Соммерс) | до 30.09.2025 | Новое |

ВАЖНО:
1. Если в диалоге есть упоминания конкретных дат — используй их в колонке "Срок".
2. Если даты нет, но есть указание на срок ("сегодня", "на следующей неделе") — преобразуй в конкретную дату относительно текущей даты {datetime.now().strftime('%d.%m.%Y')}.
3. Фокусируйся на технических деталях: СБП, терминалы (Ingenico/Newland), прошивки, возвраты, стейт-холдеры, логи.
4. Не придумывай информацию, которой нет в диалоге.
5. Если информация неполная — помечай статус как "Требует уточнения".

Ответ должен быть строго в формате Markdown без дополнительных комментариев.
"""

    # Запрос к Ollama
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Диалог для анализа:\n{dialogue}"}
            ],
            options={
                "temperature": float(os.getenv("TEMPERATURE", 0.1)),
                "top_p": float(os.getenv("TOP_P", 0.9)),
                "repeat_penalty": float(os.getenv("REPEAT_PENALTY", 1.15)),
                "num_predict": 2000
            }
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"❌ Ошибка при запросе к Ollama: {e}")
        # Резервный анализ (упрощённый)
        return f"⚠️ **Ошибка анализа**\n\nНе удалось проанализировать диалог из-за ошибки:\n`{str(e)}`\n\n**Сырые данные диалога:**\n{dialogue[:1000]}..."

# === 3. Сохранение результатов ===
def save_results(filename: str, segments: list, analysis_md: str):
    """Сохранение результатов в JSON и Markdown."""
    base_name = Path(filename).stem
    
    # Сохранение сегментов в JSON
    json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "segments": segments
        }, f, ensure_ascii=False, indent=2)
    
    # Сохранение анализа в Markdown
    md_path = os.path.join(OUTPUT_DIR, f"{base_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(analysis_md)
    
    print(f"✅ Результаты сохранены:\n   - JSON: {json_path}\n   - Отчёт: {md_path}")
    return md_path

# === Основная функция ===
def main():
    parser = argparse.ArgumentParser(description="Pipeline для автоматического анализа встреч")
    parser.add_argument("audio_file", help="Путь к аудиофайлу (.webm, .mp3, .wav и др.)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Устройство для обработки")
    args = parser.parse_args()
    
    # Проверка устройства
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA недоступна, переключаюсь на CPU")
        args.device = "cpu"
    
    print(f"\n🚀 Запуск pipeline для файла: {args.audio_file}")
    print(f"⚙️ Устройство: {args.device}")
    print(f"🧠 Модель анализа: {MODEL_NAME}\n")
    
    try:
        # 1. Транскрибация и диаризация
        segments = transcribe_and_diarize(args.audio_file, device=args.device)
        
        if not segments:
            print("❌ Не удалось получить сегменты речи. Проверьте аудиофайл.")
            sys.exit(1)
        
        # 2. Анализ через Ollama
        analysis_md = analyze_with_ollama(segments)
        
        # 3. Сохранение результатов
        md_path = save_results(os.path.basename(args.audio_file), segments, analysis_md)
        
        print(f"\n🎉 Анализ завершён! Полный отчёт:\n{md_path}")
        
        # Вывод краткого содержания в консоль
        summary_start = analysis_md.find("### 📝 Краткое содержание")
        if summary_start != -1:
            summary_end = analysis_md.find("###", summary_start + 1)
            print("\n📋 КРАТКОЕ СОДЕРЖАНИЕ:")
            print(analysis_md[summary_start:summary_end].strip())
        
    except KeyboardInterrupt:
        print("\n🛑 Обработка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()