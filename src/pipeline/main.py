#!/usr/bin/env python3
"""
Единый pipeline для автоматической обработки встреч:
1. Транскрибация + диаризация через WhisperX
2. Анализ через локальную LLM (Ollama)
3. Генерация структурированного отчёта в Markdown
4. Сохранение данных в Postgres и Qdrant

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
import hashlib
import traceback

# === Загрузка конфигурации ===
load_dotenv()
INPUT_DIR = os.getenv("INPUT_DIR", "input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
MODEL_NAME = os.getenv("MODEL_NAME", "phi3:medium-128k")
HF_TOKEN = os.getenv("HF_TOKEN")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Импорт модулей баз данных ===
from src.storage.postgres import init_db, get_db_session, Meeting, Speaker, Fragment
from src.storage.qdrant import init_qdrant_client, create_collections_if_not_exists
import hashlib
import json

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

# === 3. Извлечение технических терминов ===
def extract_technical_terms(text: str) -> list:
    """Извлечение технических терминов из текста"""
    technical_terms = [
        'TSP', 'ПИоТ', 'ЕГАИС', 'маркировка', 'офлайн', 'Честный ЗНАК',
        'СБП', 'эквайринг', 'POS', 'терминал', 'прошивка', 'возвраты',
        'стейт-холдеры', 'логи', 'API', 'интеграция', 'безопасность',
        'Sommers', 'Соммерс', 'контрагент', 'клиент', 'продукт', 'тариф',
        'UTM', 'транзакция', 'авторизация', 'отказ', 'комиссия'
    ]
    
    found_terms = []
    text_lower = text.lower()
    
    for term in technical_terms:
        if term.lower() in text_lower:
            found_terms.append(term)
    
    return found_terms

# === 4. Сохранение в базы данных ===
def save_to_databases(session, qdrant_client, filename, segments, analysis_md, original_audio_path):
    """Сохранение результатов в Postgres и Qdrant"""
    print("💾 Сохранение данных в базы...")
    
    try:
        # Вычисление хеша аудиофайла
        with open(original_audio_path, 'rb') as f:
            audio_content = f.read()
            audio_hash = hashlib.sha256(audio_content).hexdigest()
        
        # 1. Создание записи о встрече
        meeting = Meeting(
            filename=filename,
            start_time=datetime.now(),
            duration_sec=sum(seg.get('end', 0) - seg.get('start', 0) for seg in segments),
            audio_hash=audio_hash,
            status='completed',
            quality_score=0.95,
            context_tags=json.dumps(["meeting", "transcription"])
        )
        session.add(meeting)
        session.flush()
        
        # 2. Обработка спикеров
        speaker_cache = {}
        
        for seg in segments:
            speaker_name = seg.get('speaker', 'SPEAKER_00')
            
            if speaker_name not in speaker_cache:
                # Поиск существующего спикера
                speaker = session.query(Speaker).filter_by(external_id=speaker_name).first()
                if not speaker:
                    # Создание нового спикера
                    speaker = Speaker(
                        external_id=speaker_name,
                        name=speaker_name,
                        role='неизвестно'
                    )
                    session.add(speaker)
                    session.flush()
                speaker_cache[speaker_name] = speaker.id
        
        # 3. Создание фрагментов
        for i, seg in enumerate(segments):
            speaker_name = seg.get('speaker', 'SPEAKER_00')
            speaker_id = speaker_cache[speaker_name]
            
            # Извлечение технических терминов
            technical_terms = extract_technical_terms(seg.get('text', ''))
            
            fragment = Fragment(
                meeting_id=meeting.id,
                start_time=seg.get('start', 0),
                end_time=seg.get('end', 0),
                speaker_id=speaker_id,
                text=seg.get('text', '').strip(),
                raw_text=seg.get('text', ''),
                importance_score=0.8,
                business_value='обсуждение',
                technical_terms=json.dumps(technical_terms),
                semantic_cluster=i // 5  # Простая кластеризация по 5 фрагментов
            )
            session.add(fragment)
            session.flush()
        
        # 4. Коммит изменений
        session.commit()
        print(f"✅ Данные успешно сохранены в Postgres! Встреча ID: {meeting.id}")
        
        # 5. Сохранение отчёта в файл (для обратной совместимости)
        base_name = Path(filename).stem
        md_path = os.path.join(OUTPUT_DIR, f"{base_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(analysis_md)
        
        return md_path
        
    except Exception as e:
        session.rollback()
        print(f"❌ Ошибка при сохранении в базы: {e}")
        traceback.print_exc()
        raise
    finally:
        session.close()

# === Основная функция ===
def main(audio_file: str, device: str = "cuda"):
    # Проверка существования файла
    if not os.path.exists(audio_file):
        print(f"❌ Файл не найден: {audio_file}")
        sys.exit(1)

    # Проверка устройства
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA недоступна, переключаюсь на CPU")
        device = "cpu"

    print(f"\n🚀 Запуск pipeline для файла: {audio_file}")
    print(f"⚙️ Устройство: {device}")
    print(f"🧠 Модель анализа: {MODEL_NAME}\n")

    try:
        # Инициализация баз данных
        print("🔧 Инициализация баз данных...")
        engine = init_db()
        qdrant_client = init_qdrant_client()
        create_collections_if_not_exists(qdrant_client)

        session = get_db_session(engine)

        # 1. Транскрибация и диаризация
        segments = transcribe_and_diarize(audio_file, device=device)

        if not segments:
            print("❌ Не удалось получить сегменты речи. Проверьте аудиофайл.")
            sys.exit(1)

        print(f"✅ Получено {len(segments)} сегментов")

        # 2. Анализ через Ollama
        analysis_md = analyze_with_ollama(segments)

        # 3. Сохранение в базы данных
        md_path = save_to_databases(session, qdrant_client, os.path.basename(audio_file), segments, analysis_md, audio_file)

        print(f"\n🎉 Анализ завершён! Полный отчёт:\n{md_path}")

        # Вывод краткого содержания в консоль
        summary_start = analysis_md.find("### 📝 Краткое содержание")
        if summary_start != -1:
            summary_end = analysis_md.find("###", summary_start + 1)
            if summary_end == -1:
                summary_end = len(analysis_md)
            print("\n📋 КРАТКОЕ СОДЕРЖАНИЕ:")
            print(analysis_md[summary_start:summary_end].strip())

    except KeyboardInterrupt:
        print("\n🛑 Обработка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)