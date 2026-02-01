#!/usr/bin/env python3
"""
Единый pipeline для автоматической обработки встреч:
1. Транскрибация + диаризация через WhisperX (если указан аудиофайл)
2. Анализ через локальную LLM (Ollama/llama-cpp)
3. Генерация структурированного отчёта в Markdown
4. Сохранение данных в Postgres и Qdrant (только при обработке аудио)

Поддерживает offline-режим без интернета.
"""
import os
import sys
import json
import gc
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
HF_TOKEN = os.getenv("HF_TOKEN")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Импорт модулей баз данных ===
from src.storage.postgres import init_db, get_db_session, Meeting, Speaker, Fragment
from src.storage.qdrant import init_qdrant_client, create_collections_if_not_exists

# === Импорт нового интерфейса генерации ===
from src.config.models import get_models_config
from src.ai.generator import generate_text

def _free_gpu_memory():
    """Принудительное освобождение всей доступной памяти после этапа обработки"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    # Дополнительная очистка кэша PyTorch
    if hasattr(torch.cuda, 'cudnn'):
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        print(f"[DEBUG] VRAM после очистки: {torch.cuda.memory_allocated() / 1024**2:.1f} / {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f} MB")

# === НОВАЯ ФУНКЦИЯ: загрузка сегментов из JSON ===
def load_segments_from_json(json_path: str) -> tuple[list, str, str, float]:
    """Загружает сегменты из существующего JSON.
    Возвращает: (сегменты, исходное имя файла, аудиохеш, длительность)"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'transcription' not in data:
        raise ValueError(f"JSON не содержит поля 'transcription': {json_path}")
    
    segments = [
        {
            'speaker': seg['speaker'],
            'text': seg['text'],
            'start': seg['start'],
            'end': seg['end']
        }
        for seg in data['transcription']
    ]
    
    metadata = data.get('metadata', {})
    orig_filename = metadata.get('filename', Path(json_path).stem)
    audio_hash = metadata.get('audio_hash', hashlib.sha256(Path(json_path).name.encode()).hexdigest()[:16])
    duration = metadata.get('duration_sec', sum(seg['end'] - seg['start'] for seg in segments))
    
    return segments, orig_filename, audio_hash, duration

# === 1. Транскрибация и диаризация (на основе transcribe_v3.py) ===
def transcribe_and_diarize(audio_path: str, device: str = "cuda"):
    """Транскрибация + диаризация через WhisperX с гарантированным освобождением памяти."""
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
    
    # Выравнивание — загружаем ТОЛЬКО если нужно
    model_a, metadata = whisperx.load_align_model(
        language_code="ru",
        device=device,
        model_name="facebook/wav2vec2-base-960h"
    )
    result = whisperx.align(result["segments"], model_a, metadata, audio_data, device)
    
    # Освобождаем модель выравнивания СРАЗУ после использования
    del model_a
    _free_gpu_memory()
    
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
    
    # 🔥 КРИТИЧНО: освобождаем ВСЕ модели и данные этапа транскрибации
    del model
    del diarize_model
    del audio_data
    del result
    del diarize_segments
    _free_gpu_memory()
    
    # Очистка временных файлов
    if os.path.exists(wav_path):
        os.remove(wav_path)
    
    print(f"[DEBUG] WhisperX завершён. Сегментов: {len(segments)}")
    return segments

# === 2. Анализ через Ollama (локальная LLM) ===
def analyze_with_model(segments: list) -> str:
    """Анализ диалога через универсальный интерфейс (Ollama или llama-cpp)."""
    # Загрузка активного профиля модели
    models_cfg = get_models_config()
    profile = models_cfg.get_active_profile()
    print(f"[DEBUG] Анализ через {profile.backend} ({profile.key})...")
    
    # Формирование диалога для промпта
    dialogue = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in segments if seg["text"]])
    
    # Системный промпт
    system_prompt = f"""
Ты — эксперт по анализу технических встреч в компании Соммерс. Соммерс специализируется на IT-решениях для эквайринга, POS-терминалов и автоматизации бизнеса. Твои клиенты — банки, ритейл и частные мерчанты.
Проанализируй диалог и подготовь СТРУКТУРИРОВАННЫЙ ОТЧЁТ в формате Markdown со следующими разделами:
### 📝 Краткое содержание
- Общая тема встречи (1-2 предложения)
### ⚠️ Проблемы и решения
Для каждой темы обсуждения:
- Название проблемы (кратко)
- Кто озвучил проблему (спикер)
- Суть проблемы
- Предложенные решения (если есть)
- Кто предложил решение (спикер)
- Статус решения (реализовано/в работе/требует согласования)
ВАЖНО:
1. Если в диалоге есть упоминания конкретных дат — используй их.
2. Если даты нет, но есть указание на срок ("сегодня", "на следующей неделе") — преобразуй в конкретную дату относительно текущей даты {datetime.now().strftime('%d.%m.%Y')}.
3. Фокусируйся на технических деталях: СБП, терминалы (Ingenico/Newland), прошивки, возвраты, стейт-холдеры, логи.
4. Не придумывай информацию, которой нет в диалоге.
5. Если информация неполная — помечай статус как "Требует уточнения".
Ответ должен быть строго в формате Markdown без дополнительных комментариев.
"""
    
    # Генерация
    try:
        if profile.backend == "ollama":
            # Для Ollama используем чат-интерфейс с системной ролью
            response = ollama.chat(
                model=profile.name,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": f"Диалог для анализа:\n{dialogue}"}
                ],
                options={
                    "temperature": profile.params.get("temperature", 0.1),
                    "top_p": profile.params.get("top_p", 0.9),
                    "repeat_penalty": profile.params.get("repeat_penalty", 1.15),
                    "num_predict": profile.params.get("num_predict", 2000),
                },
                stream=False
            )
            return response["message"]["content"].strip()
        else:
            # Для llama-cpp используем единый интерфейс generate_text()
            full_prompt = f"Ты — эксперт по анализу технических встреч в компании Соммерс.\n{system_prompt.strip()}\nДиалог для анализа:\n{dialogue}"
            return generate_text(full_prompt, profile)
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        return f"⚠️ **Ошибка анализа**\nНе удалось проанализировать диалог из-за ошибки:\n`{str(e)}`\n\n**Сырые данные диалога:**\n{dialogue[:1000]}..."

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

# === 4. Сохранение ТОЛЬКО в файлы (без БД) ===
def save_to_file_only(filename: str, segments: list, analysis_md: str, audio_hash: str, duration_sec: float) -> str:
    """
    Сохранение результатов ТОЛЬКО в файлы (без БД).
    Принимает хеш и длительность напрямую — не требует доступа к аудиофайлу.
    Возвращает путь к основному Markdown-файлу.
    """
    base_name = Path(filename).stem
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # 1. Сохранение Markdown-отчёта
    md_path = output_path / f"{base_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(analysis_md)
    print(f"📄 Markdown-отчёт сохранён: {md_path}")
    
    # 2. Формирование полного JSON-результата
    json_result = {
        "metadata": {
            "filename": filename,
            "audio_hash": audio_hash,
            "processed_at": datetime.now().isoformat(),
            "duration_sec": duration_sec,
            "segment_count": len(segments),
            "reanalyzed_at": datetime.now().isoformat()  # маркер переанализа
        },
        "transcription": [
            {
                "speaker": seg.get('speaker', 'SPEAKER_00'),
                "text": seg.get('text', '').strip(),
                "start": seg.get('start', 0),
                "end": seg.get('end', 0),
                "technical_terms": extract_technical_terms(seg.get('text', ''))
            }
            for seg in segments
        ],
        "analysis": {
            "raw_markdown": analysis_md,
            "extracted_terms": list(set(term for seg in segments for term in extract_technical_terms(seg.get('text', ''))))
        }
    }
    
    # 3. Сохранение JSON (перезапись существующего файла)
    json_path = output_path / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)
    print(f"📦 JSON обновлён: {json_path}")
    
    return str(md_path)

# === 5. Сохранение в базы данных (только при обработке аудио) ===
def save_to_databases(session, qdrant_client, filename, segments, analysis_md, original_audio_path):
    """Сохранение результатов в Postgres и Qdrant + всегда сохранять файлы на диск"""
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
        
        # === ВСЕГДА сохранять файлы на диск (единая логика с save_to_file_only) ===
        base_name = Path(filename).stem
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        
        # 5. Сохранение Markdown-отчёта
        md_path = output_path / f"{base_name}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(analysis_md)
        print(f"📄 Markdown-отчёт сохранён: {md_path}")
        
        # 6. Сохранение JSON с полной структурой (включая ссылку на запись в БД)
        json_result = {
            "metadata": {
                "filename": filename,
                "audio_hash": audio_hash,
                "processed_at": datetime.now().isoformat(),
                "duration_sec": meeting.duration_sec,
                "segment_count": len(segments),
                "meeting_id": meeting.id  # ← критично: связь с записью в БД
            },
            "transcription": [
                {
                    "speaker": seg.get('speaker', 'SPEAKER_00'),
                    "text": seg.get('text', '').strip(),
                    "start": seg.get('start', 0),
                    "end": seg.get('end', 0),
                    "technical_terms": extract_technical_terms(seg.get('text', ''))
                }
                for seg in segments
            ],
            "analysis": {
                "raw_markdown": analysis_md,
                "extracted_terms": list(set(term for seg in segments for term in extract_technical_terms(seg.get('text', ''))))
            }
        }
        json_path = output_path / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        print(f"📦 JSON-данные сохранены: {json_path}")
        
        return str(md_path)
    except Exception as e:
        session.rollback()
        print(f"❌ Ошибка при сохранении в базы: {e}")
        traceback.print_exc()
        raise
    finally:
        session.close()

# === Основная функция ===
def main(audio_file: str = None, json_file: str = None, device: str = "cuda", no_db: bool = False):
    # Валидация источника (уже выполнена в CLI, но дублируем для защиты)
    if not audio_file and not json_file:
        raise ValueError("Требуется аудиофайл ИЛИ json_file")
    if audio_file and json_file:
        raise ValueError("Укажите ТОЛЬКО один источник")
    
    # Проверка устройства
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA недоступна, переключаюсь на CPU")
        device = "cpu"
    
    # Определение режима
    is_reanalyze_mode = json_file is not None
    print(f"\n🚀 Режим: {'Переанализ JSON' if is_reanalyze_mode else 'Полная обработка'}")
    if is_reanalyze_mode:
        print("⏭️  Запись в БД недоступна в режиме --json")
    
    try:
        # === Режим 1: Переанализ из JSON ===
        if is_reanalyze_mode:
            if not os.path.exists(json_file):
                print(f"❌ JSON не найден: {json_file}")
                sys.exit(1)
            
            segments, orig_filename, audio_hash, duration = load_segments_from_json(json_file)
            print(f"✅ Загружено {len(segments)} сегментов из {orig_filename}")
            
            # В режиме переанализа БД не инициализируется — нет связи с исходным аудио
            session = qdrant_client = None
        
        # === Режим 2: Полная обработка аудио ===
        else:
            if not os.path.exists(audio_file):
                print(f"❌ Файл не найден: {audio_file}")
                sys.exit(1)
            
            # Инициализация БД (если разрешена)
            if not no_db:
                print("🔧 Инициализация баз данных...")
                engine = init_db()
                qdrant_client = init_qdrant_client()
                create_collections_if_not_exists(qdrant_client)
                session = get_db_session(engine)
            else:
                print("⏭️  Пропуск инициализации баз данных (режим --no-db)")
                session = qdrant_client = None
            
            # Транскрибация
            segments = transcribe_and_diarize(audio_file, device=device)
            orig_filename = os.path.basename(audio_file)
            duration = sum(seg.get('end', 0) - seg.get('start', 0) for seg in segments)
            
            # Вычисление хеша (только для аудио)
            with open(audio_file, 'rb') as f:
                audio_hash = hashlib.sha256(f.read()).hexdigest()
        
        # === Общий этап: анализ через LLM ===
        print("[DEBUG] Очистка памяти перед анализом через LLM...")
        _free_gpu_memory()
        analysis_md = analyze_with_model(segments)
        
        # === Сохранение результатов ===
        if is_reanalyze_mode or no_db:
            # Режим без БД: только файлы
            md_path = save_to_file_only(orig_filename, segments, analysis_md, audio_hash, duration)
        else:
            # Полный режим: БД + файлы
            md_path = save_to_databases(session, qdrant_client, orig_filename, segments, analysis_md, audio_file)
        
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