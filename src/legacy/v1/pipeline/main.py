#!/usr/bin/env python3
"""
Единый legacy-pipeline для обработки встреч.

ВАЖНО:
- WhisperX вызывается ТОЛЬКО если НЕ передан precomputed_segments_path
- sys.exit() ЗАПРЕЩЁН — только исключения
"""

import os
import json
import gc
import sys
import hashlib
import traceback
from pathlib import Path
from datetime import datetime

import torch
import ollama
from dotenv import load_dotenv

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

INPUT_DIR = os.getenv("INPUT_DIR", "input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
HF_TOKEN = os.getenv("HF_TOKEN")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================================================
# STORAGE / AI IMPORTS
# =========================================================

from src.legacy.v1.storage.postgres import (
    init_db,
    get_db_session,
    Meeting,
    Speaker,
    Fragment,
)
from src.legacy.v1.storage.qdrant import (
    init_qdrant_client,
    create_collections_if_not_exists,
)

from src.legacy.v1.config.models import get_models_config

from src.legacy.v1.ai.generator import generate_text

# =========================================================
# UTILS
# =========================================================

def _free_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# =========================================================
# LOAD SEGMENTS FROM JSON
# =========================================================

def load_segments_from_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "transcription" not in data:
        raise ValueError("JSON не содержит transcription")

    segments = [
        {
            "speaker": seg["speaker"],
            "text": seg["text"],
            "start": seg["start"],
            "end": seg["end"],
        }
        for seg in data["transcription"]
    ]

    meta = data.get("metadata", {})
    filename = meta.get("filename", Path(json_path).stem)
    audio_hash = meta.get("audio_hash", "")
    duration = meta.get(
        "duration_sec",
        sum(seg["end"] - seg["start"] for seg in segments),
    )

    return segments, filename, audio_hash, duration


# =========================================================
# WHISPERX (ЕДИНСТВЕННОЕ МЕСТО)
# =========================================================

def transcribe_and_diarize(audio_path: str, device: str = "cuda"):
    import whisperx
    from pydub import AudioSegment

    print(f"[DEBUG] WhisperX load (device={device})")

    model = whisperx.load_model(
        "large-v3",
        device,
        compute_type="float16" if device == "cuda" else "int8",
    )

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)

    wav_path = "temp_audio.wav"
    audio.export(wav_path, format="wav")

    audio_data = whisperx.load_audio(wav_path)
    result = model.transcribe(audio_data, language="ru")

    model_a, metadata = whisperx.load_align_model(
        language_code="ru",
        device=device,
        model_name="facebook/wav2vec2-base-960h",
    )

    result = whisperx.align(
        result["segments"], model_a, metadata, audio_data, device
    )

    del model_a
    _free_gpu_memory()

    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=HF_TOKEN,
        device=device,
    )
    diarized = diarize_model(audio_data)
    result = whisperx.assign_word_speakers(diarized, result)

    segments = [
        {
            "speaker": seg.get("speaker", "SPEAKER_00"),
            "text": seg.get("text", "").strip(),
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
        }
        for seg in result["segments"]
    ]

    del model, diarize_model, audio_data, diarized, result
    _free_gpu_memory()

    if os.path.exists(wav_path):
        os.remove(wav_path)

    print(f"[DEBUG] WhisperX done: {len(segments)} segments")
    return segments


# =========================================================
# ANALYSIS
# =========================================================

def analyze_with_model(segments: list) -> str:
    cfg = get_models_config()
    profile = cfg.get_active_profile()

    dialogue = "\n".join(
        f"{s['speaker']}: {s['text']}" for s in segments if s["text"]
    )

    system_prompt = f"""
Ты — эксперт по анализу технических встреч компании Соммерс.
Подготовь СТРУКТУРИРОВАННЫЙ Markdown-отчёт.

Дата: {datetime.now().strftime('%d.%m.%Y')}
"""

    if profile.backend == "ollama":
        response = ollama.chat(
            model=profile.name,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": dialogue},
            ],
            stream=False,
        )
        return response["message"]["content"].strip()

    return generate_text(
        f"{system_prompt}\n\n{dialogue}",
        profile,
    )


# =========================================================
# SAVE FILES ONLY
# =========================================================

def save_to_file_only(
    filename: str,
    segments: list,
    analysis_md: str,
    audio_hash: str,
    duration: float,
):
    base = Path(filename).stem
    out = Path(OUTPUT_DIR)

    md_path = out / f"{base}.md"
    json_path = out / f"{base}.json"

    md_path.write_text(analysis_md, encoding="utf-8")

    json.dump(
        {
            "metadata": {
                "filename": filename,
                "audio_hash": audio_hash,
                "duration_sec": duration,
                "processed_at": datetime.now().isoformat(),
            },
            "transcription": segments,
            "analysis": {"raw_markdown": analysis_md},
        },
        open(json_path, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    return str(md_path)


# =========================================================
# SAVE TO DATABASES
# =========================================================

def save_to_databases(session, qdrant_client, filename, segments, analysis_md, original_audio_path):
    """Идемпотентное сохранение результатов в Postgres и файлы"""
    print("💾 Сохранение данных в базы...")

    try:
        # --- hash ---
        with open(original_audio_path, "rb") as f:
            audio_hash = hashlib.sha256(f.read()).hexdigest()

        # --- ищем существующую встречу ---
        meeting = (
            session.query(Meeting)
            .filter(Meeting.filename == filename)
            .one_or_none()
        )

        if meeting:
            print(f"♻️ Встреча уже существует (id={meeting.id}), обновляем")
            meeting.updated_at = datetime.now()
            meeting.status = "completed"

            # чистим старые фрагменты
            session.query(Fragment).filter(
                Fragment.meeting_id == meeting.id
            ).delete()

        else:
            print("🆕 Создаём новую встречу")
            meeting = Meeting(
                filename=filename,
                start_time=datetime.now(),
                duration_sec=0,
                audio_hash=audio_hash,
                status="completed",
                quality_score=0.95,
                context_tags="[]",
            )
            session.add(meeting)
            session.flush()

        # --- спикеры ---
        speaker_cache = {}
        for seg in segments:
            name = seg.get("speaker", "SPEAKER_00")
            if name not in speaker_cache:
                speaker = (
                    session.query(Speaker)
                    .filter_by(external_id=name)
                    .one_or_none()
                )
                if not speaker:
                    speaker = Speaker(
                        external_id=name,
                        name=name,
                        role="unknown",
                    )
                    session.add(speaker)
                    session.flush()
                speaker_cache[name] = speaker.id

        # --- фрагменты ---
        duration = 0.0
        for i, seg in enumerate(segments):
            duration += seg["end"] - seg["start"]
            fragment = Fragment(
                meeting_id=meeting.id,
                start_time=seg["start"],
                end_time=seg["end"],
                speaker_id=speaker_cache[seg["speaker"]],
                text=seg["text"],
                raw_text=seg["text"],
                importance_score=0.8,
                business_value="discussion",
                technical_terms=json.dumps(
                    extract_technical_terms(seg["text"])
                ),
                semantic_cluster=i // 5,
            )
            session.add(fragment)

        meeting.duration_sec = duration
        session.commit()

        print(f"✅ Сохранено в Postgres (meeting_id={meeting.id})")

        # --- файлы ---
        base = Path(filename).stem
        out = Path(OUTPUT_DIR)
        out.mkdir(exist_ok=True)

        md_path = out / f"{base}.md"
        md_path.write_text(analysis_md, encoding="utf-8")

        json_path = out / f"{base}.json"
        json_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "filename": filename,
                        "audio_hash": audio_hash,
                        "meeting_id": meeting.id,
                        "duration_sec": duration,
                    },
                    "transcription": segments,
                    "analysis": analysis_md,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return str(md_path)

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()



# =========================================================
# MAIN
# =========================================================

def main(
    audio_file: str = None,
    json_file: str = None,
    device: str = "cuda",
    no_db: bool = False,
    precomputed_segments_path: str | None = None,
):
    if not audio_file and not json_file:
        raise ValueError("Требуется аудиофайл ИЛИ json_file")

    if audio_file and json_file:
        raise ValueError("Укажите ТОЛЬКО один источник")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    is_reanalyze_mode = json_file is not None
    print(f"\n🚀 Режим: {'Переанализ JSON' if is_reanalyze_mode else 'Полная обработка'}")

    try:
        # --- JSON ---
        if is_reanalyze_mode:
            segments, orig_filename, audio_hash, duration = load_segments_from_json(json_file)
            session = None
            qdrant_client = None

        # --- AUDIO ---
        else:
            if not precomputed_segments_path:
                raise RuntimeError(
                    "Whisper запрещён в legacy pipeline. "
                    "Используй TranscriptionStep."
                )

            print(f"[DEBUG] Используем предрасчитанные сегменты: {precomputed_segments_path}")
            with open(precomputed_segments_path, "r", encoding="utf-8") as f:
                segments = json.load(f)

            orig_filename = os.path.basename(audio_file)
            duration = sum(seg.get("end", 0) - seg.get("start", 0) for seg in segments)

            with open(audio_file, "rb") as f:
                audio_hash = hashlib.sha256(f.read()).hexdigest()

            if not no_db:
                engine = init_db()
                qdrant_client = init_qdrant_client()
                create_collections_if_not_exists(qdrant_client)
                session = get_db_session(engine)
            else:
                session = None
                qdrant_client = None

        _free_gpu_memory()
        analysis_md = analyze_with_model(segments)

        if is_reanalyze_mode or no_db:
            md_path = save_to_file_only(
                orig_filename,
                segments,
                analysis_md,
                audio_hash,
                duration,
            )
        else:
            md_path = save_to_databases(
                session,
                qdrant_client,
                orig_filename,
                segments,
                analysis_md,
                audio_file,
            )

        print(f"\n🎉 Анализ завершён:\n{md_path}")

    except Exception:
        traceback.print_exc()
        raise

