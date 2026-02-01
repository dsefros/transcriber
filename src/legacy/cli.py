#!/usr/bin/env python3
"""CLI для pipeline обработки встреч."""
import argparse
from pathlib import Path
from src.legacy.pipeline.main import main as run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Pipeline для автоматического анализа встреч")
    parser.add_argument("audio_file", nargs="?", type=Path, 
                        help="Путь к аудиофайлу (обязателен, если не указан --json)")
    parser.add_argument("--json", type=Path, 
                        help="Путь к JSON для переанализа (пропускает транскрибацию и запись в БД)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-db", action="store_true", 
                        help="Без записи в базы данных (только файлы). Игнорируется при --json")

    args = parser.parse_args()

    # Валидация: ровно один источник
    if not args.audio_file and not args.json:
        parser.error("Укажите аудиофайл ИЛИ --json")
    if args.audio_file and args.json:
        parser.error("Укажите ТОЛЬКО один источник: аудио ИЛИ JSON")

    # Автоматическое включение --no-db при --json (запись в БД недоступна)
    effective_no_db = True if args.json else args.no_db

    # Запуск пайплайна
    run_pipeline(
        audio_file=str(args.audio_file) if args.audio_file else None,
        json_file=str(args.json) if args.json else None,
        device=args.device,
        no_db=effective_no_db
    )

if __name__ == "__main__":
    main()