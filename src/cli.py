#!/usr/bin/env python3
"""CLI для pipeline обработки встреч."""
import argparse
from pathlib import Path
from src.pipeline.main import main as run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Pipeline для автоматического анализа встреч")
    parser.add_argument("audio_file", type=Path, help="Путь к аудиофайлу")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    
    # Валидация файла
    if not args.audio_file.exists():
        raise FileNotFoundError(f"Файл не найден: {args.audio_file}")
    
    # Запуск основного пайплайна
    run_pipeline(str(args.audio_file), args.device)

if __name__ == "__main__":
    main()
