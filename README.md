# Transcriber Pipeline

Локальный оффлайн-пайплайн для транскрибации и анализа технических встреч.

## Быстрый старт

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Настройка переменных окружения
cp .env.example .env
nano .env  # заполните HF_TOKEN, DATABASE_URL и др.

# 3. Запуск обработки
python src/cli.py data/raw/ваш_файл.webm --device cuda
Требования
Python 3.10+
CUDA 11.8+ (для GPU)
Ollama (локальная LLM)
Postgres + Qdrant (локально)
