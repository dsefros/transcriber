# Transcriber Pipeline

Локальный оффлайн-пайплайн для транскрибации и анализа технических встреч.

## Структура проекта
transcriber/
├── src/ # Исходный код
│ ├── pipeline/ # Основной пайплайн обработки
│ ├── storage/ # Работа с БД (Postgres, Qdrant)
│ ├── utils/ # Вспомогательные утилиты
│ └── models/ # Pydantic-модели данных
├── config/ # Конфигурация (.yaml)
├── data/ # Данные (не в гите!)
│ ├── raw/ # Исходные аудиофайлы
│ └── processed/ # Результаты обработки
├── docs/ # Документация
│ └── terms/ # Шаблоны технических терминов
└── .env.example # Пример переменных окружения

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
