# Transcriber Pipeline

Локальный оффлайн-пайплайн для транскрибации и анализа технических встреч.

## Быстрый старт

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Настройка переменных окружения
cp .env.example .env
nano .env  # заполните HF_TOKEN, DATABASE_URL и др.

# 3. Канонический запуск runtime
PYTHONPATH=. python -m src.app.cli data/raw/ваш_файл.webm
```

## Runtime Architecture (Canonical)

Активный runtime запускается через CLI-модуль `src.app.cli`, который создает `Job`, инициализирует `Worker` из `src.worker` и передает задачу в активный job/pipeline runtime.

`src.worker` является composition root активного runtime: именно здесь связываются canonical job repository, transcription adapter, LLM adapter и application services. Runtime truth для wiring находится в этом модуле, а не в legacy-скриптах или старых entrypoint-файлах.

Инфраструктурные адаптеры образуют boundary между runtime и внешними системами:

- transcription boundary: `src/infrastructure/transcription/whisperx_adapter.py`
- LLM boundary: `src/infrastructure/llm/adapter.py`
- persistence and artifact helpers: `src/infrastructure/storage/*`

Конфигурация моделей для активного runtime загружается через `src.config.models`, который читает `models.yaml`, выбирает active profile и передает нормализованный конфиг в инфраструктурные адаптеры.

Точный active flow сейчас такой:

- `src.app.cli` → `src.worker` → `JobRunner` → `PipelineOrchestrator` → `TranscriptionStep` → `AnalysisStep`

Persistence в active runtime описывается точнее так:

- job и job_steps state сохраняются в Postgres
- pipeline artifacts записываются в `output/`
- persistence не описывается как отдельная pipeline stage; это side effect шагов и orchestration runtime

## Legacy / Compatibility Modules

`src/legacy/v1/*` не является частью active runtime. Эти модули сохранены только для compatibility imports, migration support и отдельных manual workflows.

Новый runtime-код не должен импортировать модули из `src.legacy`. Legacy-дерево рассматривается как quarantine zone: его можно терпеть для обратной совместимости, но нельзя использовать как источник новой runtime-логики.

Совместимые shim-модули остаются намеренно:

- `src/infrastructure/llm/config.py` — compatibility shim с dict-shaped config
- `src/infrastructure/transcription/legacy_adapter.py` — alias shim для migration path
- `src/legacy/v1/*` — legacy import paths и compatibility-only сценарии

## Config Truth

Канонический источник model-конфигурации — `models.yaml`.

Канонический загрузчик — `src.config.models`. Active runtime использует его для чтения профилей, валидации и выбора текущей модели.

Семантика выбора active profile:

- если установлен `ACTIVE_MODEL_PROFILE`, используется он
- иначе используется `default_model` из `models.yaml`
- если профиль не найден или не определен, canonical loader поднимает ошибку конфигурации

`src/infrastructure/llm/config.py` сохраняется только как compatibility-only shim для старых import paths. Новая runtime wiring и новая документация должны ссылаться на `src.config.models`.

## Entrypoint

Канонический способ запуска runtime:

```bash
PYTHONPATH=. python -m src.app.cli <audio>
```

Именно этот entrypoint должен использоваться для проверки активного runtime после migration. CLI-флаг `--json` существует, но это не канонический active path для обработки аудио и его не следует описывать как нормальный режим запуска runtime.

Старые скрипты, legacy pipeline main и исторические `src/cli.py`-style инструкции не являются canonical entrypoints и не должны использоваться как источник runtime truth.

## Contributor Rules

- не добавляйте новые импорты из `src.legacy` в active runtime
- не дублируйте model/config loaders
- adapters должны сохранять границы между runtime и внешними зависимостями
- runtime truth для wiring находится в `src.worker`

## Требования

- Python 3.10+
- CUDA 11.8+ (для GPU)
- Ollama (локальная LLM)
- Postgres (локально для active runtime)
- Qdrant может встречаться в legacy/planned/non-canonical контекстах, но не является обязательным для текущего active runtime path.
