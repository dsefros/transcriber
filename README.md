# Transcriber Pipeline

Локальный оффлайн-пайплайн для транскрибации и анализа технических встреч.

## Быстрый старт

```bash
# 1. Базовые зависимости для config/CLI/test collection
pip install -r requirements/runtime.txt
pip install -r requirements/ml.txt

# 2. Настройка переменных окружения
cp .env.example .env
nano .env  # заполните HF_TOKEN, DATABASE_URL и др.

# 3. Проверка supported environment
# runtime_doctor автоматически читает локальный .env, если файл существует
python -m src.app.runtime_doctor

# 4. Канонический запуск runtime
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

- `src.app.cli` → preflight checks → `src.worker` → `JobRunner` → `PipelineOrchestrator` → `TranscriptionStep` → `AnalysisStep`

CLI теперь выполняет лёгкий preflight до создания `Worker`: проверяет существование входного пути, загрузку канонического `models.yaml`, разрешение active profile, наличие обязательной зависимости для выбранного LLM backend, наличие ML-стека для аудио runtime и обязательный `DATABASE_URL`. Ошибки намеренно ранние и actionable, чтобы оператор не ждал подключения к БД или загрузки моделей перед очевидным отказом.

Persistence в active runtime описывается точнее так:

- job и job_steps state сохраняются в Postgres
- pipeline artifacts записываются в `output/`
- persistence не описывается как отдельная pipeline stage; это side effect шагов и orchestration runtime

## Legacy / Compatibility Modules

`src/legacy/v1/*` не является частью active runtime. Эти модули сохранены только для compatibility imports, migration support и отдельных manual workflows.

Новый runtime-код не должен импортировать модули из `src.legacy`. Legacy-дерево рассматривается как quarantine zone: его можно терпеть для обратной совместимости, но нельзя использовать как источник новой runtime-логики.

Совместимые shim-модули остаются намеренно:

- `src/infrastructure/llm/config.py` — compatibility shim с dict-shaped config поверх `src.config.models`
- `src/infrastructure/transcription/legacy_adapter.py` — минимальный compatibility facade для старого adapter import path
- `src/legacy/v1/*` — legacy import paths и compatibility-only/manual сценарии

Эта legacy surface остаётся только как quarantine boundary. Active runtime не должен расширять её или тянуть её обратно в canonical flow.

Что уже убрано из поддерживаемой surface:

- legacy snapshot-копии runtime helper-модулей больше не считаются поддерживаемыми import paths
- исторические одноразовые LLM-клиенты вне canonical/compatibility boundary удалены как dead legacy code
- compatibility facade для transcription adapter сохраняет только старый alias-символ и больше не выглядит как альтернативный canonical export

Что остаётся допустимым только как compatibility layer:

- `src/legacy/v1/pipeline/main.py` — ручной v1 workflow, не canonical entrypoint
- `src/legacy/v1/storage/*` — re-export canonical storage helpers для старых import paths
- `src/legacy/v1/config/models.py` и `src/infrastructure/llm/config.py` — thin wrappers над `src.config.models`

Если вы пишете новый код, используйте только canonical imports и entrypoints из active runtime.

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

- Supported Python baseline: 3.10 or 3.11
- Supported GPU baseline for canonical WhisperX runtime: NVIDIA GPU + CUDA 12.1 userspace (`torch==2.3.0+cu121`)
- Postgres обязателен для active runtime worker (`DATABASE_URL`)
- `HF_TOKEN` рекомендуется как часть supported baseline для pyannote diarization/download flows
- Ollama требуется только для профилей `backend=ollama`; локальный GGUF-файл требуется только для профилей `backend=llama_cpp`
- Qdrant может встречаться в legacy/planned/non-canonical контекстах, но не является обязательным для текущего active runtime path.


## Testing

Тесты теперь разделены на явные слои и по умолчанию запускаются в hermetic-режиме: `pytest` очищает большую часть process environment для каждого теста, поэтому локальные `.env`, `ACTIVE_MODEL_PROFILE`, `DATABASE_URL` и другие пользовательские export-переменные не должны влиять на результат без явного fixture/monkeypatch в самом тесте.

Слои тестов:

- `unit` — быстрые hermetic-тесты логики, конфигурации и import/preflight-контрактов
- `integration` — более широкие runtime/boundary-тесты без обращения к реальным внешним сервисам
- `smoke` — узкие сквозные проверки канонических pipeline boundary

Рекомендуемые команды:

```bash
# весь suite
python -m pytest

# быстрый локальный прогон по умолчанию
python -m pytest -m unit

# runtime/boundary слой отдельно
python -m pytest -m integration

# smoke-проверки канонического pipeline контракта
python -m pytest -m smoke
```

Тестам, которым нужен runtime env, следует задавать его явно через test fixtures/`monkeypatch`, а не через реальный `.env`. Если тесту нужен собственный `models.yaml` или временная рабочая директория, он должен создавать их локально через fixture в `tests/conftest.py`.

## Environment Expectations

- Лёгкие тесты и import/collection-проверки должны проходить без полного ML-стека. Для этого heavy imports в canonical runtime отложены до реального выполнения backend/runtime кода.
- Реальный audio runtime по-прежнему требует `torch`, `whisperx`, `pydub` и сопутствующий ML stack. Если этих пакетов нет, CLI завершится на preflight с подсказкой по установке.
- Active runtime по-прежнему требует `DATABASE_URL`, потому что `src.worker` сохраняет job state в Postgres до запуска pipeline.


## Runtime Environment Contract

Подробный поддерживаемый baseline, разделение dependency roles и использование runtime doctor (включая auto-load локального `.env`) описаны в `docs/runtime-environment.md`.
