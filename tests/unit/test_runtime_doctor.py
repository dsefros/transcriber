from pathlib import Path
import importlib.metadata
import json
from unittest.mock import patch

import pytest

from src.app.runtime_doctor import CheckResult, _gpu_status, collect_runtime_report, main

pytestmark = pytest.mark.unit


def test_runtime_doctor_reports_missing_required_prerequisites(models_config_factory, runtime_env, monkeypatch):
    config_path = models_config_factory()
    runtime_env(HF_TOKEN='token')
    monkeypatch.chdir(config_path.parent)

    versions = {
        'PyYAML': '6.0.2',
        'SQLAlchemy': '2.0.29',
        'python-dotenv': '1.0.1',
        'psutil': '5.9.8',
        'ollama': '0.4.7',
        'torch': '2.3.0+cu121',
        'torchaudio': '2.3.0+cu121',
        'whisperx': '3.3.1',
        'pyannote.audio': '3.4.0',
        'pydub': '0.25.1',
        'transformers': '4.40.0',
        'huggingface-hub': '0.23.0',
    }

    def fake_version(name: str):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError
        return versions[name]

    with patch('src.app.runtime_doctor.metadata.version', side_effect=fake_version), \
         patch('src.app.runtime_doctor._database_status', return_value=CheckResult('database', 'ok', 'configured')), \
         patch('src.app.runtime_doctor._gpu_status', return_value=CheckResult('gpu', 'ok', 'cuda ok')):
        report = collect_runtime_report(models_path=str(config_path))

    assert report['summary']['fail'] == 0
    assert report['models']['active_profile'] == 'primary'
    assert any(check['name'] == 'whisperx' and check['status'] == 'ok' for check in report['checks'])


def test_runtime_doctor_warns_for_missing_local_llama_model(models_config_factory, runtime_env, monkeypatch):
    config_path = models_config_factory(
        profiles={
            'primary': {
                'backend': 'llama_cpp',
                'path': '/tmp/missing-model.gguf',
                'description': 'Local model',
            }
        }
    )
    runtime_env()
    monkeypatch.chdir(config_path.parent)

    with patch('src.app.runtime_doctor._database_status', return_value=CheckResult('database', 'ok', 'configured')), \
         patch('src.app.runtime_doctor._gpu_status', return_value=CheckResult('gpu', 'warn', 'no cuda')):
        report = collect_runtime_report(models_path=str(config_path))

    model_check = next(check for check in report['checks'] if check['name'] == 'models.yaml')
    assert model_check['status'] == 'warn'
    assert 'configured GGUF path is missing' in model_check['detail']


def test_runtime_doctor_main_supports_json_output(models_config_factory, runtime_env, monkeypatch, capsys):
    config_path = models_config_factory()
    runtime_env(HF_TOKEN='token')
    monkeypatch.chdir(config_path.parent)

    with patch('src.app.runtime_doctor.collect_runtime_report', return_value={'platform': {'python': '3.10.0', 'system': 'Linux', 'machine': 'x86_64'}, 'supported_baseline': {'python': ['3.10', '3.11'], 'gpu': 'gpu', 'database': 'db', 'llm_backends': ['ollama']}, 'models': {'active_profile': 'primary', 'backend': 'ollama'}, 'checks': [], 'summary': {'ok': 1, 'warn': 0, 'fail': 0}}):
        exit_code = main(['--models-config', str(config_path), '--json'])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output['models']['active_profile'] == 'primary'


def test_runtime_doctor_gpu_status_handles_broken_torch_import(monkeypatch):
    with patch('src.app.runtime_doctor.metadata.version', return_value='2.3.0+cu121'), \
         patch('importlib.import_module', side_effect=RuntimeError('broken torch import')):
        result = _gpu_status()

    assert result.status == 'fail'
    assert 'could not be imported cleanly' in result.detail


def test_runtime_doctor_reports_transcription_env_defaults(models_config_factory, runtime_env, monkeypatch):
    config_path = models_config_factory()
    runtime_env()
    monkeypatch.chdir(config_path.parent)
    for name in ('TRANSCRIPTION_MODEL_NAME', 'TRANSCRIPTION_DEVICE', 'ALIGNMENT_LANGUAGE_CODE', 'ALIGNMENT_MODEL_NAME'):
        monkeypatch.delenv(name, raising=False)

    with patch('src.app.runtime_doctor._database_status', return_value=CheckResult('database', 'ok', 'configured')), \
         patch('src.app.runtime_doctor._gpu_status', return_value=CheckResult('gpu', 'ok', 'cuda ok')):
        report = collect_runtime_report(models_path=str(config_path))

    assert report['supported_baseline']['transcription'] == {
        'model_name': 'large-v3',
        'device': 'cuda',
        'alignment_language_code': 'ru',
        'alignment_model_name': 'facebook/wav2vec2-base-960h',
    }
    assert any(check['name'] == 'TRANSCRIPTION_MODEL_NAME' and check['status'] == 'warn' for check in report['checks'])
    assert any(check['name'] == 'TRANSCRIPTION_DEVICE' and check['status'] == 'warn' for check in report['checks'])


def test_runtime_doctor_reports_transcription_env_overrides(models_config_factory, runtime_env, monkeypatch):
    config_path = models_config_factory()
    runtime_env(
        TRANSCRIPTION_MODEL_NAME='small',
        TRANSCRIPTION_DEVICE='cpu',
        ALIGNMENT_LANGUAGE_CODE='en',
        ALIGNMENT_MODEL_NAME='custom-aligner',
    )
    monkeypatch.chdir(config_path.parent)

    with patch('src.app.runtime_doctor._database_status', return_value=CheckResult('database', 'ok', 'configured')), \
         patch('src.app.runtime_doctor._gpu_status', return_value=CheckResult('gpu', 'warn', 'cpu only')):
        report = collect_runtime_report(models_path=str(config_path))

    assert report['supported_baseline']['transcription'] == {
        'model_name': 'small',
        'device': 'cpu',
        'alignment_language_code': 'en',
        'alignment_model_name': 'custom-aligner',
    }
    assert any(check['name'] == 'TRANSCRIPTION_MODEL_NAME' and check['status'] == 'ok' for check in report['checks'])
    assert any(check['name'] == 'TRANSCRIPTION_DEVICE' and check['status'] == 'ok' for check in report['checks'])


def test_runtime_doctor_does_not_warn_for_deprecated_llm_env_vars_when_absent(models_config_factory, runtime_env, monkeypatch):
    config_path = models_config_factory()
    runtime_env()
    monkeypatch.chdir(config_path.parent)
    for name in ('TEMPERATURE', 'NUM_CTX', 'NUM_PREDICT', 'TOP_P', 'REPEAT_PENALTY'):
        monkeypatch.delenv(name, raising=False)

    with patch('src.app.runtime_doctor._database_status', return_value=CheckResult('database', 'ok', 'configured')), \
         patch('src.app.runtime_doctor._gpu_status', return_value=CheckResult('gpu', 'ok', 'cuda ok')):
        report = collect_runtime_report(models_path=str(config_path))

    assert all(check['name'] not in {'TEMPERATURE', 'NUM_CTX', 'NUM_PREDICT', 'TOP_P', 'REPEAT_PENALTY'} for check in report['checks'])


def test_runtime_doctor_warns_for_deprecated_llm_env_vars_when_present(models_config_factory, runtime_env, monkeypatch):
    config_path = models_config_factory()
    runtime_env(TEMPERATURE='0.1', NUM_CTX='16384')
    monkeypatch.chdir(config_path.parent)

    with patch('src.app.runtime_doctor._database_status', return_value=CheckResult('database', 'ok', 'configured')), \
         patch('src.app.runtime_doctor._gpu_status', return_value=CheckResult('gpu', 'ok', 'cuda ok')):
        report = collect_runtime_report(models_path=str(config_path))

    warnings = {check['name']: check for check in report['checks'] if check['name'] in {'TEMPERATURE', 'NUM_CTX'}}
    assert warnings['TEMPERATURE']['status'] == 'warn'
    assert 'ignored by the canonical runtime' in warnings['TEMPERATURE']['detail']
    assert "'primary' profile params" in warnings['TEMPERATURE']['detail']
    assert warnings['NUM_CTX']['status'] == 'warn'


def test_runtime_doctor_deprecated_llm_env_warnings_do_not_change_active_profile(models_config_factory, runtime_env, monkeypatch):
    config_path = models_config_factory()
    runtime_env(ACTIVE_MODEL_PROFILE='primary', TOP_P='0.9')
    monkeypatch.chdir(config_path.parent)

    with patch('src.app.runtime_doctor._database_status', return_value=CheckResult('database', 'ok', 'configured')), \
         patch('src.app.runtime_doctor._gpu_status', return_value=CheckResult('gpu', 'ok', 'cuda ok')):
        report = collect_runtime_report(models_path=str(config_path))

    assert report['models']['active_profile'] == 'primary'
    top_p_warning = next(check for check in report['checks'] if check['name'] == 'TOP_P')
    assert top_p_warning['status'] == 'warn'


def test_operator_docs_and_env_example_reflect_models_yaml_policy():
    env_example = Path('.env.example').read_text()
    readme = Path('README.md').read_text()
    runtime_doc = Path('docs/runtime-environment.md').read_text()

    for legacy_name in ('TEMPERATURE', 'NUM_CTX', 'NUM_PREDICT', 'TOP_P', 'REPEAT_PENALTY'):
        assert f'\n{legacy_name}=' not in env_example

    assert 'models.yaml' in env_example
    assert '.env выбирает активный профиль' in env_example
    assert 'Генерационные параметры LLM' in env_example
    assert 'canonical operator-facing source of LLM backend and generation configuration' in readme
    assert 'ACTIVE_MODEL_PROFILE' in readme
    assert 'unsupported and ignored by the canonical runtime' in readme
    assert 'canonical operator-facing source of LLM backend and generation configuration' in runtime_doc
    assert '`ACTIVE_MODEL_PROFILE` only selects the active `models.yaml` profile' in runtime_doc
    assert 'unsupported and ignored by the canonical runtime' in runtime_doc
