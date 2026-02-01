"""
Загрузчик конфигурации моделей из models.yaml
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class ModelConfigError(Exception):
    """Ошибка конфигурации модели"""
    pass

class ModelProfile:
    """Профиль модели с валидацией"""
    def __init__(self, key: str, config: Dict[str, Any]):
        self.key = key
        self.backend = config.get('backend')
        self.name = config.get('name')
        self.path = config.get('path')
        self.description = config.get('description', '')
        self.params = config.get('params', {})
        
        # Валидация обязательных полей
        if not self.backend:
            raise ModelConfigError(f"Профиль '{key}': отсутствует поле 'backend'")
        
        if self.backend == 'ollama' and not self.name:
            raise ModelConfigError(f"Профиль '{key}': для backend='ollama' требуется поле 'name'")
        if self.backend == 'llama_cpp' and not self.path:
            raise ModelConfigError(f"Профиль '{key}': для backend='llama_cpp' требуется поле 'path'")

    def __repr__(self):
        return f"<ModelProfile key={self.key} backend={self.backend}>"

class ModelsConfig:
    """Контейнер для всех профилей моделей"""
    def __init__(self, config_path: str = "models.yaml"):
        self.config_path = Path(config_path)
        self.profiles: Dict[str, ModelProfile] = {}
        self.default_model: str = ""
        self._load()
    
    def _load(self):
        if not self.config_path.exists():
            raise ModelConfigError(f"Файл конфигурации не найден: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        self.default_model = raw_config.get('default_model', '')
        profiles_raw = raw_config.get('profiles', {})
        
        if not profiles_raw:
            raise ModelConfigError("В конфигурации отсутствуют профили моделей")
        
        # Загрузка всех профилей
        for key, cfg in profiles_raw.items():
            try:
                self.profiles[key] = ModelProfile(key, cfg)
            except ModelConfigError as e:
                print(f"⚠️ Пропущен некорректный профиль '{key}': {e}")
    
    def get_active_profile(self) -> ModelProfile:
        """Возвращает профиль, указанный в ACTIVE_MODEL_PROFILE или по умолчанию"""
        active_key = os.getenv('ACTIVE_MODEL_PROFILE', self.default_model)
        
        if not active_key:
            raise ModelConfigError(
                "Не указана активная модель. Установите ACTIVE_MODEL_PROFILE в .env "
                "или задайте default_model в models.yaml"
            )
        
        profile = self.profiles.get(active_key)
        if not profile:
            available = ', '.join(self.profiles.keys())
            raise ModelConfigError(
                f"Профиль '{active_key}' не найден. Доступные профили: {available}"
            )
        
        return profile
    
    def list_profiles(self) -> Dict[str, str]:
        """Возвращает словарь {ключ: описание} для всех профилей"""
        return {key: profile.description for key, profile in self.profiles.items()}

# Глобальный экземпляр для импорта
_models_config: Optional[ModelsConfig] = None

def get_models_config() -> ModelsConfig:
    """Ленивая инициализация конфигурации"""
    global _models_config
    if _models_config is None:
        _models_config = ModelsConfig()
    return _models_config