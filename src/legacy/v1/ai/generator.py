"""
Единый интерфейс генерации текста для разных бэкендов
"""
import ollama
import torch
import gc
from pathlib import Path
from src.legacy.config.models import ModelProfile

# Кэш для ленивой загрузки
_OLLAMA_CLIENT = ollama.Client()
_LLAMA_CPP_MODEL = None
_LLAMA_CPP_PROFILE_KEY = None

def _free_gpu_memory():
    """Принудительное освобождение памяти"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def _generate_ollama(prompt: str, profile: ModelProfile) -> str:
    """Генерация через Ollama API"""
    generation_options = {
        "temperature": profile.params.get("temperature", 0.1),
        "top_p": profile.params.get("top_p", 0.9),
        "repeat_penalty": profile.params.get("repeat_penalty", 1.15),
        "num_predict": profile.params.get("num_predict", 2000),
    }
    generation_options = {k: v for k, v in generation_options.items() if v is not None}
    
    try:
        response = _OLLAMA_CLIENT.chat(
            model=profile.name,
            messages=[{"role": "user", "content": prompt}],
            options=generation_options,
            stream=False
        )
        return response["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Ollama ошибка ({profile.name}): {e}")

def _load_llama_cpp_model(profile: ModelProfile):
    """Ленивая загрузка модели llama.cpp с кэшированием и очисткой памяти"""
    global _LLAMA_CPP_MODEL, _LLAMA_CPP_PROFILE_KEY
    if _LLAMA_CPP_MODEL is not None and _LLAMA_CPP_PROFILE_KEY == profile.key:
        return _LLAMA_CPP_MODEL
    
    # 🔥 Очистка перед загрузкой новой модели
    _free_gpu_memory()
    
    model_path = Path(profile.path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    
    from llama_cpp import Llama
    load_params = {
        "model_path": str(model_path),
        "n_ctx": profile.params.get("n_ctx", 4096),
        "n_gpu_layers": profile.params.get("n_gpu_layers", 35),
        "n_batch": profile.params.get("n_batch", 512),
        "verbose": profile.params.get("verbose", False),
    }
    print(f"⏳ Загрузка {profile.key} в GPU (слоёв: {load_params['n_gpu_layers']})...")
    _LLAMA_CPP_MODEL = Llama(**load_params)
    _LLAMA_CPP_PROFILE_KEY = profile.key
    print(f"✅ {profile.key} готова")
    return _LLAMA_CPP_MODEL

def _format_prompt_for_model(prompt: str, model_key: str) -> str:
    """Форматирование промпта под специфику модели (без дублирующего <s>)"""
    if "mistral" in model_key.lower():
        return f"[INST] {prompt} [/INST]"
    elif "phi3" in model_key.lower():
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    else:
        return prompt

def _generate_llama_cpp(prompt: str, profile: ModelProfile) -> str:
    """Генерация через llama-cpp-python"""
    try:
        llm = _load_llama_cpp_model(profile)
        formatted_prompt = _format_prompt_for_model(prompt, profile.key)
        generation_params = {
            "prompt": formatted_prompt,
            "temperature": profile.params.get("temperature", 0.1),
            "top_p": profile.params.get("top_p", 0.9),
            "repeat_penalty": profile.params.get("repeat_penalty", 1.15),
            "max_tokens": profile.params.get("max_tokens", 2048),
            "stop": ["</s>", "<|end|>", "<|user|>", "<|assistant|>", "[INST]", "[/INST]"],
        }
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        response = llm(**generation_params)
        text = response["choices"][0]["text"].strip()
        for token in ["</s>", "<|end|>", "<|user|>", "<|assistant|>", "[INST]", "[/INST]"]:
            text = text.replace(token, "").strip()
        return text
    except Exception as e:
        raise RuntimeError(f"llama-cpp ошибка ({profile.key}): {e}")

def generate_text(prompt: str, profile: ModelProfile) -> str:
    """Единая точка входа для генерации"""
    if profile.backend == "ollama":
        return _generate_ollama(prompt, profile)
    elif profile.backend == "llama_cpp":
        return _generate_llama_cpp(prompt, profile)
    else:
        raise ValueError(f"Неизвестный бэкенд: {profile.backend}")