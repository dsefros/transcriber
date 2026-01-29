#!/bin/bash

echo "🚀 Установка системы транскрибации для WSL (Ubuntu)"

# Обновление пакетов
sudo apt update && sudo apt upgrade -y

# Установка базовых инструментов
sudo apt install -y python3 python3-pip ffmpeg wget git curl

# Проверка Python и pip
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не установлен"
    exit 1
fi

if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 не установлен"
    exit 1
fi

# Установка CUDA (для RTX 4070)
echo "🎮 Установка CUDA..."
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# Создание папок
mkdir -p input output models

# Создание requirements.txt
cat > requirements.txt << 'EOF2'
torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
openai-whisper
pydub
pyannote.audio==2.1
transformers
huggingface-hub
numpy
tqdm
EOF2

# Установка Python-зависимостей
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Финал
echo ""
echo "✅ Установка завершена!"
echo "📌 Войдите в Hugging Face: huggingface-cli login"
echo "📦 Кладите файлы в папку input/"
echo "▶️ Запуск: python3 transcribe.py --device cuda"
echo "💡 После установки выполните в PowerShell: wsl --shutdown"
echo ""
