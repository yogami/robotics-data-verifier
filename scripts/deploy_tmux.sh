#!/bin/bash
set -e

echo "Starting deployment..."
cd /root/robotics-data-verifier

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir "lerobot<2" -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=200
pip install --no-cache-dir cryptography datasets huggingface_hub torchvision torchaudio

huggingface-cli login --token "$HF_TOKEN"

if [ ! -d "data/aloha_infected_0" ]; then
    echo "Building dataset..."
    python build_ds_with_images.py
fi

echo "Running the seed sweep..."
bash scripts/run_all_seeds.sh 2>&1 | tee master_sweep.log
