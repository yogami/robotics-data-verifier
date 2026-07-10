#!/bin/bash
apt-get update && apt-get install -y ffmpeg libsm6 libxext6 screen
cd /root/robotics-data-verifier
python3 -m venv venv
source venv/bin/activate
pip install -r https://raw.githubusercontent.com/huggingface/lerobot/main/requirements.txt
pip install -e "git+https://github.com/huggingface/lerobot.git#egg=lerobot"
pip install huggingface_hub wandb
huggingface-cli download gopalyami/aloha-act-sweep --repo-type model --include "data/*" --local-dir . --token "$HF_TOKEN"
python build_ds_with_images.py --parquet data/infection_0.parquet --output data/aloha_infected_0
screen -dmS training bash -c 'CUDA_VISIBLE_DEVICES=0 python train_act.py --parquet data/aloha_infected_0 --output-model outputs/act_model_0_300 --output-eval outputs/eval_0_300.json --epochs 100 --hf-repo gopalyami/aloha-act-sweep --hf-token "$HF_TOKEN" --hf-branch run_infected_0_seed_300 --seed 300 --infection-level 0 > train.log 2>&1'
