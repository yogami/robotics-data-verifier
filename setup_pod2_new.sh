#!/bin/bash
cd /root/robotics-data-verifier
source venv/bin/activate
pip install "lerobot<2" -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=100
python build_ds_with_images.py --parquet data/infection_0.parquet --output data/aloha_infected_0
CUDA_VISIBLE_DEVICES=0 python train_act.py --parquet data/aloha_infected_0 --output-model outputs/act_model_0_300 --output-eval outputs/eval_0_300.json --epochs 100 --hf-repo gopalyami/aloha-act-sweep --hf-token "$HF_TOKEN" --hf-branch run_infected_0_seed_300 --seed 300 --infection-level 0 > train.log 2>&1
