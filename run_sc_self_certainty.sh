#!/bin/bash
#SBATCH --job-name self_certainty_csqa
#SBATCH --gpus 2
#SBATCH --time 3-00:00:00
#SBATCH --nodelist ac01


# model_name=google/gemma-3-4b-it
model_name=meta-llama/Llama-3.2-3B-Instruct
dataset=gsm8k
seed=0

python sc_with_self_certainty.py \
    --model_name ${model_name} \
    --dataset ${dataset} \
    --max_new_tokens 1024 \
    --batch_size 16 \
    --seed ${seed}

python compute_prediction.py \
    --model_name ${model_name} \
    --dataset ${dataset} \
    --batch_size 128

python compute_correctness.py \
    --input_file ./logs/self_certainty/sc_16_${dataset}_1024_self_certainty_${model_name//\//_}_seed_0.jsonl