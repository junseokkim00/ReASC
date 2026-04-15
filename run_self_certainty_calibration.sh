#!/bin/bash
#SBATCH --job-name run
#SBATCH --gpus 2
#SBATCH --time 1-00:00:00
#SBATCH --nodelist a6k01

model_name=Qwen/Qwen2.5-7B-Instruct
dataset=gsm8k
seed=0

python self_certainty_calibration.py \
    --model_name ${model_name} \
    --dataset ${dataset} \
    --max_new_tokens 1024 \
    --batch_size 16 \
    --sample_size 128 \
    --seed ${seed}

python compute_correctness.py \
    --input_file ./logs/self_certainty_samples/sample_128_${dataset}_1024_self_certainty_${model_name//\//_}_seed_0.jsonl