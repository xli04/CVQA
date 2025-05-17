#!/bin/bash

PYTHON_PATH=${PYTHON_PATH:-/root/miniconda3/envs/vqacl_2/bin/python}

name=VQAv2_Our
output=snap/$name

PYTHONPATH=$PYTHONPATH:./src \
$PYTHON_PATH MM-Prompt/src/cvqa.py \
    --train karpathy_train \
    --valid karpathy_val \
    --test karpathy_test \
    --optim adamw \
    --warmup_ratio 0.1 \
    --clip_grad_norm 5 \
    --lr 1e-4 \
    --epochs 1 \
    --num_workers 4 \
    --backbone 't5-base' \
    --output $output ${@:2} \
    --num_beams 5 \
    --batch_size 40 \
    --valid_batch_size 50 \
    --from_scratch \





    




