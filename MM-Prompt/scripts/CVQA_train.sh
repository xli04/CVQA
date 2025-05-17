#!/bin/bash

# Set the desired Python interpreter. Modify this path to the interpreter you want to use.
PYTHON_PATH=${PYTHON_PATH:-/root/miniconda3/envs/vqacl_2/bin/python}

# Define the name and output directory
name=checkpoint
output=snap/$name

# Use the specified Python interpreter to run the script
PYTHONPATH=$PYTHONPATH:./src \
$PYTHON_PATH MM-Prompt/src/cvqa.py \
    --train karpathy_train \
    --valid karpathy_val \
    --test karpathy_test \
    --optim adamw \
    --warmup_ratio 0.1 \
    --clip_grad_norm 5 \
    --lr 1e-4 \
    --epochs 25 \
    --num_workers 8 \
    --backbone 't5-base' \
    --output $output ${@:2} \
    --num_beams 5 \
    --batch_size 100 \
    --valid_batch_size 100 \
    --from_scratch \
    --memory False \
    --m_size 1500 \
    --comp_cate G-1 \
    --now_train





    



