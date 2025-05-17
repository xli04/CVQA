#!/bin/bash

# Set the desired Python interpreter. Modify this path to the interpreter you want to use.
PYTHON_PATH=${PYTHON_PATH:-/root/miniconda3/envs/vqacl_2/bin/python}

# Define training configurations
port=66680
m_size=500
epoch=3
seed=6666
name=nextqa_CL
output=snap/nextqa/checkpoint

# Run the script using the specified Python interpreter
PYTHONPATH=$PYTHONPATH:./src \
$PYTHON_PATH VL-T5/nextqa/nextqa_CL.py \
    --local_rank 0 \
    --multiGPU \
    --optim adamw \
    --warmup_ratio 0.1 \
    --clip_grad_norm 5 \
    --num_workers 8 \
    --backbone t5-base \
    --num_beams 5 \
    --valid_batch_size 100 \
    --epochs $epoch \
    --batch_size 100 \
    --from_scratch \
    --memory True\
    --m_size $m_size \
    --comp_cate G-1 \
    --ifseed \
    --seed $seed \
    --proto_beta 0.5 \
    --proto_alpha 0.3 \
    --output $output





