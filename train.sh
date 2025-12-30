#!/bin/bash

lr=5e-5
dropout=0.1
weight_decay=1e-2
batchsize=10
dataset_path="/data/LATEX-EnxText-new/"
output_dir="/data/fength/HDDiffuser-Text/outputs/"
tensorboard_dir="/data/fength/HDDiffuser-Text/HDlayout-logs/"
device="cuda:0"
epochs=200
# resume="/data/fength/HDDiffuser-Text/outputs/20251229-113319/20251229_113350/checkpoint.pth"
# resume=None

python main.py \
    --lr "${lr}" \
    --dropout "${dropout}" \
    --batch_size "${batchsize}" \
    --weight_decay "${weight_decay}" \
    --no_aux_loss \
    --device "${device}" \
    --epochs "${epochs}" \
    --dataset_path "${dataset_path}" \
    --output_dir "${output_dir}" \
    --tensorboard_dir "${tensorboard_dir}"\
    # --resume "${resume}"