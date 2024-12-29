#!/bin/bash

gpuid=1
work_dir="./"
img_name_suffix="_0000.nii.gz"
data_path="/home/alvin/UltrAi/Datasets/ai_ready_datasets/other_datasets/MICCAI2022_multi_site_prostate_dataset/reorganized"

python precompute_embeddings.py \
    --work_dir "" \
    --data_path $data_path \
    --img_name_suffix $img_name_suffix \
    --device "cuda:$gpuid"