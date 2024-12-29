#!/bin/bash

gpuid=0
center=1
work_dir="./"
mixprecision=true
pred_embedding=true
python desam_train_gridpoints.py \
    --gpuid $gpuid \
    --center $center \
    --work_dir $work_dir \
    --pred_embedding $pred_embedding \
    --mixprecision $mixprecision