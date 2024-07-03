#!/bin/bash
NUM_PROC=1
GPUS=0

cd src
shift
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC ./main.py redundant \
    --exp_id test \
    --data_dir ../data/data_npz/test \
    --reg_loss l1 --cls_loss bce \
    --Prob_weight 0 --RC_weight 0 --Func_weight 0 --Redundant_weight 10\
    --num_rounds 1 \
    --batch_size 2 \
    --gpus ${GPUS} \
    --no_rc
