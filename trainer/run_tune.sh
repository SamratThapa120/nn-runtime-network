#!/bin/bash

for i in {1..200}
do
    echo "Run $i"
    CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=2 graphsage_tuner.py listmle_gsage_tune $i tune1
done
