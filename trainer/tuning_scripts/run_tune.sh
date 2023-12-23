#!/bin/bash

for i in {0..200}
do
    echo "Run $i"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 graphsage_xla_tuner.py listmle_gsage_default_xla_improve $i tune1
done
