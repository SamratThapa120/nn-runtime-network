#multi-gpu training
CUDA_VISIBLE_DEVICES=0,1,6,7 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=4 train.py listmle_gsage_random_xla_embedding_hop1 