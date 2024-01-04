#multi-gpu training
CUDA_VISIBLE_DEVICES=2,3,4,5  python -m torch.distributed.launch --nproc_per_node=4 train.py listmle_gsage_random_xla_embedding_hop2 
