#multi-gpu training
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 train.py listmle_graphsage
#single-gpu training
# CUDA_VISIBLE_DEVICES=1 python train.py whisper_characterwise_nolm_ctcloss_frozenenc
