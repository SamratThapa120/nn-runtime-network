#multi-gpu training
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=2 train.py listmle_gsage_tune
#single-gpu training
# CUDA_VISIBLE_DEVICES=1 python train.py whisper_characterwise_nolm_ctcloss_frozenenc

