import torch
import torch.distributed as dist

torch.cuda.empty_cache()
dist.destroy_process_group()
