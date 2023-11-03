
import importlib
import sys
import os
import numpy as np
sys.path.append("../")
import datetime

import torch.distributed as dist

if __name__ == "__main__":
    if len(sys.argv)==3:
        module_name = sys.argv[2]
    elif len(sys.argv)==2:
        module_name = sys.argv[1]
    module = importlib.import_module(f"configs.{module_name}")
    base_obj = module.Configs()
    if base_obj.DISTRIBUTED:
        dist.init_process_group(backend='nccl',timeout=datetime.timedelta(seconds=7200000))
    from trainer.graphsage_trainer import Trainer
    trainer = Trainer(base_obj)

    trainer.train(base_obj.PRUNING_TOLERANCE)
    dist.destroy_process_group()
    print(f"Completed Training, and validation for {module_name}!")
