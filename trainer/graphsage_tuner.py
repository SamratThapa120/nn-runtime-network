
import importlib
import sys
import os
import numpy as np
sys.path.append("../")
import datetime
import optuna
import torch.distributed as dist
import pandas as pd
import torch
import os 
from ml_graph_timer.model.graphsage import LayoutGraphModel,GraphModelArugments
from ml_graph_timer.dataset.layout_dataset import NpzDataset,GraphCollator
from ml_graph_timer.losses.losses import CustomMAELoss
from allrank.models.losses import listMLE
import shutil

if __name__ == "__main__":

    if len(sys.argv)==3:
        module_name = sys.argv[2]
    elif len(sys.argv)==2:
        module_name = sys.argv[1]
    module = importlib.import_module(f"configs.{module_name}")
    tune_info = module.Configs()

    from trainer.graphsage_trainer import Trainer

    def objective(trail: optuna.trial.Trial):
        base_obj = module.Configs()
        base_obj.DISTRIBUTED=False
        base_obj.LR = trail.suggest_float("LR",1e-5,1e-1)
        base_obj.optimizer = torch.optim.Adam(base_obj.model.parameters(),lr=base_obj.LR)
        base_obj.EPOCHS = 40
        if os.path.exists(base_obj.OUTPUTDIR):
            shutil.rmtree(base_obj.OUTPUTDIR)
        trainer = Trainer(base_obj)
            
        opa = trainer.tune()
        return opa
    
    study = optuna.create_study(direction="maximize", 
                                study_name='hptune_convtrans', 
                                storage=tune_info.OPTUNA_TUNING_DB, 
                                load_if_exists=True)
    study.optimize(objective, n_trials=tune_info.OPTUNA_TUNING_TRAILS,catch=(RuntimeError))
    print(f"Completed Tuning for {module_name}!")
