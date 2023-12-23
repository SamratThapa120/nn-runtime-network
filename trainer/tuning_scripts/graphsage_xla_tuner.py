
import importlib
import sys
import os
import numpy as np
sys.path.append("../../")
import datetime
import optuna
import torch.distributed as dist
import pandas as pd
import torch
import os 
from ml_graph_timer.model.graphsage import LayoutGraphModel,GraphModelArugments
from ml_graph_timer.dataset.layout_dataset import NpzDataset,GraphCollator
from ml_graph_timer.losses.losses import CustomMAELoss
from allrank.models.losses import listMLE,listNet
import shutil
import time

def get_latest_running_trial(study):
    # Fetch all trials from the study
    trials = study.get_trials()

    # Filter out trials that are still running
    running_trials = [trial for trial in trials if trial.state == optuna.trial.TrialState.RUNNING]

    # Sort the running trials by their start datetime and return the latest
    sorted_running_trials = sorted(running_trials, key=lambda x: x.datetime_start, reverse=True)

    return sorted_running_trials[0] if sorted_running_trials else None

def get_trial_by_name(study, name):
    for trial in reversed(study.get_trials()):
        if trial.user_attrs.get("unique_name") == name:
            return trial
    return None

if __name__ == "__main__":
    module_name = sys.argv[2]
    tune_id = sys.argv[3]
    UID = sys.argv[4]


    module = importlib.import_module(f"configs.{module_name}")
    base_obj = module.Configs()

    if base_obj.DISTRIBUTED:
        dist.init_process_group(backend='nccl',timeout=datetime.timedelta(seconds=7200000))

    from trainer.graphsage_trainer import Trainer
    if base_obj.DISTRIBUTED:
        rank = dist.get_rank()
    else:
        rank=0
    print("RANK, TUNEID, MODULENAME : ",rank,tune_id,module_name)

    study = optuna.create_study(direction="maximize", study_name='hptune_convtrans',storage="sqlite:///study_xla_default.db",load_if_exists=True)
    if rank==0:
        trail = study.ask()
        trail.set_user_attr("unique_name", UID+tune_id)
        max_configs = trail.suggest_categorical("max_config",[4,8,16,32,64])
        losstype= trail.suggest_categorical("losstype",["listnet","listmle"])
    dist.barrier()
    if rank!=0:
        time.sleep(5)
        trail = get_trial_by_name(study,  UID+tune_id)
        max_configs = trail.params["max_config"]
        losstype = trail.params["losstype"]
    
    base_obj = module.Configs()
    base_obj.train_dataset.max_configs = max_configs
    base_obj.criterion = listMLE if losstype=="listmle" else listNet
    
    trainer = Trainer(base_obj)

    try:
        value = trainer.train(prune_epochs=10)
    except Exception as e:
        print("ERROR OCCURED: ",str(e))
        if rank==0:
            trail.set_user_attr("error_name", str(e))
            value = trainer.evaluation_callback.opa
    
    if rank==0:
        if os.path.exists(base_obj.OUTPUTDIR):
            shutil.rmtree(base_obj.OUTPUTDIR)
        study.tell(trial=trail,values=value)
    dist.barrier()
    print(f"Completed Tuning for {module_name}!")
