
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

    study = optuna.create_study(direction="maximize", study_name='hptune_convtrans',storage="sqlite:///study.db",load_if_exists=True)

    dimensions = {
        0: [128,256],
        1: [256,512],
        2: [128,512],
    }
    if rank==0:
        trail = study.ask()
        trail.set_user_attr("unique_name", UID+tune_id)
        hiddim = trail.suggest_categorical("dimensions",[0,1,2])
        opcode_dim, graphsage_in = dimensions[hiddim]
        graphsage_layers = trail.suggest_int("graphsage_layers",2,3)
        graphsage_dropout= trail.suggest_int("graphsage_dropout",0,1)
        final_dropout = trail.suggest_int("final_dropout",0,1)
        graphsage_project = trail.suggest_categorical("graphsage_project",[True,False])
        graphsage_aggr = trail.suggest_categorical("graphsage_aggr",["mean","max"])
        LR = trail.suggest_float("LR",1e-5,1e-2)
        WD = trail.suggest_float("WD",1e-6,1e-4)
        project_after_graph_encoder = trail.suggest_categorical("project_after_graph_encoder",[True,False])

    dist.barrier()
    if rank!=0:
        time.sleep(5)
        trail = get_trial_by_name(study,  UID+tune_id)
        hiddim = trail.params["dimensions"]
        opcode_dim, graphsage_in = dimensions[hiddim]
        graphsage_layers = trail.params["graphsage_layers"]
        graphsage_dropout = trail.params["graphsage_dropout"]
        final_dropout = trail.params["final_dropout"]
        graphsage_project = trail.params["graphsage_project"]
        graphsage_aggr = trail.params["graphsage_aggr"]
        LR = trail.params["LR"]
        WD = trail.params["WD"]
        project_after_graph_encoder = trail.params["project_after_graph_encoder"]
        
    base_obj = module.Configs()
    base_obj.model_dims = GraphModelArugments(
        num_opcodes= 120,
        opcode_dim= opcode_dim,
        node_feature_dim= 126+opcode_dim,
        node_feature_dropout=0.0,
        node_feature_expand= 1,
        graphsage_in= graphsage_in,
        graphsage_hidden= graphsage_in,
        graphsage_layers= graphsage_layers,
        graphsage_dropout= graphsage_dropout/10,
        final_dropout= final_dropout/10,
        embedding_dropout=0.0,
        attention_blocks= 0,
        drop_rate= 0.1,
        attention_dropout= 0.1,
        num_heads= 2,
        is_pair_modeling= base_obj.IS_PAIR_TRAINING,
        project_after_graph_encoder = project_after_graph_encoder,
        graphsage_project =graphsage_project ,
        graphsage_normalize = True,
        graphsage_aggr = graphsage_aggr,
        return_positive_values = False,
    )
    base_obj.model = LayoutGraphModel(base_obj.model_dims)
    base_obj.LR = LR
    base_obj.WD = WD
    base_obj.optimizer = torch.optim.Adam(base_obj.model.parameters(),lr=base_obj.LR,weight_decay=base_obj.WD)
    base_obj.EPOCHS = 50
    base_obj.scheduler = None
    # base_obj.scheduler = torch.optim.lr_scheduler.OneCycleLR(base_obj.optimizer,max_lr=base_obj.LR,steps_per_epoch=base_obj.steps_per_epoch,epochs=base_obj.EPOCHS,pct_start=0.1)
    
    trainer = Trainer(base_obj)

    try:
        value = trainer.train(prune_epochs=4)
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
