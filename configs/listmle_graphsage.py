import pandas as pd
import torch
import os 
from ml_graph_timer.model.graphsage import LayoutGraphModel,GraphModelArugments
from ml_graph_timer.dataset.layout_dataset import NpzDataset,GraphCollator

from allrank.models.losses import listMLE

from .base import Base

class Configs(Base):
    OUTPUTDIR="../workdir/whisperbase_characterlevel"

    TRAIN_DATA_PATH="/app/dataset/various_splits/all_layout/train"
    VALID_DATA_PATH="/app/dataset/various_splits/all_layout/valid"
    VALID_DATA_PATH="/app/dataset/various_splits/all_layout/test"
    NORMALIZER_PATH="/app/dataset/various_splits/all_layout/normalizers.npy"
    USE_DATASET_LEN=None   #Set to small number while debugging
    SAMPLES_PER_GPU=8
    N_GPU=1
    VALIDATION_BS=4
    VALIDATION_FREQUENCY=1
    PIN_MEMORY=True
    NUM_WORKERS=4
    NUM_WORKERS_VAL=4
    DISTRIBUTED=True

    WD=1e-5
    LR=0.0005
    EPOCHS=50
    MIN_CONFIGS=2
    SAMPLE_CONFIGS=10
    RUNTIME_PADDING=-1
    CONFIG_PADDING=0
    IS_PAIR_TRAINING=False

    AUTOCAST=False
    GRADIENT_STEPS=1
    VALIDATION_FREQUENCY=1000

    def __init__(self,inference_files=None,inference_text=None,use_numpy=False):
        self.device = "cuda"
        self.model_dims = GraphModelArugments()
        self.model = LayoutGraphModel(self.model_dims)
        
        self.train_dataset = NpzDataset(self.TRAIN_DATA_PATH,min_configs=self.MIN_CONFIGS, max_configs=self.SAMPLE_CONFIGS,normalizers=self.NORMALIZER_PATH,sample_num=self.USE_DATASET_LEN)
        self.valid_dataset = NpzDataset(self.VALID_DATA_PATH,min_configs=self.MIN_CONFIGS, max_configs=self.SAMPLE_CONFIGS,normalizers=self.NORMALIZER_PATH,sample_num = self.USE_DATASET_LEN)

        print(f"length of train: {len(self.train_dataset)}, length of valid: {len(self.valid_dataset)}")

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.LR,weight_decay=self.WD)
        self.steps_per_epoch = len(self.train_dataset)//(self.SAMPLES_PER_GPU*self.N_GPU)+1
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.LR,steps_per_epoch=self.steps_per_epoch,epochs=self.EPOCHS,pct_start=0.1)
        self.criterion = listMLE

        self.dataloder_collate = GraphCollator(max_configs=self.SAMPLE_CONFIGS,configs_padding=self.CONFIG_PADDING,runtime_padding=self.RUNTIME_PADDING,provide_pair_matrix=self.IS_PAIR_TRAINING)