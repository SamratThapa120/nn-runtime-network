import pandas as pd
import torch
import os 
from ml_graph_timer.model.graphsage import LayoutGraphModel,GraphModelArugments
from ml_graph_timer.dataset.layout_dataset import NpzDataset,GraphCollator,StreamingCollator
from ml_graph_timer.losses.losses import CustomMAELoss,CustomMSELoss
from allrank.models.losses import listMLE

from .base import Base

class Configs(Base):
    OUTPUTDIR="../workdir/listmle_graphsage_bestparams"

    TRAIN_DATA_PATH="/app/dataset/various_splits/all_layout_trimmed/train"
    VALID_DATA_PATH="/app/dataset/various_splits/all_layout_trimmed/valid"
    TEST_DATA_PATH="/app/dataset/various_splits/all_layout_trimmed/test"
    NORMALIZER_PATH="/app/dataset/various_splits/all_layout/normalizers/normalizers.npy"

    OPTUNA_TUNING_DB="sqlite:///study.db"
    OPTUNA_TUNING_TRAILS= 1000

    USE_DATASET_LEN=None   #Set to small number while debugging
    SAMPLES_PER_GPU=2
    N_GPU=4
    VALIDATION_BS=4
    PIN_MEMORY=True
    NUM_WORKERS=4
    NUM_WORKERS_VAL=4
    DISTRIBUTED=True

    LR=0.001

    EPOCHS=500
    MIN_CONFIGS=2
    SAMPLE_CONFIGS=16
    SAMPLE_CONFIGS_VAL=16
    RUNTIME_PADDING=-1
    CONFIG_PADDING=0
    IS_PAIR_TRAINING=False

    AUTOCAST=False
    GRADIENT_STEPS=1
    VALIDATION_FREQUENCY=2   # Number of epochs

    CLIP_NORM=1e-2
    WD=0.000023

    PRUNING_TOLERANCE=10
    def __init__(self,inference_files=None,inference_text=None,use_numpy=False):
        self.device = "cuda"
        self.model_dims = GraphModelArugments(
            num_opcodes= 120,
            opcode_dim= 128,
            node_feature_dim= 126+128,
            node_feature_dropout=0.0,
            node_feature_expand= 1,
            graphsage_in= 512,
            graphsage_hidden= 512,
            graphsage_layers= 2,
            graphsage_dropout= 0.0,
            final_dropout= 0.0,
            embedding_dropout= 0.0,
            is_pair_modeling= self.IS_PAIR_TRAINING,
            project_after_graph_encoder = True,
            graphsage_aggr = "mean",
            return_positive_values = False,
            graphsage_project = False,
        )
        self.model = LayoutGraphModel(self.model_dims)
        
        self.train_dataset = NpzDataset(self.TRAIN_DATA_PATH,min_configs=self.MIN_CONFIGS, max_configs=self.SAMPLE_CONFIGS,normalizers=self.NORMALIZER_PATH,sample_num=self.USE_DATASET_LEN)
        self.valid_dataset = NpzDataset(self.VALID_DATA_PATH,min_configs=self.MIN_CONFIGS, max_configs=self.SAMPLE_CONFIGS_VAL,normalizers=self.NORMALIZER_PATH,sample_num = self.USE_DATASET_LEN,random_config_sampling=False,isvalid=True)
        self.test_dataset = NpzDataset(self.TEST_DATA_PATH,min_configs=self.MIN_CONFIGS, max_configs=-1,normalizers=self.NORMALIZER_PATH,sample_num = self.USE_DATASET_LEN,random_config_sampling=False,isvalid=True)

        print(f"length of train: {len(self.train_dataset)}, length of valid: {len(self.valid_dataset)}, length of test: {len(self.test_dataset)}")

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.LR,weight_decay=self.WD)
        self.steps_per_epoch = len(self.train_dataset)//(self.SAMPLES_PER_GPU*self.N_GPU)+1
        self.VALIDATION_FREQUENCY  = self.VALIDATION_FREQUENCY * self.steps_per_epoch
        # self.scheduler = None
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.LR,steps_per_epoch=self.steps_per_epoch,epochs=self.EPOCHS,pct_start=0.1)
        # self.criterion = CustomMAELoss(padding=self.RUNTIME_PADDING)
        # self.criterion = CustomMSELoss(padding=self.RUNTIME_PADDING)
    
        self.criterion = listMLE


        self.dataloder_collate = GraphCollator(max_configs=self.SAMPLE_CONFIGS,configs_padding=self.CONFIG_PADDING,runtime_padding=self.RUNTIME_PADDING,provide_pair_matrix=self.IS_PAIR_TRAINING)
        self.dataloder_collate_val = GraphCollator(max_configs=self.SAMPLE_CONFIGS_VAL,configs_padding=self.CONFIG_PADDING,runtime_padding=self.RUNTIME_PADDING,provide_pair_matrix=self.IS_PAIR_TRAINING)
        
        self.stream_dataloder_collate = StreamingCollator(batch_size=2,max_configs=self.SAMPLE_CONFIGS,configs_padding=self.CONFIG_PADDING,runtime_padding=self.RUNTIME_PADDING,provide_pair_matrix=self.IS_PAIR_TRAINING)
