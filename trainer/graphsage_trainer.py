import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from .utils import setup_logger,MetricsStore
import os
from tqdm import tqdm
from ml_graph_timer.callbacks.evaluation import ModelValidationCallback

from contextlib import nullcontext

class Trainer:
    def __init__(self, base_obj):

        self.__dict__.update(base_obj.get_all_attributes())

        if self.DISTRIBUTED:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = f"cuda:{self.rank}"
            self.model = self.model.to(self.device)
            self.model = DistributedDataParallel(self.model, device_ids=[self.rank],find_unused_parameters=False)
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank,shuffle=True)
        else:
            self.rank=0
            self.train_sampler = None
            self.model = self.model.to(self.device)

        self.start_epoch = 0
        self.current_step=0
        if os.path.exists(os.path.join(self.OUTPUTDIR,"latest_model.pkl")):
            if self.DISTRIBUTED:
                model = self.model.module
            else:
                model = self.model
            statedict = torch.load(os.path.join(self.OUTPUTDIR,"latest_model.pkl"))
            self.current_step = statedict["current_step"]
            self.start_epoch = self.current_step//self.steps_per_epoch
            model.load_state_dict(statedict["model_state_dict"])
            print("loaded model state from step: ",self.current_step)
            if self.scheduler is not None:
                for _ in range(self.current_step):
                    self.scheduler.step()

        collate_func=None
        if hasattr(self,"dataloder_collate"):
            print("Using collate function..")
            collate_func= self.dataloder_collate
        self.train_loader = DataLoader(self.train_dataset,collate_fn=collate_func ,batch_size=self.SAMPLES_PER_GPU, sampler=self.train_sampler, pin_memory=self.PIN_MEMORY, num_workers=self.NUM_WORKERS)
        
        os.makedirs(self.OUTPUTDIR,exist_ok=True)
        self.logger = setup_logger(os.path.join(self.OUTPUTDIR,"logs.txt"))
        self.metrics = MetricsStore()

        if self.rank==0:
            self.valid_loader = DataLoader(self.valid_dataset,collate_fn=collate_func, batch_size=self.VALIDATION_BS, pin_memory=self.PIN_MEMORY, num_workers=self.NUM_WORKERS_VAL)
            self.evaluation_callback = ModelValidationCallback(self,self.metrics,self.valid_loader)
        if self.AUTOCAST:
            self.train_context = torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)
        else:
            self.train_context = nullcontext()
        self.accum_steps = self.GRADIENT_STEPS
    #@profile
    def train_one_epoch(self,epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        tqdm_loader = tqdm(self.train_loader,desc=f"Train epoch: {epoch}",disable=self.rank!=0)
        updatefreq=5

        self.optimizer.zero_grad()
        for i,batch in enumerate(tqdm_loader):
            with self.train_context  and torch.set_grad_enabled(True):
                outputs = self.model(batch["node_features"].to(self.device), 
                                     batch["node_separation"].to(self.device), 
                                     batch["node_ops"].to(self.device), 
                                     batch["edges"].to(self.device), 
                                     batch["batches"].to(self.device)
                                     )
                loss = self.criterion(outputs, batch["config_runtimes"].to(self.device))/self.accum_steps
                loss.backward()
                if ((i + 1) % self.accum_steps == 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP_NORM)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
            total_loss += loss.item()
            if i%updatefreq==0:
                if torch.isnan(loss):
                    print("Found nan loss")
                    exit()
                tqdm_loader.set_description(f"loss: {loss.item():.4f} ")
            num_batches += 1
            self.current_step+=1
            if self.current_step%self.VALIDATION_FREQUENCY==0:
                self.optimizer.zero_grad()
                dist.barrier()
                if self.rank == 0:
                    self.validate(self.current_step)
                    avg_loss = total_loss / num_batches
                    self.metrics(self.current_step,"training_loss",avg_loss)
                    self.logger.info(f"###Iter: {self.current_step}  ::  {self.metrics.get_metrics_by_epoch(self.current_step)}")
                    num_batches=0
                    total_loss=0.0
                dist.barrier()

    def validate(self,current_step):
        if self.rank != 0 or current_step%self.VALIDATION_FREQUENCY!=0:
            return
        self.model.eval()
        self.evaluation_callback(current_step)

    def infer(self, batch):
        with torch.no_grad():
            outputs = self.model(batch["node_features"].to(self.device), 
                                batch["node_separation"].to(self.device), 
                                batch["node_ops"].to(self.device), 
                                batch["edges"].to(self.device), 
                                batch["batches"].to(self.device)
                            ).detach().cpu()
        return outputs


    def get_state_dict(self):
        if self.DISTRIBUTED:
            model = self.model.module.state_dict()  
        else:
            model = self.model.state_dict()
        return model
    
    def _savemodel(self,current_step,path):
        torch.save({
            'current_step': current_step,
            'model_state_dict': self.get_state_dict(),
        }, path)

    def train(self):
        if self.rank==0:
            print("Starting training....")
        for epoch in range(self.start_epoch,self.EPOCHS):
            if self.DISTRIBUTED:
                self.train_sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            self._savemodel(self.current_step,os.path.join(self.OUTPUTDIR,"latest_model.pkl"))
        if self.rank==0:
            self.metrics.to_dataframe().to_csv(os.path.join(self.OUTPUTDIR,"metrics.csv"))