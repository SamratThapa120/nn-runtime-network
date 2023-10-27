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
import optuna
from dataclasses import dataclass, asdict
import json

class Trainer:
    def __init__(self, base_obj):

        self.__dict__.update(base_obj.get_all_attributes())

        #Create a dictionary to save the configuration files
        all_configs = self.__dict__.copy()
        all_configs = {k:v for k,v in all_configs.items() if type(v) in {int,float,str,bool}}
        all_configs["model_arguments"] = asdict(self.model.arguments)

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
            with open(os.path.join(self.OUTPUTDIR,'configs.json'), 'w') as file:
                json.dump(all_configs, file, indent=4)
        if self.AUTOCAST:
            self.train_context = torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)
        else:
            self.train_context = nullcontext()
        self.accum_steps = self.GRADIENT_STEPS
        self.target_key = "optimization_matrix" if self.IS_PAIR_TRAINING else "config_runtimes"
    
    def continue_training(self,tolerance=5):
        trainlosses = self.metrics.get_metric_all("training_loss")
        opa = self.metrics.get_metric_all("ordered_pair_accuracy")
        if min(trainlosses) not in trainlosses[-tolerance:]:
            return False
        if max(opa) not in opa[-tolerance:]:
            return False
        if len(opa)>tolerance and max(opa)<0.58:
            return False
        return True
    #@profile
    def train_one_epoch(self,epoch,early_stop=False,tolerance=5):
        continue_training=True
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        tqdm_loader = tqdm(self.train_loader,desc=f"Train epoch: {epoch}",disable=self.rank!=0)
        updatefreq=5

        self.optimizer.zero_grad()
        for i,batch in enumerate(tqdm_loader):
            with self.train_context  and torch.set_grad_enabled(True):
                outputs = self.model(batch["node_features"].to(self.device), 
                                     batch["node_config_features"].to(self.device), 
                                     batch["node_separation"].to(self.device), 
                                     batch["node_ops"].to(self.device), 
                                     batch["edges"].to(self.device), 
                                     batch["batches"].to(self.device)
                                     )
                
                loss = self.criterion(outputs, batch[self.target_key].to(self.device))/self.accum_steps
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
                    continue_training=False
                tqdm_loader.set_description(f"loss: {loss.item():.4f} ")
            num_batches += 1
            self.current_step+=1
            if self.current_step%self.VALIDATION_FREQUENCY==0:
                self.optimizer.zero_grad()
                if self.rank == 0:
                    self.validate(self.current_step)
                    self.logger.info(f"###Iter: {self.current_step}  ::  {self.metrics.get_metrics_by_epoch(self.current_step)}")
                    if early_stop==True:
                        continue_training = self.continue_training(tolerance=tolerance)
                if self.DISTRIBUTED:
                    dist.barrier()
        if self.rank == 0:            
            avg_loss = total_loss / num_batches
            self.metrics(self.current_step,"training_loss",avg_loss)
            self.logger.info(f"###Iter: {self.current_step}  ::  {self.metrics.get_metrics_by_epoch(self.current_step)}")
        return continue_training
    def validate(self,current_step):
        if self.rank != 0 or current_step%self.VALIDATION_FREQUENCY!=0:
            return
        self.model.eval()
        self.evaluation_callback(current_step)

    def infer(self, batch):
        with torch.no_grad():
            outputs = self.model(batch["node_features"].to(self.device), 
                                batch["node_config_features"].to(self.device),  
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

    def train(self,prune_epochs=5):
        if self.rank==0:
            print("Starting training....")
        for epoch in range(self.start_epoch,self.EPOCHS):
            self.train_sampler.set_epoch(epoch)
            dist.barrier()
                
            continue_train = self.train_one_epoch(epoch,early_stop=True,tolerance=prune_epochs)
            if self.rank == 0:
                self._savemodel(self.current_step,os.path.join(self.OUTPUTDIR,"latest_model.pkl"))
                if not continue_train :
                    should_continue = torch.tensor(0.0).cuda()
                else:
                    should_continue = torch.tensor(1.0).cuda()
            else:
                should_continue = torch.tensor(1.0).cuda()

            dist.all_reduce(should_continue, op=dist.ReduceOp.MIN)
            dist.barrier()
            if should_continue.item() == 0:
                print(f"Early stopping RANK: {self.rank} ..... ")
                return self.evaluation_callback.opa if self.rank==0 else -1

        if self.rank==0:
            self.metrics.to_dataframe().to_csv(os.path.join(self.OUTPUTDIR,"metrics.csv"))
        
        return self.evaluation_callback.opa if self.rank==0 else -1
    
    def tune(self,prune_epochs=5):

        if self.rank==0:
            print("Starting tuning....")
        for epoch in range(self.start_epoch,self.EPOCHS):
            if self.DISTRIBUTED:
                self.train_sampler.set_epoch(epoch)
            continue_train = self.train_one_epoch(epoch,early_stop=True,tolerance=prune_epochs)
            if not continue_train:
                return self.evaluation_callback.opa
        return self.evaluation_callback.opa if self.rank==0 else -1