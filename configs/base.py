
import torch
class Base:
    def load_state_dict(self,path,map_location="cpu"):
        statedict = torch.load(path,map_location=map_location)
        print("loading model checkpoint from epoch: ",statedict["current_step"])
        self.model.load_state_dict(statedict["model_state_dict"])
    
    
    def get_all_attributes(obj):
        attributes = {}
        for key, value in vars(obj.__class__).items():
            if not key.startswith('__'):
                attributes[key] = value
        
        for key, value in vars(obj).items():
            if not key.startswith('__'):
                attributes[key] = value           
        return attributes