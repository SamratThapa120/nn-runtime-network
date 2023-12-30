from typing import Any
import numpy as np
import json
class ComposeAll:
    def __init__(self,transforms=[]) -> None:
        self.transforms = transforms
    def __call__(self, input):
        for tform in self.transforms:
            input = tform(input)    
        return input
    def __getitem__(self,idx):
        return self.transforms[idx]
    
class ConfigCentricTransform:
    def __init__(self,distance=0):
        self.distance=distance
    def __call__(self,info):
        mask = np.isin(info["edge_index"], info["node_config_ids"]).any(axis=1)
        connected_nodes = np.unique(info["edge_index"][mask])
        for i in range(self.distance):
            mask = np.isin(info["edge_index"], connected_nodes).any(axis=1)
            connected_nodes = np.unique(info["edge_index"][mask])  

        info["edge_index"] = info["edge_index"][mask]
        info["node_feat"] = info["node_feat"][connected_nodes]
        info["node_opcode"] = info["node_opcode"][connected_nodes]
        updated_nodes = {x:i for i,x in enumerate(connected_nodes)}
        vectorized_replace = np.vectorize(lambda x: updated_nodes[x])

        info["edge_index"] = vectorized_replace(info["edge_index"])
        info["node_config_ids"] = vectorized_replace(info["node_config_ids"])
        return info

class AddFeatures:
    def __init__(self,mask_value=-1,negate_minor_major_layout=True,add_minor_major_layout_sizes=True,add_conf_output_sizes=True):
        self.negate_minor_major_layout = negate_minor_major_layout
        self.mask_value = mask_value
        self.add_minor_major_layout_sizes = add_minor_major_layout_sizes
        self.add_conf_output_sizes = add_conf_output_sizes

    def __call__(self,info):
        mask = info["node_feat"][:,21:27]<0
        if self.negate_minor_major_layout:
            info["node_feat"][:,134:][mask] = self.mask_value
        if self.add_minor_major_layout_sizes:
            indices = np.where(~mask)
            new_feat = self.mask_value * np.ones_like(info["node_feat"][:,21:27])
            dims = info["node_feat"][:,134:][indices[0],indices[1]].astype(int)
            new_feat[indices[0],indices[1]] = info["node_feat"][:,21:27][indices[0],dims]
            info["node_feat"][:,134:] = new_feat
            # info["node_feat"] = np.concatenate([info["node_feat"],new_feat],axis=1)
        if self.add_conf_output_sizes:
            indices = np.where(info["node_config_feat"][:,:,0:6]!=-1)
            new_feat = self.mask_value * np.ones_like(info["node_config_feat"][:,:,0:6])
            dims = info["node_config_feat"][:,:,0:6][indices[0],indices[1],indices[2]].astype(int)

            new_feat[indices[0],indices[1],indices[2]] = info["node_feat"][:,21:27][info["node_config_ids"][indices[1]],dims]
            info["node_config_feat"][:,:,0:6]  = new_feat
            # info["node_config_feat"] = np.concatenate([info["node_config_feat"],new_feat],axis=2)       
        return info

    
class LogNormalization:
    def __init__(self,mask_value=-1):
        self.log_columns = [21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  37,  38,  39,43,  44,  51,  52,  53,  54,  59,  60,  67,  68,  75,  76,  83,84, 107, 108, 109, 110, 111, 112, 117, 118, 119, 120, 121, 122,123, 124, 126, 127, 129, 130, 131, 132]
        self.mask_value = mask_value
    def __call__(self,info):
        mask = info["node_feat"][:, self.log_columns] < 1
        info["node_feat"][:, self.log_columns] = np.where(mask, 1, info["node_feat"][:, self.log_columns])
        info["node_feat"][:, self.log_columns] = np.log2(info["node_feat"][:, self.log_columns])
        info["node_feat"][:, self.log_columns] = np.where(mask, self.mask_value, info["node_feat"][:, self.log_columns])
        return info
    
class SqrtNormalization:
    def __init__(self,mask_value=0):
        self.log_columns = [21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  37,  38,  39,43,  44,  51,  52,  53,  54,  59,  60,  67,  68,  75,  76,  83,84, 107, 108, 109, 110, 111, 112, 117, 118, 119, 120, 121, 122,123, 124, 126, 127, 129, 130, 131, 132]
        self.mask_value = mask_value
    def __call__(self,info):
        mask = info["node_feat"][:, self.log_columns] < 0
        info["node_feat"][:, self.log_columns] = np.where(mask, 1, info["node_feat"][:, self.log_columns])
        info["node_feat"][:, self.log_columns] = np.sqrt(info["node_feat"][:, self.log_columns])
        info["node_feat"][:, self.log_columns] = np.where(mask, self.mask_value, info["node_feat"][:, self.log_columns])
        return info
    
class CategorizeFilter:
    def get_unique(self,array):
        total = array[:,0]
        for i in range(1,array.shape[1]):
            total = total+ (10**i)* array[:,i]
        return total.astype(int)
    def __init__(self,information="/app/dataset/various_splits/all_layout/uniques/values.json"):
        self.nf_groups =[
            ("dimension",31,37),
            ("minmaj",134,140),
        ]
        self.cf_groups =[
            ("output",0,6),
            ("input",6,12),
            ("kernel",12,18),
        ]
        self.keep_indices_nf= [  0,   3,   6,   7,  10,  13,  15,  18,  20,  21,  22,  23,  24,
         25,  26,  27,  28,  29,  30, 37,38,  39,  40,  41,  43,  44,  45,  46,  47,  48,  49,  51,  52,
         53,  54,  55,  59,  60,  61,  62,  63,  64,  67,  68,  69,  70,
         71,  72,  73,  75,  76,  77,  78,  79,  80,  81,  83,  84,  85,
         86,  87,  91,  92,  93,  94,  95,  96,  97,  99, 100, 101, 102,
        103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117,
        118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
        131, 132, 133]
        self.cat_info = {k:{a:i for i,a in enumerate(np.sort(v))} for k,v in json.load(open(information)).items()}
    def __call__(self, info):
        concats = [info["node_feat"][:,self.keep_indices_nf]]
        for dim,i,j in self.nf_groups:
            vectorizer = np.vectorize(lambda x: self.cat_info[dim][x])
            concats.append(vectorizer(self.get_unique(info["node_feat"][:,i:j])).reshape(-1,1))
        info["node_feat"] = np.concatenate(concats,1)
        concats = []
        c,n,_ = info["node_config_feat"].shape
        for dim,i,j in self.cf_groups:
            vectorizer = np.vectorize(lambda x: self.cat_info[dim][x] if x in self.cat_info[dim] else 0)
            concats.append(vectorizer(self.get_unique(info["node_config_feat"][:,:,i:j].reshape(-1,j-i))).reshape(c,n,1))
        info["node_config_feat"] = np.concatenate(concats,2)
        return info

class AggregateCategoricalFeats:
    def __init__(self,mask_value=-1,negate_minor_major_layout=True):
        self.nf_groups =[31,32,33,34,35,36,134,135,136,137,138,139]
        self.cf_groups =[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]
        self.keep_indices_nf= [  0,   3,   6,   7,  10,  13,  15,  18,  20,  21,  22,  23,  24,
         25,  26,  27,  28,  29,  30, 37,38,  39,  40,  41,  43,  44,  45,  46,  47,  48,  49,  51,  52,
         53,  54,  55,  59,  60,  61,  62,  63,  64,  67,  68,  69,  70,
         71,  72,  73,  75,  76,  77,  78,  79,  80,  81,  83,  84,  85,
         86,  87,  91,  92,  93,  94,  95,  96,  97,  99, 100, 101, 102,
        103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117,
        118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
        131, 132, 133]
        self.keep_indices_cf= [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]
        self.mask_value = mask_value
        self.negate_minor_major_layout = negate_minor_major_layout

    def __call__(self, info):
        # mask = info["node_feat"][:,21:27]<=0
        # if self.negate_minor_major_layout:
        #     info["node_feat"][:,134:][mask] = self.mask_value

        concats = [info["node_feat"][:,self.keep_indices_nf]]
        tmp = info["node_feat"][:,self.nf_groups]
        concats.append(np.where(tmp==-1,6,tmp))
        info["node_feat"] = np.concatenate(concats,1)

        tmp=info["node_config_feat"][:,:,self.cf_groups]
        info["node_config_feat"] = np.where(tmp==-1,6,tmp)
        return info

class RemoveFeaturesAfterFE:
    def __init__(self,):
        self.keep_indices_nf= [  0,   3,   6,   7,  10,  13,  15,  18,  20,  21,  22,  23,  24,
         25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,
         38,  39,  40,  41,  43,  44,  45,  46,  47,  48,  49,  51,  52,
         53,  54,  55,  59,  60,  61,  62,  63,  64,  67,  68,  69,  70,
         71,  72,  73,  75,  76,  77,  78,  79,  80,  81,  83,  84,  85,
         86,  87,  91,  92,  93,  94,  95,  96,  97,  99, 100, 101, 102,
        103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117,
        118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
        131, 132, 133, 134, 135, 136, 137, 138,140,141,142,143,144,145]
        self.keep_indices_cf= [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,18,19,20,21,22,23]

    def __call__(self,info):
        info["node_feat"] = info["node_feat"][:,self.keep_indices_nf]
        info["node_config_feat"] = info["node_config_feat"][:,:,self.keep_indices_cf]
        return info


class FixFeatureNoise:
    def __init__(self,mask_value=-1):
        self.mask_value = mask_value
        
    def __call__(self,info):
        mask = info["node_feat"][:,21:27]<0
        info["node_feat"][:,134:] =np.where(mask,self.mask_value,info["node_feat"][:,134:])
        
        mask =np.cumsum(info["node_feat"][:,36:30:-1],1)[:,::-1]==0
        info["node_feat"][:,31:37] = np.where(mask,self.mask_value,info["node_feat"][:,31:37])
        
        return info
class RemoveFeatures:
    def __init__(self,):
        self.keep_indices_nf= [  0,   3,   6,   7,  10,  13,  15,  18,  20,  21,  22,  23,  24,
         25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,
         38,  39,  40,  41,  43,  44,  45,  46,  47,  48,  49,  51,  52,
         53,  54,  55,  59,  60,  61,  62,  63,  64,  67,  68,  69,  70,
         71,  72,  73,  75,  76,  77,  78,  79,  80,  81,  83,  84,  85,
         86,  87,  91,  92,  93,  94,  95,  96,  97,  99, 100, 101, 102,
        103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117,
        118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
        131, 132, 133, 134, 135, 136, 137, 138]
        self.keep_indices_cf= [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]

    def __call__(self,info):
        info["node_feat"] = info["node_feat"][:,self.keep_indices_nf]
        info["node_config_feat"] = info["node_config_feat"][:,:,self.keep_indices_cf]
        return info

class AmplifyCategorical:
    def __init__(self,amplification_factor=10):
        self.amplification_factor = amplification_factor
        self.cat_indices_nf = [19, 20, 21, 22, 23, 24, 107, 108, 109, 110, 111]

    def __call__(self,info):
        info["node_feat"][:,self.cat_indices_nf] = info["node_feat"][:,self.cat_indices_nf]*self.amplification_factor
        info["node_config_feat"] = info["node_config_feat"]*self.amplification_factor
        return info
    
class RemoveFeaturesTile:
    def __init__(self,):
        self.keep_indices_nf= [  0,   3,   6,   8,   9,  10,  13,  15,  18,  21,  22,  23,  24,
         25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  37,  38,  39,
         43,  44,  45,  46,  47,  51,  52,  53,  54,  55,  59,  60,  61,
         62,  63,  67,  68,  69,  70,  71,  75,  76,  77,  78,  79,  83,
         84,  85,  86,  87,  91,  92,  93,  94,  95,  96,  97,  99, 100,
        101, 102, 103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
        115, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
        129, 130, 131, 132, 134, 135, 136, 137, 138, 139]
        self.keep_indices_cf= [ 0,  1,  2,  3,  4,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 22, 23]

    def __call__(self,info):
        info["node_feat"] = info["node_feat"][:,self.keep_indices_nf]
        info["node_config_feat"] = info["node_config_feat"][:,:,self.keep_indices_cf]
        return info
    
class OneHotFeaturesAfterRemoval:
    def __init__(self, max_size=6,padding_value=-1):
        self.onehot_indices_nf = [19, 20, 21, 22, 23, 24, 107, 108, 109, 110, 111]
        self.other_indices_nf = [x for x in range(112) if x not in self.onehot_indices_nf]
        self.onehot_indices_cf = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self.max_size = max_size
        self.padding_value = padding_value
    def __call__(self,info):
        categorical = []
        for i in self.onehot_indices_nf:
            indices = info["node_feat"][:,i].astype(int)
            onehot_array = np.eye(self.max_size)[indices]
            onehot_array[indices==self.padding_value,:] = 0
            categorical.append(onehot_array)
        categorical.append(info["node_feat"][:,self.other_indices_nf])
        info["node_feat"] = np.concatenate(categorical,axis=1)


        # categorical = []
        # for i in self.onehot_indices_cf:
        #     n,c = info["node_config_feat"][:,:,i].shape
        #     indices = info["node_config_feat"][:,:,i].reshape(-1).astype(int)
        #     onehot_array = np.eye(self.max_size)[indices]
        #     onehot_array[indices==self.padding_value,:] = 0
        #     categorical.append(onehot_array.reshape(n,c,self.max_size))
        # info["node_config_feat"] = np.concatenate(categorical,axis=2)
        return info