import numpy as np

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