import glob
import numpy as np
from torch.utils.data import Dataset
import os
import torch
import numpy as np

def histogram_equalized_sampling(data, max_configs):
    num_bins = max_configs // 2

    # Compute histogram bins and bin indices for each data point
    counts, bin_edges = np.histogram(data, bins=num_bins)
    indices = np.digitize(data, bin_edges[:-1]) - 1  # Digitize into zero-based bin indices

    # Filter out empty bins
    non_empty_bins = np.nonzero(counts)[0]
    counts = counts[non_empty_bins]

    # Calculate the number of samples to choose from each bin based on counts
    samples_per_bin = np.ceil(max_configs / len(non_empty_bins)).astype(int)

    # The array to collect the selected indices
    selected_indices = np.array([], dtype=int)

    # Sample from each non-empty bin
    for bin_idx in non_empty_bins:
        in_bin_indices = np.where(indices == bin_idx)[0]
        if len(in_bin_indices) > samples_per_bin:
            chosen_indices = np.random.choice(in_bin_indices, size=samples_per_bin, replace=False)
        else:
            chosen_indices = in_bin_indices

        selected_indices = np.concatenate([selected_indices, chosen_indices])

    # Shuffle the selected indices to avoid any potential ordering bias
    np.random.shuffle(selected_indices)

    # If we have too many indices due to ceiling operation, trim the excess
    if len(selected_indices) > max_configs:
        selected_indices = selected_indices[:max_configs]

    # If we still need more indices, we will add them from the bins with most counts
    if len(selected_indices) < max_configs:
        additional_samples_needed = max_configs - len(selected_indices)
        # Get bins sorted by the number of counts (desc) and select the top ones needed for more samples
        bins_sorted_by_count = np.argsort(-counts)
        while additional_samples_needed > 0 and len(bins_sorted_by_count) > 0:
            # Select the bin with the most samples left
            bin_for_additional_samples = bins_sorted_by_count[0]
            # Get the remaining indices from the most populous bin
            remaining_indices = np.setdiff1d(np.where(indices == bin_for_additional_samples)[0], selected_indices)
            if len(remaining_indices) == 0:
                # If the bin is exhausted, move to the next one
                bins_sorted_by_count = bins_sorted_by_count[1:]
                continue

            # Calculate how many samples to take from this bin
            samples_to_take = min(len(remaining_indices), additional_samples_needed)
            additional_indices = np.random.choice(remaining_indices, size=samples_to_take, replace=False)

            # Append these additional samples to our selected indices
            selected_indices = np.concatenate([selected_indices, additional_indices])

            # Update the number of additional samples needed
            additional_samples_needed -= samples_to_take

    return selected_indices     

import networkx as nx
import nxmetis

def partition_graph(edges, nodes, max_nodes=1000):
    edge_list = [(int(edge[0]), int(edge[1])) for edge in edges.t().tolist()]
    G = nx.Graph()
    G.add_edges_from(edge_list)
    num_partitions = max(1, nodes // max_nodes)
    partitions = nxmetis.partition(G, num_partitions)
    subgraph_ids = torch.zeros(nodes, dtype=torch.int64)
    for i, partition in enumerate(partitions[1]):
        for node in partition:
            subgraph_ids[node] = i
    return subgraph_ids
class NpzDataset(Dataset):
    """Holds one data partition (train, test, validation) on device memory."""

    def __init__(self, files, min_configs=2, 
                 max_configs=-1, normalizers=None,pad_config_nodes=True,
                 pad_config_nodes_val=-1,normalize_runtime=False,
                 random_config_sampling=True,sample_num=None,
                 isvalid=False,normalizer=None,is_tile=False,transforms=None,histogram_sampling=False,window_sampling=0,
                 equidistance=False,gst_training=0):
        self.gst_training = gst_training
        self.equidistance = equidistance
        self.window_sampling = window_sampling
        self.histogram_sampling = histogram_sampling
        self.transforms = transforms
        self.isvalid = isvalid
        self.files = glob.glob(os.path.join(files, "*.npz"))
        self.model_types = [f.split("/")[-1].split("___valid")[0] for f in self.files]
        if sample_num:
            self.files = np.random.choice(self.files,min(len(self.files),sample_num))
        self.min_configs = min_configs
        self.max_configs = max_configs
        self.normalize = normalizers is not None
        self.normalizer_type = normalizer
        if normalizers:
            normalizers = np.load(normalizers,allow_pickle=True).item()
            if self.normalizer_type=="z_score":
                mean, std = normalizers["mean_node_feat"].astype(np.float32), normalizers["std_node_feat"].astype(np.float32)
                self.node_feat_norms = (
                    (std != 0)& (~np.isnan(std)), np.expand_dims(mean,0), np.expand_dims(std,0)
                )
                mean, std = normalizers["mean_node_config_feat"].astype(np.float32), normalizers["std_node_config_feat"].astype(np.float32)
                self.node_config_feat_norms = (
                    (std != 0)&(~np.isnan(std)),np.expand_dims(mean,[0,1]), np.expand_dims(std,[0,1])
                )
            else:
                maxf, minf = normalizers["max_node_feat"], normalizers["min_node_feat"]
                self.node_feat_norms = (
                    (minf != maxf), np.expand_dims(minf,0), np.expand_dims(maxf,0)
                )
                maxf, minf = normalizers["max_node_config_feat"], normalizers["min_node_config_feat"]
                self.node_config_feat_norms = (
                    (minf != maxf),np.expand_dims(minf,[0,1]), np.expand_dims(maxf,[0,1])
                )
        self.pad_config_nodes = pad_config_nodes
        self.pad_config_nodes_val = pad_config_nodes_val
        self.normalize_runtime = normalize_runtime
        self.random_config_sampling = random_config_sampling
        self.is_tile = is_tile
    def __len__(self):
        return len(self.files)
    def _apply_normalizer(self, feature_matrix, used_columns, min_feat, max_feat, axis=1):
        if self.normalizer_type=="z_score":
            feature_matrix = feature_matrix[:, used_columns] if axis == 1 else feature_matrix[:, :, used_columns]
            mean = min_feat[:, used_columns] if axis == 1 else min_feat[:, :, used_columns]
            std = max_feat[:, used_columns] if axis == 1 else max_feat[:, :, used_columns]
            return (feature_matrix - mean) / std
        else:
            feature_matrix = feature_matrix[:, used_columns] if axis == 1 else feature_matrix[:, :, used_columns]
            min_feat = min_feat[:, used_columns] if axis == 1 else min_feat[:, :, used_columns]
            max_feat = max_feat[:, used_columns] if axis == 1 else max_feat[:, :, used_columns]
            return (feature_matrix - min_feat) / (max_feat - min_feat)
        
    def load_files(self,index):
        return dict(np.load(self.files[index]))
    
    def __getitem__(self, index):
        npz_data = self.load_files(index)
        graph_id = os.path.splitext(os.path.basename(self.files[index]))[0]

        if self.is_tile:
            npz_data['node_config_feat'] =  np.stack([npz_data["config_feat"]]*len(npz_data["node_feat"]),axis=1)
            npz_data['node_config_ids'] =  np.arange(len(npz_data["node_feat"]))
        num_configs = npz_data['config_runtime'].shape[0]

        if num_configs < self.min_configs:
            print('graph has only %i configurations' % num_configs)

        if self.max_configs > 0 and num_configs > self.max_configs:
            if self.isvalid:
                sorted_times = np.argsort(npz_data['config_runtime'])
                indices = np.linspace(0, len(sorted_times) - 1, self.max_configs, dtype=int)
                keep_indices = sorted_times[indices]
            elif self.window_sampling>0:
                sorted_times = np.argsort(npz_data['config_runtime'])
                window_size = self.max_configs // self.window_sampling
                chosen_idxs = np.random.choice(np.arange(0,len(npz_data['config_runtime']),window_size),self.window_sampling)
                keep_indices = np.concatenate([sorted_times[i:i+window_size] for i in chosen_idxs])
            elif self.histogram_sampling:
                keep_indices = histogram_equalized_sampling(npz_data['config_runtime'],self.max_configs)
            elif self.random_config_sampling:
                keep_indices = np.random.choice(np.arange(len(npz_data['node_config_feat'])),min(self.max_configs,len(npz_data['node_config_feat'])))
            else:
                npz_data['argsort_config_runtime'] = np.argsort(npz_data['config_runtime'])
                third = self.max_configs // 3
                keep_indices = np.concatenate([
                    npz_data['argsort_config_runtime'][:third],  # Good configs.
                    npz_data['argsort_config_runtime'][-third:],  # Bad configs.
                    np.random.choice(
                        npz_data['argsort_config_runtime'][third:-third],
                        self.max_configs - 2 * third)
                ], axis=0)
            np.random.shuffle(keep_indices)
            npz_data['node_config_feat'] = npz_data['node_config_feat'][keep_indices]
            npz_data['config_runtime'] = npz_data['config_runtime'][keep_indices]
        if self.transforms:
            npz_data = self.transforms(npz_data)
        if self.pad_config_nodes:
            padded_shape = (npz_data['node_config_feat'].shape[0],npz_data["node_feat"].shape[0],npz_data['node_config_feat'].shape[2])
            newconf = np.ones(padded_shape)*self.pad_config_nodes_val
            newconf[:,npz_data["node_config_ids"],:] = npz_data["node_config_feat"]
            npz_data["node_config_feat"] = newconf

        npz_data['config_runtime'] = npz_data['config_runtime']/1e6
        if self.normalize_runtime:
            mmin,mmax = npz_data['config_runtime'].min(),npz_data['config_runtime'].max()
            if mmin==mmax:
                mmin=0
            npz_data['config_runtime'] = (npz_data['config_runtime']-mmin)/(mmax-mmin)
        node_feats = npz_data["node_feat"]
        if self.normalize:
            node_feats = self._apply_normalizer(node_feats, *self.node_feat_norms, axis=1)
        node_feats = torch.tensor(node_feats)

        node_conf_feats = npz_data["node_config_feat"]
        if self.normalize:
            # node_conf_feats = node_conf_feats[:, :, self.node_config_feat_norms[0]]
            # mmin,mmax = node_conf_feats.min(axis=0,keepdims=True).max(axis=1,keepdims=True),node_conf_feats.max(axis=0,keepdims=True).max(axis=1,keepdims=True)
            # mask = (mmin!=mmax).reshape(-1)
            # node_conf_feats[:,:,mask] = 0.0
            # node_conf_feats[:,:,mask] = (node_conf_feats[:,:,mask]-mmin[:,:,mask])/(mmax[:,:,mask]-mmin[:,:,mask])
            node_conf_feats = self._apply_normalizer(node_conf_feats, *self.node_config_feat_norms, axis=2)
        node_conf_feats = torch.tensor(node_conf_feats)

        if self.equidistance:
            npz_data["config_runtime"] = np.argsort(npz_data["config_runtime"])            
        data_dict = {
            'node_features': node_feats,
            'node_ops': torch.tensor(npz_data["node_opcode"]),
            'edges': torch.tensor(npz_data["edge_index"], dtype=torch.int32),
            'node_config_features': node_conf_feats,
            'node_config_ids': torch.tensor(npz_data["node_config_ids"], dtype=torch.int32),
            'config_runtimes': torch.tensor(npz_data["config_runtime"]),
            'graph_id': graph_id,
            'total_nodes': npz_data['node_feat'].shape[0],
            'total_edges': npz_data['edge_index'].shape[0],
            'total_configs': npz_data['config_runtime'].shape[0],
            'total_config_nodes': npz_data['node_config_ids'].shape[0]
        }
        if self.gst_training>0:
            if node_feats.shape[0]<self.gst_training*2:
                data_dict["gst_subgraphs"] = torch.zeros(node_feats.shape[0])
            else:
                data_dict["gst_subgraphs"] = partition_graph(data_dict["edges"].T,node_feats.shape[0],self.gst_training)
        return data_dict
class CopyFeatureDataset(NpzDataset):

    def __getitem__(self, index):
        data = super().__getitem__(index)
        inpseq = data["node_config_features"][:,data["edges"][:,0],:6]
        outseq = data["node_config_features"][:,data["edges"][:,1],6:12]
        c,n,d = data["node_config_features"].shape

        new_feat = torch.zeros(c,n,1)
        new_feat[:,data["edges"][:,1],0] = torch.all(inpseq==outseq,dim=2).float()

        data["node_config_features"] = torch.cat([data["node_config_features"],new_feat],dim=2)
        return data

class GraphCollator:
    def __init__(self,max_configs=10,configs_padding=-1,runtime_padding=-1,mask_invalid_value=-1,provide_pair_matrix=False,clip_max=False):
        self.configs_padding = configs_padding
        self.runtime_padding = runtime_padding
        self.provide_pair_matrix = provide_pair_matrix
        self.mask_invalid_value = mask_invalid_value
        self.clip_max = clip_max
        self.max_configs = max_configs
        
    def _process_node_config_features(self, config_features,max_configs):
        # Trim or pad the "configs" dimension
        if config_features.shape[0] > max_configs:
            config_features = config_features[:max_configs]  # Trim
        elif config_features.shape[0] < max_configs:
            padding_size = max_configs - config_features.shape[0]
            padding = torch.full((padding_size, *config_features.shape[1:]), self.configs_padding)
            config_features = torch.cat([config_features, padding], dim=0)  # Pad
        return config_features
    
    def _process_config_runtimes(self, runtimes,max_configs):
        # Trim or pad the "configs" dimension for runtimes
        if runtimes.shape[0] > max_configs:
            runtimes = runtimes[:max_configs]  # Trim
        elif runtimes.shape[0] < max_configs:
            padding_size = max_configs - runtimes.shape[0]
            padding = torch.full((padding_size,), self.runtime_padding)
            runtimes = torch.cat([runtimes, padding], dim=0)  # Pad
        return runtimes
    
    def calculate_pair_gt_tensor_classification(self,T):
        batch_size, configs = T.shape

        # Extend the last and second last dimension of T for broadcasting
        T_ext1 = T.unsqueeze(-1)   # shape becomes [batch_size, configs, 1]
        T_ext2 = T.unsqueeze(-2)   # shape becomes [batch_size, 1, configs]

        # Initialize F with 2s (since that's one of the conditions)
        F = torch.ones(batch_size, configs, configs)

        # Conditions to create the tensor F
        F[T_ext1 > T_ext2] = 2
        F[T_ext1 < T_ext2] = 0

        mask1 = (T_ext1 == self.runtime_padding).expand_as(F)
        F[mask1] = self.mask_invalid_value

        # Setting diagonal to -1
        for i in range(configs):
            F[:, i, i] = self.mask_invalid_value

        return F
    def calculate_pair_gt_tensor(self,T):

        # Extend dimensions of T for broadcasting
        T_ext1 = T.unsqueeze(-1)   # shape becomes [batch_size, configs, 1]
        T_ext2 = T.unsqueeze(-2)   # shape becomes [batch_size, 1, configs]

        # Compute F according to the given formula
        F = T_ext1 - T_ext2

        # Masking conditions
        mask_invalid = (T_ext1 == self.runtime_padding) | (T_ext2 == self.runtime_padding)

        F[mask_invalid] = self.mask_invalid_value
        for i in range(F.shape[0]):
            F[i].fill_diagonal_(self.mask_invalid_value)
        return F

    def __call__(self, batch):
        """
        batch: List of dictionaries. Each dictionary corresponds to data for a single graph.
        """
        # Node features
        node_feats = []
        node_conf_feats = []

        edges = []
        nodeops = []
        batch_no=[]
        subg_no = []
        if self.clip_max:
            max_configs = max([len(item["config_runtimes"]) for item in batch])
        else:
            max_configs = self.max_configs
        for i,item in enumerate(batch):
            pad_trim_conf = self._process_node_config_features(item["node_config_features"],max_configs)
            for ptc in pad_trim_conf:
                node_feats.append(item["node_features"])
                node_conf_feats.append(ptc)
                edges.append(item["edges"])
                nodeops.append(item["node_ops"])
                batch_no.append(i)
                if "gst_subgraphs" in item:
                    subg_no.append(item["gst_subgraphs"])
        # Edges
        node_separation = torch.cumsum(torch.tensor([f.shape[0] for f in node_feats]), dim=0)
        cum_node_counts = torch.cat([torch.tensor([0]), node_separation[:-1]])  # cumulative node counts
        edges_list = [item + cnt for item, cnt in zip(edges, cum_node_counts)]
        edges = torch.cat(edges_list, dim=0)

        # Config runtimes
        config_runtimes_list = [self._process_config_runtimes(item["config_runtimes"],max_configs) for item in batch]
        config_runtimes = torch.stack(config_runtimes_list)  # Concatenate along the batch dimension

        if self.provide_pair_matrix:
            optimization_matrix = self.calculate_pair_gt_tensor(config_runtimes)
        else:
            optimization_matrix = torch.tensor([0])
        data = {
            "node_features": torch.cat(node_feats,dim=0).float(),
            "node_config_features": torch.cat(node_conf_feats,dim=0).float(),
            "node_separation": node_separation.long(),
            "node_ops": torch.cat(nodeops, dim=0).long(),
            "edges": edges.permute(1,0).long(),
            "config_runtimes": config_runtimes.float(),
            "batches": torch.tensor(batch_no).long(),
            "optimization_matrix": optimization_matrix.float()
        }
        if len(subg_no)>0:
            subgraph_separation = torch.cumsum(torch.tensor([f.max()+1 for f in subg_no]), dim=0)
            cum_node_counts = torch.cat([torch.tensor([0]), subgraph_separation[:-1]])  # cumulative node counts
            data["subgraphs"] = torch.cat([item + cnt for item, cnt in zip(subg_no, cum_node_counts)],dim=0)
        return data
class RandomDefaultFusedNpzDataset(NpzDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.uniques = np.unique([f.split("/")[-1] for f in self.files]).tolist()
    def __len__(self):
        return len(self.uniques)
    def load_files(self,index):
        grouped_files = [dict(np.load(f)) for f in self.files if self.uniques[index]==f.split("/")[-1]]
        example = grouped_files[0]
        example["node_config_feat"] = np.concatenate([n["node_config_feat"] for n in grouped_files],0)
        example["config_runtime"] = np.concatenate([n["config_runtime"] for n in grouped_files],0)
        return example
    
class StreamingCollator(GraphCollator):
    def __init__(self,batch_size=128,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.batch_size = batch_size

    def __call__(self, info):
        assert len(info)==1
        info = info[0]
        COPY_KEYS = ['node_features','node_ops','edges','node_config_ids','graph_id']
        total_configs = len(info["config_runtimes"])

        batches = []
        for i in range(0,total_configs,self.max_configs):
            if len(batches)==self.batch_size:
                yield super().__call__(batches)
                batches = []
            tdata = {k:info[k] for k in COPY_KEYS}
            tdata["node_config_features"] = info["node_config_features"][i:(i+self.max_configs)]
            tdata["config_runtimes"] = info["config_runtimes"][i:(i+self.max_configs)]
            batches.append(tdata)
        if i+self.max_configs<total_configs:
            tdata = {k:info[k] for k in COPY_KEYS}
            tdata["node_config_features"] = info["node_config_features"][i:]
            tdata["config_runtimes"] = info["config_runtimes"][i:]
            batches.append(tdata) 
        if len(batches)>0:
            yield super().__call__(batches)
