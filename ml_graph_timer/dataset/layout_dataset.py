import glob
import numpy as np
from torch.utils.data import Dataset
import os
import torch

class NpzDataset(Dataset):
    """Holds one data partition (train, test, validation) on device memory."""

    def __init__(self, files, min_configs=2, max_configs=-1, normalizers=None,pad_config_nodes=True,pad_config_nodes_val=-1,normalize_runtime=True,random_config_sampling=True,sample_num=None):
        self.files = glob.glob(os.path.join(files, "*.npz"))
        if sample_num:
            self.files = np.random.choice(self.files,min(len(self.files),sample_num))
        self.min_configs = min_configs
        self.max_configs = max_configs
        self.normalize = normalizers is not None
        if normalizers:
            normalizers = np.load(normalizers,allow_pickle=True).item()
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
    def __len__(self):
        return len(self.files)
    def _apply_normalizer(self, feature_matrix, used_columns, min_feat, max_feat, axis=1):
        feature_matrix = feature_matrix[:, used_columns] if axis == 1 else feature_matrix[:, :, used_columns]
        min_feat = min_feat[:, used_columns] if axis == 1 else min_feat[:, :, used_columns]
        max_feat = max_feat[:, used_columns] if axis == 1 else max_feat[:, :, used_columns]
        return (feature_matrix - min_feat) / (max_feat - min_feat)

    def __getitem__(self, index):
        npz_file = np.load(self.files[index])

        graph_id = os.path.splitext(os.path.basename(self.files[index]))[0]

        npz_data = dict(npz_file.items())
        num_configs = npz_data['node_config_feat'].shape[0]

        assert npz_data['node_config_feat'].shape[2] == 18

        npz_data['node_splits'] = npz_data['node_splits'].reshape([-1])
        npz_data['argsort_config_runtime'] = np.argsort(npz_data['config_runtime'])

        if num_configs < self.min_configs:
            print('graph has only %i configurations' % num_configs)

        if self.max_configs > 0 and num_configs > self.max_configs:
            if self.random_config_sampling:
                keep_indices = np.random.choice(np.arange(len(npz_data['node_config_feat'])),min(self.max_configs,len(npz_data['node_config_feat'])))
            else:
                third = self.max_configs // 3
                keep_indices = np.concatenate([
                    npz_data['argsort_config_runtime'][:third],  # Good configs.
                    npz_data['argsort_config_runtime'][-third:],  # Bad configs.
                    np.random.choice(
                        npz_data['argsort_config_runtime'][third:-third],
                        self.max_configs - 2 * third)
                ], axis=0)
            npz_data['node_config_feat'] = npz_data['node_config_feat'][keep_indices]
            npz_data['config_runtime'] = npz_data['config_runtime'][keep_indices]
            
        if self.pad_config_nodes:
            padded_shape = (npz_data['node_config_feat'].shape[0],npz_data["node_feat"].shape[0],npz_data['node_config_feat'].shape[2])
            newconf = np.ones(padded_shape)*self.pad_config_nodes_val
            newconf[:,npz_data["node_config_ids"],:] = npz_data["node_config_feat"]
            npz_data["node_config_feat"] = newconf
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
            node_conf_feats = self._apply_normalizer(node_conf_feats, *self.node_config_feat_norms, axis=2)
        node_conf_feats = torch.tensor(node_conf_feats)

        data_dict = {
            'node_features': node_feats,
            'node_ops': torch.tensor(npz_data["node_opcode"]),
            'edges': torch.tensor(npz_data["edge_index"], dtype=torch.int32),
            'node_config_features': node_conf_feats,
            'node_config_ids': torch.tensor(npz_data["node_config_ids"], dtype=torch.int32),
            'node_splits': torch.tensor(npz_data["node_splits"]),
            'config_runtimes': torch.tensor(npz_data["config_runtime"]),
            'graph_id': graph_id,
            'total_nodes': npz_data['node_feat'].shape[0],
            'total_edges': npz_data['edge_index'].shape[0],
            'total_configs': npz_data['config_runtime'].shape[0],
            'total_config_nodes': npz_data['node_config_ids'].shape[0]
        }
        
        return data_dict

class GraphCollator:
    def __init__(self,max_configs=10,configs_padding=0,runtime_padding=-1,provide_pair_matrix=False):
        self.max_configs = max_configs
        self.configs_padding = configs_padding
        self.runtime_padding = runtime_padding
        self.provide_pair_matrix = provide_pair_matrix
    def _process_node_config_features(self, config_features):
        # Trim or pad the "configs" dimension
        if config_features.shape[0] > self.max_configs:
            config_features = config_features[:self.max_configs]  # Trim
        elif config_features.shape[0] < self.max_configs:
            padding_size = self.max_configs - config_features.shape[0]
            padding = torch.full((padding_size, *config_features.shape[1:]), self.configs_padding)
            config_features = torch.cat([config_features, padding], dim=0)  # Pad
        return config_features
    
    def _process_config_runtimes(self, runtimes):
        # Trim or pad the "configs" dimension for runtimes
        if runtimes.shape[0] > self.max_configs:
            runtimes = runtimes[:self.max_configs]  # Trim
        elif runtimes.shape[0] < self.max_configs:
            padding_size = self.max_configs - runtimes.shape[0]
            padding = torch.full((padding_size,), self.runtime_padding)
            runtimes = torch.cat([runtimes, padding], dim=0)  # Pad
        return runtimes
    
    def calculate_pair_gt_tensor(self,T):
        batch_size, configs = T.shape

        # Extend the last and second last dimension of T for broadcasting
        T_ext1 = T.unsqueeze(-1)   # shape becomes [batch_size, configs, 1]
        T_ext2 = T.unsqueeze(-2)   # shape becomes [batch_size, 1, configs]

        # Initialize F with 2s (since that's one of the conditions)
        F = 2 * torch.ones(batch_size, configs, configs)

        # Conditions to create the tensor F
        F[T_ext1 > T_ext2] = 0
        F[T_ext1 < T_ext2] = 1

        mask1 = (T_ext1 < 0).expand_as(F)
        mask2 = (T_ext2 < 0).expand_as(F)
        
        F[mask1 | mask2] = -1

        # Setting diagonal to -1
        for i in range(configs):
            F[:, i, i] = -1

        return F

    def __call__(self, batch):
        """
        batch: List of dictionaries. Each dictionary corresponds to data for a single graph.
        """
        
        # Node features
        node_feats = []
        edges = []
        nodeops = []
        batch_no=[]
        for i,item in enumerate(batch):
            pad_trim_conf = self._process_node_config_features(item["node_config_features"])
            for ptc in pad_trim_conf:
                node_feats.append(torch.cat([item["node_features"],ptc], dim=1))
                edges.append(item["edges"])
                nodeops.append(item["node_ops"])
                batch_no.append(i)
        # Edges
        node_separation = torch.cumsum(torch.tensor([f.shape[0] for f in node_feats]), dim=0)
        cum_node_counts = torch.cat([torch.tensor([0]), node_separation[:-1]])  # cumulative node counts
        edges_list = [item + cnt for item, cnt in zip(edges, cum_node_counts)]
        edges = torch.cat(edges_list, dim=0)

        # Config runtimes
        config_runtimes_list = [self._process_config_runtimes(item["config_runtimes"]) for item in batch]
        config_runtimes = torch.stack(config_runtimes_list)  # Concatenate along the batch dimension

        if self.provide_pair_matrix:
            optimization_matrix = self.calculate_pair_gt_tensor(config_runtimes)
        else:
            optimization_matrix = torch.tensor([0])
        return {
            "node_features": torch.cat(node_feats,dim=0).float(),
            "node_separation": node_separation.long(),
            "node_ops": torch.cat(nodeops, dim=0).long(),
            "edges": edges.permute(1,0).long(),
            "config_runtimes": config_runtimes.float(),
            "batches": torch.tensor(batch_no).long(),
            "optimization_matrix": optimization_matrix.long()
        }