import numpy as np

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