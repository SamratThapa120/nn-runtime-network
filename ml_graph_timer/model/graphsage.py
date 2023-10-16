import torch
from torch_geometric.nn.models import GraphSAGE
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GraphModelArugments:
    num_opcodes: int = 120
    opcode_dim: int = 64
    graphsage_in: int = 128
    graphsage_hidden: int = 256
    graphsage_layers: int = 3
    graphsage_dropout: int = 0.1

class L2NormalizationLayer(torch.nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class LayoutGraphModel(torch.nn.Module):
    def __init__(self,arguments: GraphModelArugments):
        super().__init__()
        self.opcode_embeddings = torch.nn.Embedding(arguments.num_opcodes,arguments.opcode_dim)
        self.graph_encoder = GraphSAGE(in_channels=arguments.graphsage_in,
                                       hidden_channels=arguments.graphsage_hidden,
                                       num_layers=arguments.graphsage_layers,
                                       dropout=arguments.graphsage_dropout,
                                       norm=L2NormalizationLayer(dim=1),
                                       act=torch.nn.LeakyReLU()
                                       )
        
    def forward(self,node_features, node_separation, node_ops, edges, batches):
        opcode_embed = self.opcode_embeddings(node_ops)
        node_features = torch.cat([node_features,opcode_embed],dim=1)
        

