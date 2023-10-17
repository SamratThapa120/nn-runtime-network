import torch
from torch_geometric.nn.models import GraphSAGE
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GraphModelArugments:
    num_opcodes: int = 120
    opcode_dim: int = 64
    node_feature_dim: int = 126+opcode_dim
    node_feature_dropout: float=0.1
    node_feature_expand: int = 2

    graphsage_in: int = 256
    graphsage_hidden: int = 512
    graphsage_layers: int = 3
    graphsage_dropout: float = 0.1

    final_dropout: float = 0.1
    embedding_dropout: float = 0.1
    
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
        self.arguments = arguments
        self.opcode_embeddings = torch.nn.Embedding(arguments.num_opcodes,arguments.opcode_dim)
        self.graph_encoder = GraphSAGE(in_channels=arguments.graphsage_in,
                                       hidden_channels=arguments.graphsage_hidden,
                                       num_layers=arguments.graphsage_layers,
                                       dropout=arguments.graphsage_dropout,
                                       norm=L2NormalizationLayer(dim=1),
                                       act=torch.nn.LeakyReLU()
                                       )
        self.node_features_mlp = torch.nn.Sequential(
            torch.nn.Linear(arguments.node_feature_dim,arguments.node_feature_dim*arguments.node_feature_expand),
            torch.nn.Dropout(arguments.node_feature_dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(arguments.node_feature_dim*arguments.node_feature_expand,arguments.graphsage_in)
        )
        self.embed_drop = torch.nn.Dropout(arguments.embedding_dropout)
        self.final_classifier = torch.nn.Sequential(
            torch.nn.Dropout(arguments.final_dropout),
            torch.nn.Linear(arguments.graphsage_hidden,1)
        )

    def forward(self,node_features, node_separation, node_ops, edges, batches,*args,**kwargs):
        opcode_embed = self.embed_drop(self.opcode_embeddings(node_ops))
        x = torch.cat([node_features,opcode_embed],dim=1)
        x = self.node_features_mlp(x)
        x = self.graph_encoder(x,edges)
        start_idx = 0
        aggregated = [[] for _ in torch.unique(batches)]
        for ns,b in zip(node_separation,batches):
            end_idx = ns
            aggregated[b].append(torch.sum(x[start_idx:end_idx], dim=0))
            start_idx = end_idx
        aggregated = torch.stack([torch.stack(x) for x in aggregated])
        aggregated = self.final_classifier(aggregated).squeeze()
        return aggregated
        

