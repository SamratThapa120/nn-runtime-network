import torch
from torch_geometric.nn.models import GAT,GraphSAGE
import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn

import torch
import torch.nn as nn

from .sageconv import ResidualBlock,GraphwiseLayerNorm
    

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
    attention_blocks: int = 0

    drop_rate: float = 0.1
    attention_dropout: float = 0.1
    num_heads: int = 0

    is_pair_modeling: bool = False
    project_after_graph_encoder: bool = False 
    graphsage_aggr: str = "mean"
    graphsage_normalize: bool = True
    graphsage_project: bool = False
    return_positive_values: bool = False

    model_type: str = "gsage"
    post_encoder_blocks: int = 0

    categorical_cols_start: int = 101 
    embeddings_size: int = 7
    embeddings_dim:  int = 4

class LayoutGraphModel(torch.nn.Module):
    def __init__(self,arguments: GraphModelArugments):
        super().__init__()
        self.arguments = arguments
        self.opcode_embeddings = torch.nn.Embedding(arguments.num_opcodes,arguments.opcode_dim)

        self.categorical_embeddings = torch.nn.Embedding(arguments.embeddings_size,arguments.embeddings_dim)

        if arguments.graphsage_layers<1:
            self.graph_encoder=None
        elif arguments.model_type=="gsage" :
            self.graph_encoder = GraphSAGE(in_channels=arguments.graphsage_in,
                                        hidden_channels=arguments.graphsage_hidden,
                                        num_layers=arguments.graphsage_layers,
                                        dropout=arguments.graphsage_dropout,
                                        normalize=arguments.graphsage_normalize,
                                        project = arguments.graphsage_project,
                                        aggr=arguments.graphsage_aggr,
                                        )
        elif arguments.model_type=="gat":
            self.graph_encoder = GAT(in_channels=arguments.graphsage_in,
                                        hidden_channels=arguments.graphsage_hidden,
                                        num_layers=arguments.graphsage_layers,
                                        dropout=arguments.graphsage_dropout,
                                        normalize=arguments.graphsage_normalize,
                                        project = arguments.graphsage_project,
                                        aggr=arguments.graphsage_aggr,
                                        )
        self.node_features_mlp = torch.nn.ModuleList([
            torch.nn.Linear(arguments.node_feature_dim,arguments.node_feature_dim*arguments.node_feature_expand),
            torch.nn.LeakyReLU(inplace=True),
            GraphwiseLayerNorm(arguments.node_feature_dim*arguments.node_feature_expand),
            torch.nn.Linear(arguments.node_feature_dim*arguments.node_feature_expand,arguments.graphsage_in),
            torch.nn.LeakyReLU(inplace=True),
            GraphwiseLayerNorm(arguments.graphsage_in),
        ])
        self.embed_drop = torch.nn.Dropout(arguments.embedding_dropout)
        self.final_classifier = torch.nn.Sequential(
            torch.nn.Dropout(arguments.final_dropout),
            torch.nn.Linear(arguments.graphsage_hidden,1)
        )
        self.categorical_cols_start = self.arguments.categorical_cols_start

    def forward(self,node_features, node_config_features, node_separation, node_ops, edges, batches,*args,**kwargs):
        opcode_embed = self.embed_drop(self.opcode_embeddings(node_ops))
        categorical_nf= self.categorical_embeddings(node_features[:,self.categorical_cols_start:].long()).flatten(-2)
        node_features = node_features[:,:self.categorical_cols_start]
        categorical_cf = self.categorical_embeddings(node_config_features.long()).flatten(-2)
            
        x = torch.cat([categorical_nf, categorical_cf, node_features, opcode_embed], dim=1)

        for layer in self.node_features_mlp:
            if isinstance(layer,GraphwiseLayerNorm):
                x = layer(x,node_separation)
            else:
                x = layer(x)

        if self.graph_encoder is not None:
            x = self.graph_encoder(x,edges)

        if self.arguments.project_after_graph_encoder:
            x = self.final_classifier(x)
            if self.arguments.return_positive_values:
                x = x.abs()
        start_idx = 0
        aggregated = [[] for _ in torch.unique(batches)]
        for ns,b in zip(node_separation,batches):
            end_idx = ns
            aggregated[b].append(torch.sum(x[start_idx:end_idx], dim=0))
            start_idx = end_idx
        aggregated = torch.stack([torch.stack(x) for x in aggregated])
        
        if not self.arguments.project_after_graph_encoder:
            aggregated = self.final_classifier(aggregated)
            if self.arguments.return_positive_values:
                aggregated = aggregated.abs()
        return torch.squeeze(aggregated,2)
        

