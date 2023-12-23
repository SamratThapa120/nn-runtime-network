import torch
from torch_geometric.nn.models import GAT,GraphSAGE
from torch_geometric.nn.conv import SAGEConv,GATv2Conv
import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn

import torch
import torch.nn as nn

from .sageconv import ResidualBlock,GraphwiseLayerNorm,LNormModule,PairNorm


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

    drop_rate: float = 0.1
    num_heads: int = 0

    is_pair_modeling: bool = False
    project_after_graph_encoder: bool = False 
    graphsage_aggr: str = "mean"
    graphsage_normalize: bool = False
    graphsage_project: bool = False
    return_positive_values: bool = False

    post_encoder_blocks: int = 0
    gst_drop_rate: float = 0.0

class LayoutGraphModel(torch.nn.Module):
    def __init__(self,arguments: GraphModelArugments):
        super().__init__()
        self.arguments = arguments
        self.opcode_embeddings = torch.nn.Embedding(arguments.num_opcodes,arguments.opcode_dim)
        self.gst_enabled = arguments.gst_drop_rate > 0
        if arguments.graphsage_layers<1:
            self.graph_encoder=None

        self.gcn_layers = torch.nn.ModuleList() 
        self.norm_layers = torch.nn.ModuleList() 
        self.activation = torch.nn.ModuleList() 
        self.post_encoder_mlp = nn.ModuleList()
        for i in range(self.arguments.graphsage_layers):
            self.gcn_layers.append(
                SAGEConv(in_channels=arguments.graphsage_in,
                    hidden_channels=arguments.graphsage_in,
                    out_channels=arguments.graphsage_in,
                    aggr=arguments.graphsage_aggr,
                )
            )
            self.activation.append(
                torch.nn.LeakyReLU(),
            )
            self.norm_layers.append(
                PairNorm(),
            )
            self.post_encoder_mlp.append(
                # ResidualBlock(
                #     input_dim=arguments.graphsage_in,
                #     output_dim=arguments.graphsage_in,
                #     expand_dim=arguments.graphsage_in * arguments.node_feature_expand,
                #     dropout_rate=arguments.node_feature_dropout
                # )
                torch.nn.Identity()
            )

        self.node_features_mlp = torch.nn.ModuleList([
            torch.nn.Linear(arguments.node_feature_dim,arguments.node_feature_dim*arguments.node_feature_expand),
            torch.nn.LeakyReLU(),
            GraphwiseLayerNorm(arguments.node_feature_dim*arguments.node_feature_expand),
            torch.nn.Linear(arguments.node_feature_dim*arguments.node_feature_expand,arguments.graphsage_in),
            torch.nn.LeakyReLU(),
            GraphwiseLayerNorm(arguments.graphsage_in),
            torch.nn.Linear(arguments.graphsage_in,arguments.graphsage_in),
            torch.nn.LeakyReLU()
        ])


        self.embed_drop = torch.nn.Dropout(arguments.embedding_dropout)
        self.final_classifier = torch.nn.Sequential(
            torch.nn.Dropout(arguments.final_dropout),
            torch.nn.Linear(arguments.graphsage_hidden,1)
        )

    def forward(self,node_features, node_config_features, node_separation, node_ops, edges, batches,*args,**kwargs):
        opcode_embed = self.embed_drop(self.opcode_embeddings(node_ops))
        x = torch.cat([node_features,node_config_features,opcode_embed],dim=1)

        for layer in self.node_features_mlp:
            if isinstance(layer,GraphwiseLayerNorm):
                x = layer(x,node_separation)
            else:
                x = layer(x)

        for gcn,activ,norm,block in zip(self.gcn_layers,self.activation,self.norm_layers,self.post_encoder_mlp):
            x = norm(x)
            x = gcn(x,edges)
            x = activ(x)
            # x = block(x,node_separation)
        if self.gst_enabled:
            mask = torch.rand(len(x),device=x.device)<self.arguments.gst_drop_rate
            x[mask] = x[mask].detach()

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
        

