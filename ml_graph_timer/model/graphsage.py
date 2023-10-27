import torch
from torch_geometric.nn.models import GraphSAGE
import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn
    
class L2NormalizationLayer(torch.nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class TransformerBlock(nn.Module):
    def __init__(self,dim=256, num_heads=4, expand=2, attn_dropout=0.1, drop_rate=0.1, activation=Swish()):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout,batch_first=True)
        self.dropout1 = nn.Dropout(drop_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim*expand, bias=False)
        self.linear2 = nn.Linear(dim*expand, dim, bias=False)
        self.dropout2 = nn.Dropout(drop_rate)
        self.activation = activation

    def forward(self, inputs):
        x = self.norm1(inputs)
        x, _ = self.attn(x, x, x)
        x = self.dropout1(x)
        x = x + inputs
        attn_out = x

        x = self.norm2(x)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout2(x)
        x = x + attn_out
        return x
    
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

class LayoutGraphModel(torch.nn.Module):
    def __init__(self,arguments: GraphModelArugments):
        super().__init__()
        self.arguments = arguments
        self.opcode_embeddings = torch.nn.Embedding(arguments.num_opcodes,arguments.opcode_dim)
        self.graph_encoder = GraphSAGE(in_channels=arguments.graphsage_in,
                                       hidden_channels=arguments.graphsage_hidden,
                                       num_layers=arguments.graphsage_layers,
                                       dropout=arguments.graphsage_dropout,
                                       normalize=arguments.graphsage_normalize,
                                       project = arguments.graphsage_project,
                                       aggr=arguments.graphsage_aggr,
                                       )
        self.node_features_mlp = torch.nn.Sequential(
            torch.nn.Linear(arguments.node_feature_dim,arguments.node_feature_dim*arguments.node_feature_expand),
            torch.nn.Dropout(arguments.node_feature_dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(arguments.node_feature_dim*arguments.node_feature_expand,arguments.graphsage_in),
            L2NormalizationLayer(1)
        )
        
        if arguments.attention_blocks>0:
            self.attention_module = torch.nn.Sequential(
                *[
                    TransformerBlock(arguments.graphsage_hidden,arguments.num_heads,attn_dropout=arguments.attention_dropout,drop_rate=arguments.drop_rate) for _ in range(arguments.attention_blocks)
                ]
            )
        else:
            self.attention_module = None
        self.embed_drop = torch.nn.Dropout(arguments.embedding_dropout)
        # self.aggregation_norm = torch.nn.LayerNorm(arguments.graphsage_hidden)
        # self.norm_l = L2NormalizationLayer(2)
        self.final_classifier = torch.nn.Sequential(
            torch.nn.Dropout(arguments.final_dropout),
            torch.nn.Linear(arguments.graphsage_hidden,1)
        )

    def forward(self,node_features, node_config_features, node_separation, node_ops, edges, batches,*args,**kwargs):
        opcode_embed = self.embed_drop(self.opcode_embeddings(node_ops))
        node_features = torch.concat([node_features,node_config_features],dim=1)
        x = torch.cat([node_features,opcode_embed],dim=1)
        x = self.node_features_mlp(x)
        x = self.graph_encoder(x,edges)
        if self.arguments.project_after_graph_encoder:
            # aggregated = self.aggregation_norm(aggregated)
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
        # if self.attention_module is not None:
        #     aggregated = self.attention_module(aggregated)
        
        # if self.arguments.is_pair_modeling:
        #     aggregated = self.norm_l(aggregated)
        #     return 1+torch.bmm(aggregated,aggregated.transpose(1,2))   # This is like calculating the cosine between two vectors. We add 1 to make the value in range [0,2]
        if not self.arguments.project_after_graph_encoder:
            # aggregated = self.aggregation_norm(aggregated)
            aggregated = self.final_classifier(aggregated)
            if self.arguments.return_positive_values:
                aggregated = aggregated.abs()
        return torch.squeeze(aggregated,2)
        

