import torch
from torch_geometric.nn.models import GraphSAGE,GAT
import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn

import torch
import torch.nn as nn
from pytorch_tabnet.tab_network import TabNetEncoder
class GraphwiseLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1,normalized_shape))
        self.bias = nn.Parameter(torch.zeros(1,normalized_shape))
        
    def forward(self, x, node_separation):
        start_idx = 0
        normalized_batches = []
        for ns in node_separation:
            end_idx = start_idx + ns
            batch = x[start_idx:end_idx]
            
            # Calculate mean and var for the batch
            mean = batch.mean(dim=1, keepdim=True)
            var = batch.var(dim=1, unbiased=False, keepdim=True)
            
            # Normalize the batch
            batch = (batch - mean) / torch.sqrt(var + self.eps)
            
            # Apply learnable parameters
            batch = batch * self.weight + self.bias
            
            normalized_batches.append(batch)
            start_idx = end_idx

        # Concatenate the normalized batches back into one tensor
        return torch.cat(normalized_batches, dim=0)

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

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, expand_dim, dropout_rate):
        super().__init__()
        self.norm1 = GraphwiseLayerNorm(input_dim)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.expand_linear = nn.Linear(input_dim, expand_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm2 = GraphwiseLayerNorm(expand_dim)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.shrink_linear = nn.Linear(expand_dim, output_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Projection layer for the residual connection if input and output dimensions are not the same
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = None

    def forward(self, x,node_separation):
        # Save input for the residual connection
        identity = x

        # First half of the block
        x = self.norm1(x,node_separation)
        x = self.relu1(x)
        x = self.expand_linear(x)
        x = self.dropout1(x)

        # Second half of the block
        x = self.norm2(x,node_separation)
        x = self.relu2(x)
        x = self.shrink_linear(x)
        x = self.dropout2(x)

        # Apply projection if necessary
        if self.projection is not None:
            identity = self.projection(identity)

        # Add the residual (identity)
        x += identity
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

    model_type: str = "gsage"
    post_encoder_blocks: int = 0
class LayoutGraphModel(torch.nn.Module):
    def __init__(self,arguments: GraphModelArugments):
        super().__init__()
        self.arguments = arguments
        self.opcode_embeddings = torch.nn.Embedding(arguments.num_opcodes,arguments.opcode_dim)
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
        self.tabnet_encoder = TabNetEncoder(input_dim=arguments.node_feature_dim,output_dim=arguments.graphsage_in)

        self.post_encoder_mlp = nn.ModuleList()
        input_dim = arguments.graphsage_in
        for i in range(arguments.post_encoder_blocks):
            block = ResidualBlock(
                input_dim=input_dim,
                output_dim=input_dim,
                expand_dim=input_dim * arguments.node_feature_expand,
                dropout_rate=arguments.node_feature_dropout
            )
            self.post_encoder_mlp.append(block)
            # Since the output dim of the block 

        if arguments.attention_blocks>0:
            self.attention_module = torch.nn.Sequential(
                *[
                    TransformerBlock(arguments.graphsage_hidden,arguments.num_heads,attn_dropout=arguments.attention_dropout,drop_rate=arguments.drop_rate) for _ in range(arguments.attention_blocks)
                ]
            )
        else:
            self.attention_module = None
        self.embed_drop = torch.nn.Dropout(arguments.embedding_dropout)
        self.final_classifier = torch.nn.Sequential(
            torch.nn.Dropout(arguments.final_dropout),
            torch.nn.Linear(arguments.graphsage_hidden,1)
        )

    def forward(self,node_features, node_config_features, node_separation, node_ops, edges, batches,*args,**kwargs):
        opcode_embed = self.embed_drop(self.opcode_embeddings(node_ops))
        x = torch.cat([node_features,node_config_features,opcode_embed],dim=1)
        x = self.tabnet_encoder(x)
        if self.graph_encoder is not None:
            x = self.graph_encoder(x,edges)

        for block in self.post_encoder_mlp:
            x = block(x,node_separation)

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
        
        # if self.arguments.is_pair_modeling:
        #     aggregated = self.norm_l(aggregated)
        #     return 1+torch.bmm(aggregated,aggregated.transpose(1,2))   # This is like calculating the cosine between two vectors. We add 1 to make the value in range [0,2]
        if not self.arguments.project_after_graph_encoder:
            aggregated = self.final_classifier(aggregated)
            if self.arguments.return_positive_values:
                aggregated = aggregated.abs()
        return torch.squeeze(aggregated,2)
        

