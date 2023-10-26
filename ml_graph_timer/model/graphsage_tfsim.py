import torch
import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn
from .sageconv import SAGEConv

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
    node_features: int = 112+opcode_dim
    node_feature_dim:int = 192
    config_features: int = 14
    conf_feature_dim: int = 64

    feature_dropout: float=0.0
    feature_expand: int = 1

    graphsage_in: int = node_feature_dim+conf_feature_dim
    graphsage_hidden: int = graphsage_in

    graphsage_layers: int = 2
    conv_aggregation_type: str = "max"
    mlp_layers: int = 2


    final_dropout: float = 0.0
    embedding_dropout: float = 0.0


    attention_blocks: int = 0
    drop_rate: float = 0.1
    attention_dropout: float = 0.1
    num_heads: int = 4

    is_pair_modeling: bool = False


import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, dims, alpha=0.2,use_bias=True):
        super(MLP, self).__init__()
        
        layers = []
        dims = [input_dim] + dims  # Add input dimension to the beginning
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i-1], dims[i], bias=use_bias))
            if i < len(dims) - 1:  # Do not append activation for the last layer
                layers.append(nn.LeakyReLU(alpha))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GraphSage(torch.nn.Module):
    def __init__(self,arguments: GraphModelArugments):
        super().__init__()
        self.sageconvs = torch.nn.ModuleList([SAGEConv(in_channels=arguments.graphsage_hidden,out_channels=arguments.graphsage_hidden,aggr=arguments.conv_aggregation_type) for _ in range(arguments.graphsage_layers)])
        self.mlps = torch.nn.ModuleList([MLP(input_dim=arguments.graphsage_hidden,dims=[arguments.node_feature_dim]*arguments.mlp_layers) for _ in range(arguments.graphsage_layers)])
        self.prenet = MLP(input_dim=arguments.graphsage_in,dims=[arguments.node_feature_dim]*arguments.mlp_layers)

    def forward(self,node_features, config_features, edges, *args,**kwargs):
        x = node_features
        x = torch.concat([config_features, x],dim=1)
        x = self.prenet(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        for layer,mlps in zip(self.sageconvs,self.mlps):
            y = x
            y = torch.concat([config_features, y], dim=1)
            y = F.leaky_relu(mlps(layer(y,edges)),negative_slope=0.2)
            x += y
        return x
    
class LayoutGraphModel(torch.nn.Module):
    def __init__(self,arguments: GraphModelArugments):
        super().__init__()
        self.arguments = arguments
        self.opcode_embeddings = torch.nn.Embedding(arguments.num_opcodes,arguments.opcode_dim)
        if arguments.graphsage_layers>0:
            print("Skipping graph layer")
            self.graph_encoder = GraphSage(arguments=arguments)
        else:
            self.graph_encoder = None

        self.node_features_mlp = torch.nn.Sequential(
            torch.nn.Linear(arguments.node_features,arguments.node_features*arguments.feature_expand),
            torch.nn.Dropout(arguments.feature_dropout),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(arguments.node_features*arguments.feature_expand,arguments.node_feature_dim),
            torch.nn.LeakyReLU(inplace=True),
        )

        self.config_features_mlp = torch.nn.Sequential(
            torch.nn.Linear(arguments.config_features,arguments.config_features*arguments.feature_expand),
            torch.nn.Dropout(arguments.feature_dropout),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(arguments.config_features*arguments.feature_expand,arguments.conf_feature_dim),
            torch.nn.LeakyReLU(inplace=True),
        )

        if arguments.attention_blocks>0:
            self.aggregation_norm = torch.nn.LayerNorm(arguments.graphsage_hidden)

            self.attention_module = torch.nn.Sequential(
                *[
                    TransformerBlock(arguments.graphsage_hidden,arguments.num_heads,attn_dropout=arguments.attention_dropout,drop_rate=arguments.drop_rate) for _ in range(arguments.attention_blocks)
                ]
            )
        else:
            self.attention_module = None
        self.embed_drop = torch.nn.Dropout(arguments.embedding_dropout)
        if not self.arguments.is_pair_modeling:
            self.final_classifier = torch.nn.Sequential(
                torch.nn.Dropout(arguments.final_dropout),
                torch.nn.Linear(arguments.node_feature_dim,1,bias=False)
            )

    def forward(self,node_features, node_config_features,node_separation, node_ops, edges, batches,*args,**kwargs):
        opcode_embed = self.embed_drop(self.opcode_embeddings(node_ops))
        x = torch.cat([node_features,opcode_embed],dim=1)
        x = self.node_features_mlp(x)
        node_config_features = self.config_features_mlp(node_config_features)
        if self.graph_encoder is not None:
            x = self.graph_encoder(x,node_config_features,edges)
        start_idx = 0
        aggregated = [[] for _ in torch.unique(batches)]
        for ns,b in zip(node_separation,batches):
            end_idx = ns
            aggregated[b].append(torch.sum(x[start_idx:end_idx], dim=0))
            start_idx = end_idx
        aggregated = torch.stack([torch.stack(x) for x in aggregated])
        
        if self.arguments.is_pair_modeling:
            aggregated = self.aggregation_norm(aggregated)
            aggregated_fc = self.attention_module(aggregated)
            return torch.tanh(torch.bmm(aggregated,aggregated_fc.transpose(1,2)))   # This is like calculating the cosine between two vectors. We add 1 to make the value in range [0,2]
        elif self.attention_module is not None:
            aggregated = self.aggregation_norm(aggregated)
            aggregated = self.attention_module(aggregated)

        aggregated = torch.squeeze(self.final_classifier(aggregated),2)
        return aggregated
        

