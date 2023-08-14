import torch
import torch.nn.functional as F
import torch_geometric.nn as geomnn

from utils.weight_init import kaiming_orthogonal_init

from typing import List
from torch.nn import Linear, LSTM
from torch_geometric.nn import MessagePassing, global_mean_pool, GATConv, GATv2Conv
from src.SpectraGAT.base_layers.gat_layers import GATLayerCus, GATv2LayerCus
from src.SpectraGAT.base_layers.dense_layers import LinearLayer, Residual_block

from src.GraphNet.modules import GraphNetwork
from torch.nn import ModuleList, Dropout
from torch_geometric.nn import global_add_pool


gnn_layers = {
    'gat': geomnn.GATConv,
    'gcn': geomnn.GCNConv,
    'gatv2': geomnn.GATv2Conv,
    'graphConv': geomnn.GraphConv
    }

class SpectraGNN(torch.nn.Module):
    def __init__(
        self, 
        gnn_name: str, 
        in_channels: int,
        out_channels: int,
        num_targets: int,
        gat_dp: float = 0,
        num_layers: int = 1, 
        heads: int = None,
        ) -> None:
        super().__init__()
        assert gnn_name in gnn_layers
        assert num_layers > 0
        assert len(in_channels) == num_layers and \
        len(out_channels) == num_layers

        self.gnn_name = gnn_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_targets = num_targets
        self.num_layers = num_layers
        self.heads = heads

        gnn_layer = gnn_layers[gnn_name]
        int_layers = []
      
        for i, in_c, out_c in zip(range(num_layers - 1), 
                                    in_channels[:-1], out_channels[:-1]):
                  
                if i == 0 and heads: 
                    int_layers.append(gnn_layer(in_c, out_c, heads=heads))
                elif i==0:
                    int_layers.append(gnn_layer(in_c, out_c))
                elif i > 0 and heads:
                    int_layers.append(gnn_layer(in_c*heads, out_c, heads=heads)) 
                elif i > 0:
                    int_layers.append(gnn_layer(in_c, out_c))
                int_layers.append(torch.nn.ReLU(inplace=True))
        
        if heads:
            int_layers.append(gnn_layer(in_channels[-1]*heads, 
                                        out_channels[-1], heads=1, 
                                        dropout=gat_dp))
        else:
            int_layers.append(gnn_layer(in_channels[-1], 
                                        out_channels[-1]))
        
        self.interaction_layers = torch.nn.ModuleList(int_layers)

        self.dropout = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(out_channels[-1], num_targets)      
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.interaction_layers:
            if isinstance(m, geomnn.GATConv):
                layers = [m.lin_src, m.lin_dst]
            elif isinstance(m, geomnn.GCNConv):
                layers = [m.lin]
            elif isinstance(m, geomnn.GATv2Conv):
                layers = [m.lin_r, m.lin_l]
            elif isinstance(m, geomnn.GraphConv):
                layers = [m.lin_rel, m.lin_root]
                
            for layer in layers:
                kaiming_orthogonal_init(layer.weight.data)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)
        
        kaiming_orthogonal_init(self.out.weight.data)
        self.out.bias.data.fill_(0.0)   

    def forward(self, x, 
              edge_index, batch_seg):   
    
        for layer in self.interaction_layers[:-1]:
            if isinstance(layer, geomnn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
                x = self.dropout(x)
                
        x = self.interaction_layers[-1](x, edge_index)        
        x = geomnn.global_mean_pool(x, batch_seg)
        x = self.dropout(x)
        out = self.out(x)
        return out



class SpectraGAT(torch.nn.Module):
    def __init__(
        self, 
        node_features_dim: int,
        in_channels: List[int],
        out_channels: List[int],
        n_heads: int,
        n_layers: int,
        targets: int,
        gat_type: str = 'gat_custom',
        use_residuals: bool = False,
        use_jk: bool = False
        ):
        super().__init__()

        self.node_features_dim = node_features_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.targets = targets
        self.gat_type = gat_type
        self.use_residuals = use_residuals
        self.use_jk = use_jk

        assert len(in_channels) == n_layers
        gat_types = {
            'gat_custom' : GATLayerCus,
            'gatv2_custom' : GATv2LayerCus,
            'gat' : GATConv,
            'gatv2' : GATv2Conv
        }
        assert gat_type in gat_types

        if use_residuals:
            self.pre_layer = LinearLayer(node_features_dim, in_channels[0], activation='relu')
            self.res_block = Residual_block(in_channels[0], 4, activation='relu')

        gat_layers = []

        for i, c_in, c_out in zip(range(n_layers-1), \
            in_channels[:-1], out_channels[:-1]):
            if i == 0:
                gat_layers.append(gat_types[gat_type](c_in, c_out, 
                heads=n_heads))
            elif i > 0:
                gat_layers.append(gat_types[gat_type](c_in*n_heads, c_out, 
                heads=n_heads))
            gat_layers.append(torch.nn.ReLU(inplace=True))
    
        gat_layers.append(gat_types[gat_type](in_channels[-1]*n_heads, out_channels[-1]))
        self.gat_layers = torch.nn.ModuleList(gat_layers)

        #jumping knowledge layers
        self.lstm = LSTM(out_channels[-2]*n_heads, out_channels[-2], 
        num_layers=3, bidirectional=True, batch_first=True)
        self.attn = Linear(2*out_channels[-2], 1)

        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear_out = LinearLayer(out_channels[-1], targets)

    def forward(self, x, edge_index, batch_seg):

        x = self.pre_layer(x)
        x = self.res_block(x)

        xs = []
        for layer in self.gat_layers[:-1]:
            if isinstance(layer, MessagePassing):
                x = layer(x, edge_index)
            else: 
                x = layer(x)
                x = self.dropout(x)
                xs.append(x.flatten(1).unsqueeze(-1))
        
        xs = torch.cat(xs, dim=-1).transpose(1, 2)
        alpha, _ = self.lstm(xs)
        alpha = self.attn(alpha).squeeze(-1)
        alpha = torch.softmax(alpha, dim=-1)
        x = (xs * alpha.unsqueeze(-1)).sum(1)

        x = self.gat_layers[-1](x, edge_index)
        x = global_mean_pool(x, batch_seg)
        x = self.dropout(x)
        out = self.linear_out(x)
        return out

class SpectraGraphNet(torch.nn.Module):
    def __init__(self,
                 all_params: dict = None,
                 n_layers: int = 3,
                 lin_in: int = 50,
                 n_targets: int = 100):
        super().__init__()
        assert len(all_params) == n_layers, "parameters should be provided \
            according to number of layers"
        
        #necassary_inputs = ["feat_in", "feat_hidd", "feat_out"]

        #for v in all_params.values():
         #   assert list(v.keys()) == necassary_inputs
        
        graphnets = []
        for v in all_params.values():
            graphnets.append(GraphNetwork(**v))

        self.graphnets = ModuleList(graphnets)

        self.dropout = Dropout(p=0.3)
        self.output_dense = Linear(lin_in, n_targets)
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_orthogonal_init(self.output_dense.weight.data)

    def forward(self, graph):
        for graphnet in self.graphnets:
            graph = graphnet(graph)

        x = global_add_pool(graph.x, graph.batch)
        
        x = self.dropout(x)
        out = self.output_dense(x)
        return out