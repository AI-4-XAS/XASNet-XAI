import torch
import torch.nn.functional as F
import torch_geometric.nn as geomnn

from utils.weight_init import kaiming_orthogonal_init

#from attribution_gnn2.base_layers.dense_layers import LinearLayer, Residual_block

# dictionary of different gnn layers 
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