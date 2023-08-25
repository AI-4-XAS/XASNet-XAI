from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from .dense_layers import LinearLayer, Residual_block

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
#import torch_scatter
from torch_geometric.utils import scatter


class GATLayerCus(MessagePassing):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        share_weights: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.,
        use_residuals: bool = True,
        **kwargs
        ):
        super(GATLayerCus, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None

        if share_weights:
            self.lin_r = LinearLayer(in_channels, heads*out_channels)
            self.lin_l = self.lin_r
        else:
            self.lin_r = LinearLayer(in_channels, heads*out_channels)
            self.lin_l = LinearLayer(in_channels, heads*out_channels)
        
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels).float())
        self.att_l = Parameter(torch.Tensor(1, heads, out_channels).float())

        if use_residuals:
            self.res_block = Residual_block(out_channels, 4, activation='relu')
        else:
            self.res_block = torch.nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.att_l)
        torch.nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):
        
        H, C = self.heads, self.out_channels
        
        x_target = self.lin_r(x).view(-1, H, C)
        x_source = self.lin_l(x).view(-1, H, C)

        alpha_r = (x_target * self.att_r).sum(dim=-1)
        alpha_l = (x_source * self.att_l).sum(dim=-1)

        out = self.propagate(edge_index, x=(x_source, x_target), 
                            alpha=(alpha_l, alpha_r), size=size)

        out = out.view(-1, self.heads*self.out_channels)
        return out

    def message(self, x_j, alpha_i, alpha_j, index, ptr, size_i):
        attention = F.leaky_relu((alpha_i + alpha_j), negative_slope=self.negative_slope)
        attention = softmax(attention, index, ptr, size_i)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        out = x_j * attention.unsqueeze(-1)
        return out

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, 
        dim_size=dim_size, reduce='sum')

        out = self.res_block(out)

        return out




class GATv2LayerCus(MessagePassing):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        share_weights: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.,
        use_residuals: bool = True,
        **kwargs
        ):
        super(GATv2LayerCus, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.lin_l = None
        self.lin_r = None
        self.att = None
        self._alpha = None

        if share_weights:
            self.lin_r = LinearLayer(in_channels, heads*out_channels)
            self.lin_l = self.lin_r
        else:
            self.lin_r = LinearLayer(in_channels, heads*out_channels)
            self.lin_l = LinearLayer(in_channels, heads*out_channels)
        
        self.att = Parameter(torch.Tensor(1, heads, out_channels).float())

        if use_residuals:
            self.res_block = Residual_block(out_channels, 4, activation='relu')
        else:
            self.res_block = torch.nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, size=None):
        
        H, C = self.heads, self.out_channels
        
        x_target = self.lin_r(x).view(-1, H, C)
        x_source = self.lin_l(x).view(-1, H, C)

        out = self.propagate(edge_index, x=(x_source, x_target), size=size)

        out = out.view(-1, self.heads*self.out_channels)
        return out

    def message(self, x_j, x_i, index, ptr, size_i):
        x = x_i + x_j
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        alpha = (self.att * x).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.unsqueeze(-1)
        return out

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, 
        dim_size=dim_size, reduce='sum')

        out = self.res_block(out)
        return out