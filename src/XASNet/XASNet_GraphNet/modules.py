import torch
from torch.nn import ReLU
from torch_geometric.nn import GATv2Conv, MessagePassing, global_add_pool
from .base_funcs import *



class GATEncoder(torch.nn.Module):
    """
    Encoder layer to obtain the global state of each input graph.
    The global state will be 
    """
    def __init__(self, 
    in_feat=11, 
    hidd_feat=64, 
    out_feat=20, 
    n_layers=3, 
    heads=3):
        super(GATEncoder, self).__init__()
        gats = []
        for i in range(n_layers):
            if i < 1:
                gats.append(GATv2Conv(in_feat, hidd_feat, heads=heads))
            else:
                gats.append(GATv2Conv(hidd_feat*heads, hidd_feat, heads=heads))
            gats.append(ReLU(inplace=True))
        
        gats.append(GATv2Conv(hidd_feat*heads, out_feat, heads=1))

        self.gats = torch.nn.ModuleList(gats)

        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.gats:
            if not isinstance(layer, ReLU):
                layer.reset_parameters()
    
    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        for layer in self.gats:
            if isinstance(layer, MessagePassing):
                x = layer(x, edge_index)  
        out = global_add_pool(x, batch=batch)
        return out

class BaseModel(torch.nn.Module):
    def __init__(
        self,
        feat_in=None,
        feat_out=16,
        feat_hidd=64,
        activate_final=True,
        normalize=True,
        independant=False,
        **kwargs
    ):
        super(BaseModel, self).__init__()
        self.independant = independant
        self.kwargs = kwargs
        if feat_in:
            self.mlp = make_mlp(feat_in, feat_hidd, feat_out, \
                act_final=activate_final, normalize=normalize)
        else:
            self.mlp = make_lazy_mlp(feat_hidd, feat_out, \
                act_final=activate_final, normalize=normalize)
        
        #self.encoder = GATEncoder(in_feat=gat_in, out_feat=gat_out)

    def collect_attrs(self, graph, global_attr):
        raise NotImplementedError
    
    def forward(self, graph, graph_attr, concat_graph=None):
        #graph_attr = self.encoder(graph)
        attrs = self.collect_attrs(graph, graph_attr)
        if concat_graph:
            concat_attrs = self.collect_attrs(concat_graph, 
            graph_attr)
            attrs = [torch.cat((attr, concat_attr), dim=-1) for (attr, concat_attr) in \
                zip(attrs, concat_attrs)]
        attrs = torch.cat(attrs, dim=1)
        return self.mlp(attrs)

class NodeModel(BaseModel):
    def collect_attrs(self, graph, global_attr):
        if self.independant:
            return [graph.x]
        
        node_attr, edge_attr, edge_index, batch = graph.x, graph.edge_attr, \
            graph.edge_index, graph.batch
        num_nodes = graph.x.size(0)
        row, col = edge_index
        receiver_attr_to_node = cast_edges_to_nodes(edge_attr, col, \
            num_nodes=num_nodes)
        sender_attr_to_node = cast_edges_to_nodes(edge_attr, row, \
            num_nodes=num_nodes)

        global_attr_to_nodes = cast_globals_to_nodes(global_attr, batch=batch, 
        num_nodes=num_nodes)
        
        out = [node_attr, receiver_attr_to_node, \
            sender_attr_to_node, global_attr_to_nodes]
        return out

    def forward(self, graph, graph_attr, concat_graph=None):
        graph.x = super().forward(graph, graph_attr, concat_graph)
        return graph

class EdgeModel(BaseModel):
    def collect_attrs(self, graph, global_attr):
        if self.independant:
            return [graph.edge_attr]

        node_attr, edge_attr, edge_index, batch = graph.x, graph.edge_attr, \
            graph.edge_index, graph.batch

        num_edges = graph.edge_attr.size(0)
        row, col = edge_index
        sender_attr, receiver_attr = node_attr[row, :], node_attr[col, :]

        global_attr_to_edges = cast_globals_to_edges(
            global_attr, edge_index, 
            batch, num_edges
        )

        out = [edge_attr, sender_attr, \
            receiver_attr, global_attr_to_edges]

        return out

    def forward(self, graph, graph_attr, concat_graph=None):
        graph.edge_attr = super().forward(graph, graph_attr, concat_graph)
        return graph

class GlobalModel(BaseModel):
    def collect_attrs(self, graph, global_attr):
        node_attr, edge_attr, edge_index, batch = graph.x, graph.edge_attr, \
            graph.edge_index, graph.batch
        
        num_edges = edge_attr.size(0)
        num_globals = global_attr.size(0)

        node_attr_aggr = cast_nodes_to_globals(node_attr, batch, num_globals)
        edge_attr_aggr = cast_edges_to_globals(edge_attr, edge_index, 
        batch, num_edges, num_globals)

        out = [global_attr, node_attr_aggr, edge_attr_aggr]

        return out
    
    def forward(self, graph, graph_attr, concat_graph=None):
        graph.u = super().forward(graph, graph_attr, concat_graph)
        return graph

    
class GraphNetwork(torch.nn.Module):
    def __init__(self,
     node_model_params,
     edge_model_params,
     global_model_params,
     gat_in=11, 
     gat_hidd=64,
     gat_out=20):
     super(GraphNetwork, self).__init__()

     self.gatencoder = GATEncoder(gat_in, gat_hidd, gat_out)
     self.node_model = NodeModel(**node_model_params)
     self.edge_model = EdgeModel(**edge_model_params)
     self.global_model = GlobalModel(**global_model_params)

    def forward(self, graph, concat_graph=None):
        try:
            graph_attr = graph.u
        except AttributeError:
            graph_attr = self.gatencoder(graph)

        graph = self.node_model(graph, graph_attr, concat_graph=concat_graph)
        graph = self.edge_model(graph, graph_attr, concat_graph=concat_graph)
        graph = self.global_model(graph, graph_attr, concat_graph=concat_graph)
        return graph