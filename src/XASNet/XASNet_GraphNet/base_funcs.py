import torch
import torch.nn
from torch.nn import Sequential, ReLU, Linear, LayerNorm
try:
    from torch.nn import LazyLinear
except ImportError:
    print("torch version is %s. Lazy modules are turned off." % torch.__version__)

from torch_scatter import scatter_sum


__all__ = [
    "make_mlp",
    "make_lazy_mlp",
    "cast_edges_to_nodes",
    "cast_globals_to_nodes",
    "cast_globals_to_edges",
    "cast_nodes_to_globals",
    "cast_edges_to_globals"
]

def make_mlp(n_input, n_hidd, n_output, 
             act_final=False, normalize=False):
    
    if n_hidd is None:
        mlp_dim = [n_input, n_output]
    elif isinstance(n_hidd, int):
        mlp_dim = [n_input, n_hidd, n_output] 
    elif isinstance(n_hidd, list):
        mlp_dim = [n_input] + n_hidd + [n_output]

    mlp = []
    for i in range(len(mlp_dim) - 1):
        mlp.append(Linear(mlp_dim[i], mlp_dim[i+1]))
        if i < len(mlp_dim) - 2 or act_final:
            mlp.append(ReLU(inplace=True))
        
    if normalize and n_hidd is not None:
        mlp.append(LayerNorm(n_output))
    
    mlp = Sequential(*mlp)

    return mlp 

def make_lazy_mlp(n_hidd, n_output, 
             act_final=False, normalize=False):
    
    if n_hidd is None:
        mlp_dim = [n_output]
    elif isinstance(n_hidd, int):
        mlp_dim = [n_hidd, n_output] 
    elif isinstance(n_hidd, list):
        mlp_dim = n_hidd + [n_output]

    mlp = [LazyLinear(mlp_dim[0])]
    for i in range(len(mlp_dim) - 1):
        mlp.append(Linear(mlp_dim[i], mlp_dim[i+1]))
        if i < len(mlp_dim) - 2 or act_final:
            mlp.append(ReLU(inplace=True))
        
    if normalize and n_hidd is not None:
        mlp.append(LayerNorm(n_output))
    
    mlp = Sequential(*mlp)

    return mlp 


def get_edge_counts(batch, edge_index):
    return torch.bincount(batch[edge_index[0, :]])

def cast_edges_to_nodes(edge_attrs, indices, num_nodes=None):
    edge_attr_aggr = scatter_sum(edge_attrs, indices, dim=0, dim_size=num_nodes)
    return edge_attr_aggr

def cast_globals_to_nodes(global_attr, batch, num_nodes=None):
    if batch is not None:
        _, counts = torch.unique(batch, return_counts=True)
        casted_global_attr = torch.cat([torch.repeat_interleave(global_attr[idx:idx+1, :], rep, dim=0) \
                                       for idx, rep in enumerate(counts)], dim=0)      
    else:
        assert global_attr.size(0) == 1, "batch should be provided"
        assert num_nodes is not None, "number of nodes should be specified"
        casted_global_attr = torch.cat([global_attr] * num_nodes, dim=0)

    return casted_global_attr

def cast_globals_to_edges(global_attr, edge_index=None, batch=None, num_edges=None):
    if batch is not None:
        edge_counts = get_edge_counts(batch, edge_index)
        casted_global_attr = torch.cat([torch.repeat_interleave(global_attr[idx:idx+1, :], rep, dim=0) \
                                       for idx, rep in enumerate(edge_counts)], dim=0)      
    else:
        assert global_attr.size(0) == 1, "batch should be provided"
        assert num_edges is not None, "number of edges should be specified"
        casted_global_attr = torch.cat([global_attr] * num_edges, dim=0)

    return casted_global_attr

def cast_nodes_to_globals(node_attr, batch=None, num_globals=None):
    if batch is None:
        casted_node_attr = torch.sum(node_attr, dim=0, keepdim=True)
    else:
        casted_node_attr = scatter_sum(node_attr, batch, dim=0, dim_size=num_globals)
    return casted_node_attr

def cast_edges_to_globals(edge_attr, edge_index=None, batch=None, \
    num_edges=None, num_globals=None):

    if batch is None:
        casted_edge_attr = torch.sum(edge_attr, dim=0, keepdim=True)
    else:
        node_inds = torch.unique(batch)
        edge_counts = get_edge_counts(batch, edge_index)
        assert sum(edge_counts) == num_edges
        
        indices = [torch.repeat_interleave(idx, rep) for idx, rep in \
            zip(node_inds, edge_counts)]
        indices = torch.cat(indices)
        casted_edge_attr = scatter_sum(edge_attr, index=indices, dim=0, dim_size=num_globals)
    return casted_edge_attr