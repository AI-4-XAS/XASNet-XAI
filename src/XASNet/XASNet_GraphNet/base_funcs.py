from typing import Any, Union, Optional
import numpy as np
import torch
import torch.nn
from torch.nn import Sequential, ReLU, Linear, LayerNorm
try:
    from torch.nn import LazyLinear
except ImportError:
    print("torch version is %s. Lazy modules are turned off." % torch.__version__)

#from torch_scatter import scatter_sum
from torch_geometric.utils import scatter


__all__ = [
    "make_mlp",
    "make_lazy_mlp",
    "cast_edges_to_nodes",
    "cast_globals_to_nodes",
    "cast_globals_to_edges",
    "cast_nodes_to_globals",
    "cast_edges_to_globals"
]

def make_mlp(
    n_input: int, 
    n_hidd: int, 
    n_output: int, 
    act_final: bool = False, 
    normalize: bool = False
    ) -> Any:
    """
    Function to create and stack linear MLP layers for node, edge 
    and global models in GraphNet. 

    Args:
        n_input (int): Input channels.
        n_hidd (int): Hidden channels.
        n_output (int): Output channels.
        act_final (bool, optional): If true, final ReLU activation 
            layer is applied. Defaults to False.
        normalize (bool, optional): If true, layer normalisation is 
            applied at the last MLP layer. Defaults to False.
    """
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

def make_lazy_mlp(
    n_hidd: int, 
    n_output: int, 
    act_final: bool = False, 
    normalize: bool = False
    ) -> Any:
    """
    Function to make lazy linear MLP when the input channels is not constant.
    Args:
        n_hidd (int): Hidden channels.
        n_output (int): Output channels.
        act_final (bool, optional): If true, final ReLU activation 
            layer is applied. Defaults to False.
        normalize (bool, optional): If true, layer normalisation is 
            applied at the last MLP layer. Defaults to False.
    """
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

def cast_edges_to_nodes(
    edge_attrs: torch.Tensor, 
    indices: Union[torch.Tensor, np.array], 
    num_nodes: bool = None
    ) -> torch.Tensor:
    """
    Function to cast edge state on nodes.
    Args:
        edge_attrs (torch.Tensor): Edge attributions.
        indices (Union[torch.Tensor, np.array]): Edge indices.
        num_nodes (bool, optional): Number of nodes. If true, the 0 dimension of the scattered 
            matrix is equal to number of nodes. Defaults to None.
    """
    edge_attr_aggr = scatter_sum(edge_attrs, indices, 
                                 dim=0, dim_size=num_nodes)
    return edge_attr_aggr

def cast_globals_to_nodes(
    global_attr: torch.Tensor, 
    batch: torch.Tensor, 
    num_nodes: Optional[bool]
    ) -> torch.Tensor:
    """
    Function to cast graph global state on nodes.
    Args:
        global_attr (torch.Tensor): Global attributions obtained by GATEncoder.
        batch (torch.Tensor): Batch indices of graph data.
        num_nodes (bool, optional): Number of nodes. If true, the global attribution is repeated times 
            the number of nodes in dimension 0. Defaults to None.
    """
    if batch is not None:
        _, counts = torch.unique(batch, return_counts=True)
        casted_global_attr = torch.cat([torch.repeat_interleave(global_attr[idx:idx+1, :], rep, dim=0) \
                                       for idx, rep in enumerate(counts)], dim=0)      
    else:
        assert global_attr.size(0) == 1, "batch should be provided"
        assert num_nodes is not None, "number of nodes should be specified"
        casted_global_attr = torch.cat([global_attr] * num_nodes, dim=0)

    return casted_global_attr

def cast_globals_to_edges(
    global_attr: torch.Tensor, 
    edge_index: Optional[bool], 
    batch: Optional[bool], 
    num_edges: Optional[bool]
    ):
    """
    Function to cast graph global state on edges. 
    Args:
        global_attr (torch.Tensor): Global attributions obtained by GATEncoder.
        edge_index (bool, optional): Edge index. Defaults to None.
        batch (bool, optional): Batch indices of graph data. Defaults to None.
        num_edges (bool, optional): Number of edges. If true, the global attribution is repeated times 
            the number of nodes in dimension 0. Defaults to None.

    Returns:
        _type_: _description_
    """
    if batch is not None:
        edge_counts = get_edge_counts(batch, edge_index)
        casted_global_attr = torch.cat([torch.repeat_interleave(global_attr[idx:idx+1, :], rep, dim=0) \
                                       for idx, rep in enumerate(edge_counts)], dim=0)      
    else:
        assert global_attr.size(0) == 1, "batch should be provided"
        assert num_edges is not None, "number of edges should be specified"
        casted_global_attr = torch.cat([global_attr] * num_edges, dim=0)

    return casted_global_attr

def cast_nodes_to_globals(
    node_attr: torch.Tensor, 
    batch: Optional[bool], 
    num_globals: Optional[bool]
    ) -> torch.Tensor:
    """
    Function to cast node state on globals.
    Args:
        node_attr (torch.Tensor): Node attributions.
        batch (Optional[bool]): Batch indices of graph data.
        num_globals (Optional[bool]): Number of global states. If True, the dimension 0 of casted 
            node attribution is chosen as the number of global states.
    """
    if batch is None:
        casted_node_attr = torch.sum(node_attr, dim=0, keepdim=True)
    else:
        casted_node_attr = scatter_sum(node_attr, batch, dim=0, dim_size=num_globals)
    return casted_node_attr

def cast_edges_to_globals(
    edge_attr: torch.Tensor, 
    edge_index: Optional[torch.Tensor], 
    batch: Optional[torch.Tensor], 
    num_edges: Optional[int], 
    num_globals: Optional[int]
    ):
    """
    Function to cast edge state on globals.
    Args:
        edge_attr (torch.Tensor): Edge attributions.
        edge_index (Optional[torch.Tensor]): Edge index.
        batch (Optional[torch.Tensor]): Batch indices of graph data.
        num_edges (Optional[int]): Number of edges.
        num_globals (Optional[int]): Number of global states. If True, the dimension 0 of casted 
            edge attribution is chosen as the number of global states.
    """
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