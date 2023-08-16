from typing import Any
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
import copy


def cam_gnn(graph: Any, model: Any) -> torch.Tensor:
    """
    Calculates CAM attributions for models trained with XASNetGNN module.

    Args:
        graph: Graph data. 
        model: The trained model used to predict the spectra on graph data.

    Returns:
        CAM values
    """
    model.eval()
    all_layers = [layer for layer in model.modules()]
    gnn_layers = all_layers[1]

    x, edge_index = graph.x, graph.edge_index
    with torch.no_grad():
        for layer in gnn_layers:
            if isinstance(layer, torch.nn.ReLU):
                x = layer(x)
            else:
                x = layer(x, edge_index)
    
    if isinstance(all_layers[-1], Linear):
        gap_weights = all_layers[-1].weight.data
    else:
        gap_weights = all_layers[-2].weight.data
    
    CAM = torch.matmul(x, gap_weights.T)
    CAM = F.relu(CAM)
    
    return CAM   

def cam_gat(graph, model):
    """
    Calculates CAM attributions for models trained with XASNetGAT module.

    Args:
        graph: Graph data. 
        model: The trained model used to predict the spectra on graph data.

    Returns:
        CAM values
    """   
    model.eval()
    all_layers = [layer for layer in model.modules()]
    gnn_layers = all_layers[18]
    lin_layers = [all_layers[1], all_layers[4]]

    x, edge_index = graph.x, graph.edge_index
    with torch.no_grad():
        for layer in lin_layers:
            x = layer(x)
    
    with torch.no_grad():
        for layer in gnn_layers:
            if isinstance(layer, torch.nn.ReLU):
                x = layer(x)
            else:
                x = layer(x, edge_index)
    
    if isinstance(all_layers[-1], Linear):
        gap_weights = all_layers[-1].weight.data
    else:
        gap_weights = all_layers[-2].weight.data
    
    CAM = torch.matmul(x, gap_weights.T)
    CAM = F.relu(CAM)
    
    return CAM

def cam_graphnet(graph, model):
    """
    Calculates CAM attributions for models trained with XASNetGraphNet module.

    Args:
        graph: Graph data. 
        model: The trained model used to predict the spectra on graph data.

    Returns:
        CAM values
    """
    model.eval()
    all_layers = [layer for layer in model.modules()]
    
    graphnet_layers = all_layers[1]

    with torch.no_grad():
        for layer in graphnet_layers:
            graph = layer(graph)
    
    if isinstance(all_layers[-1], Linear):
        gap_weights = all_layers[-1].weight.data
    else:
        gap_weights = all_layers[-2].weight.data
    
    CAM = torch.matmul(graph.x, gap_weights.T)
    CAM = F.relu(CAM)
    
    return CAM    