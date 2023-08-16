from typing import Any, List, Optional
import copy
import torch
import numpy as np
from .cams import cam_graphnet, cam_gnn, cam_gat

class GraphDataProducer:
    """
    Class for picking a graph from the test dataset

    Args:
        model: Trained GNN model.
        test_data: Uploaded test dataset of QM9_XAS dataset.
        idx_to_pick: Index of the graph to pick.
        gnn_type: Type of the GNN used for training.
        add_batch: If true, adds batch attribute to the passed graph data.
    Returns:
        returns a deep copy of the chosen graph
    """
    gnn_types = ['gatv2', 'gatv2cus', 'graphNet', 'gcn']

    def __init__(self, 
                 model: Any,
                 test_data: Any,
                 idx_to_pick: int,
                 gnn_type: str,
                 add_batch: bool = True):
        
        assert gnn_type in GraphDataProducer.gnn_types 
        self.model = model
        self.gnn_type = gnn_type
        self.picked_graph = None
        for graph in test_data:
            if graph.idx == idx_to_pick:
                self.picked_graph = copy.deepcopy(graph)
                self.picked_graph2 = copy.deepcopy(graph)

        assert self.picked_graph is not None, \
            "the index doesn't exist in test dataset"
        
        if add_batch:
            self.picked_graph.batch = torch.repeat_interleave(torch.tensor(0),
                                                  self.picked_graph.x.shape[0])
    
    def atom_labels(self):
        atomic_num = self.picked_graph.z
        label_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
        atom_labels = []
        for i, z in enumerate(atomic_num):
            atom_labels.append(f"{label_map[z.item()]} {i}")

        return atom_labels
    

    def predictions(self,                   
                    device: str = 'cpu'):
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            if self.gnn_type == 'graphNet':
                # input only graph
                y_pred = self.model(self.picked_graph)
            else:
                y_pred = self.model(self.picked_graph.x,
                               self.picked_graph.edge_index,
                               self.picked_graph.batch)
                
            x_pred = np.linspace(270, 300, 100)
            y_pred = y_pred.numpy().flatten()
        return x_pred, y_pred
    
    def cam(self):
        if self.gnn_type == 'graphNet':
            cam = cam_graphnet(self.picked_graph2, self.model)
        elif self.gnn_type == 'gatv2cus':
            cam = cam_gat(self.picked_graph2, self.model)
        else:
            cam = cam_gnn(self.picked_graph2, self.model)
        return cam 