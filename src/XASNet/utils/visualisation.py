from typing import Optional, Any, List
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
import networkx as nx

def plot_prediction(x_pred: torch.Tensor,
                    y_pred: torch.Tensor, 
                    y_true: torch.Tensor, 
                    normalise: Optional[bool] = False, 
                    add_peaks: Optional[bool] = False,
                    save: Optional[bool] = False,
                    save_name: Optional[str]= None):
    """
    Function to plot the predictions from trained GNN models.
    Args:
        x_pred (torch.Tensor): Energy axis of XAS spectra.
        y_pred (torch.Tensor): Predicted oscillator strength for energy axis.
        y_true (torch.Tensor): Ground truth oscillator strength for energy axis.
        normalise (Optional[bool], optional): If true, both y_pred and y_true is 
            normalised to 0 to 1. Defaults to False.
        add_peaks (Optional[bool], optional): If true, the peak information is 
            added to the plot. Defaults to False.
        save (Optional[bool], optional): If true, the plot is saved in png format. Defaults to False.
        save_name (Optional[str], optional): The name of the saved plot. Defaults to None.
    """
    if normalise:
        #standardize y pred and true 
        y_pred = MinMaxScaler().fit_transform(y_pred.reshape(-1, 1)).squeeze()
        y_true = MinMaxScaler().fit_transform(y_true.reshape(-1, 1)).squeeze()

    fig, ax = plt.subplots()
    ax.plot(x_pred, y_pred, color='r', label='prediction')
    ax.plot(x_pred, y_true, color='black', label='TDDFT')

    if add_peaks:
        peaks, _ = find_peaks(y_pred)
        widths_half = peak_widths(y_pred, peaks, rel_height=0.1)
        print(x_pred[peaks], widths_half[0])
        for x_peak, y_peak, width in zip(x_pred[peaks], \
                       y_pred[peaks], widths_half[0]):
            ax.text(x_peak, y_peak + 1, f"width = {width:.2f}", size=15)


    ax.legend()
    ax.set_yticklabels([])
    ax.set_xlabel('Energies (eV)')
    ax.set_ylabel('Intensity (arb. units)')
    ax.tick_params(axis='x', which='major', direction='out', 
                bottom=True, width=2, length=5)

    if save:
        plt.savefig(f"./{save_name}.png", 
                    dpi=300, 
                    bbox_inches='tight')
        

def plot_graph(graph: Any, 
               symbols: List[str], 
               weights: np.array, 
               save_fig: Optional[bool] = False,
               acceptor_orb: Optional[bool] = False, 
               don_orb: Optional[bool] = False):
    """
    Function to plot a heatmap on the graph. It allows to visualise the 
    ground truth contribution of atoms or CAM feature attributions for each peak.
    Args:
        graph (Any): Graph data.
        symbols (Dict[int, str]): Atomic symbols. 
        weights (np.array): weights of each atom wrt peaks in XAS spectrum. The weights 
        can be either ground truth contribution of atoms or atomic feature attributions.
        save_fig (Optional[bool], optional): if true, the plot is saved. Defaults to False.
        acceptor_orb (Optional[bool], optional): If true, the color of heatmap is blue. Defaults to False.
        don_orb (Optional[bool], optional): If true, the color of heatmap is red. Defaults to False.
    """
    if acceptor_orb:
        cmap = plt.cm.Blues
    elif don_orb:
        cmap = plt.cm.Reds
    else:
        cmap = plt.cm.Greens
        
    G=nx.Graph()

    symbols = dict(zip(
    np.arange(len(symbols)),
    symbols))
    
    for i, w in zip(range(graph.x.shape[0]),
                 weights):
        G.add_node(i, weight=w)

    target = graph.edge_index[0].tolist()
    source = graph.edge_index[1].tolist()

    for i, j in zip(target, source):
        G.add_edge(i, j)
        
    
    pos_dict = {}
    for i, xyz in enumerate(graph.pos):
        pos_dict[i] = xyz[:-1].detach().cpu().numpy()
     
    pos_g=nx.spring_layout(G, pos=pos_dict)
    
    nx.draw(G, pos=pos_g, node_color=weights,
            node_size=1000, cmap=cmap, with_labels=False)
    labels=nx.draw_networkx_labels(G, pos=pos_g, labels=symbols, 
                                   font_size=14, font_color='black',
                                  font_weight='bold')
    if save_fig:
        plt.savefig('./CAM_fig.png', dpi=300, bbox_inches='tight')
    
    return G