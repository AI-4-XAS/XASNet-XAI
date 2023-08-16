from typing import Any, Dict
import copy
import re

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def rse_loss(prediction: torch.Tensor, 
             target: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate relative spectral error (RSE) which 
    is considered as a metric to train GNN models.
    Args:
        prediction (torch.Tensor): XAS spectrum predicted by the model
        target (torch.Tensor): target XAS spectrum.

    Returns:
        torch.Tensor
    """
    dE = (300 - 270) / 100
    nom = torch.sum(dE*torch.pow((target-prediction), 2))
    denom = torch.sum(dE*target)
    return torch.sqrt(nom) / denom 

def rse_predictions(model: Any, 
                    test_data, 
                    graphnet=False, 
                    device='cpu') -> Dict[int, float]:
    """
    Function to evaluate the performance of the trained GNN model.
    The RSE predictions is performed thorough all the test dataset.

    Args:
        model:  Trained GNN model.
        test_data: Uploaded test dataset of QM9_XAS dataset.
        graphnet (bool, optional): If true, the calculations is performed based 
            on GraphNet architecture. Defaults to False.
        device (str, optional): Defaults to 'cpu'.

    Returns:
        A dictionary of all data points in test dataset, i.e. 
        keys are data index and values are RSE values.
    """
    rse_dict_test = {}
    for graph in test_data:
        graph = copy.deepcopy(graph)
        x, edge_idx, idx = graph.x, graph.edge_index, graph.idx
        batch_seg = torch.tensor(np.repeat(0, x.shape[0]), 
                                 device=device)
        model.to(device)
        model.eval()
        with torch.no_grad():
            if graphnet:
                graph.batch = batch_seg
                pred = model(graph)
            else:
                pred = model(x, edge_idx, batch_seg)

        rse_dict_test[idx] = rse_loss(pred, graph.spectrum)
    return rse_dict_test

def rse_histogram(all_rse: Dict[int, float], 
                  model_name: str, 
                  quantiles: bool = True,
                  bins: int = 140, 
                  save: bool = False):
    """
    Function to plot RSE histogram.
    Args:
        all_rse: Dict of all RSE values.
        model_name: The model name that is evaluating.
        quantiles: If true, add info about the quantiles 
            of the RSE data. Defaults to True.
        bins (int, optional): Number of bins in the histogram. 
            Defaults to 140.
        save (bool, optional): Defaults to False.
    """
    if not isinstance(all_rse, np.ndarray):
        all_rse = pd.DataFrame(np.asarray(all_rse))
    
    freq, points = np.histogram(np.asarray(all_rse), bins=bins)
    max_point = points[np.argmax(freq)]
    
    fig, ax = plt.subplots()
    all_rse.plot(kind='hist', bins=bins, 
                 density=True, alpha=0.7, ax=ax, legend=None)
    all_rse.plot(kind='kde', ax=ax, alpha=0.6, legend=None)

    if quantiles:
        quantiles = {}
        for percent in np.arange(0.05, 1, 0.2):
            quantiles[f"quant_{percent:.2f}"] = all_rse.quantile(percent)
        ymax = 0.1
        height = 2
        for name, quant in quantiles.items():
            name = re.findall(r"\d+\.\d+", name)[0]
            ax.axvline(quant[0], ymax=ymax, linestyle=":", color="black")
            ax.text(quant[0]-0.004, height, f"{float(name)*100}%", size=15, alpha=0.9)
            ymax += 0.1
            height += 2.5
    ax.axvline(max_point + 0.0014, ymax=10, linestyle=":", color="green")
    ax.text(0.13, 18, f"RSE = {max_point:.2f}")#, size=20)
    
    ax.set_xlim(0, 0.2)
    ax.set_yticks([])
    ax.set_xlabel('RSE')
    ax.set_ylabel('Frequency')
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_title(f"RSE histogram of {model_name}")
    ax.tick_params(axis='x', which='major', direction='out', bottom=True, width=2, length=5)
    plt.style.use("bmh")
    
    if save:
        fig.savefig('./rmse_hist.png', dpi=300, bbox_inches='tight')