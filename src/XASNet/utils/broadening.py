import torch
from typing import List
from tqdm import tqdm
import numpy as np


@torch.jit.script
def spectrum(E: torch.Tensor, 
             osc: torch.Tensor, 
             sigma: torch.Tensor, 
             x: torch.Tensor) -> torch.Tensor:
    """
    Function for Gaussian broadening of the XAS spectra so
    that all spectra will have the same shape for training.

    Args:
        E: XAS energies in eV.
        osc: Oscillator strength.
        sigma: Broadening of the spectra in eV.
        x: Fixed energy axis so that all spectra will have the same shape.

    Returns:
        broadened spectra 
    """
    val = torch.mul(
        torch.exp(-torch.pow((E.view(-1, 1) - x) / sigma, 2)),
        osc.view(-1, 1)
    )
    gE = torch.sum(val, 0)
    return gE

def batch_broadening(spectra_stk: List[np.ndarray],
                     sigma: float,
                     energies: torch.Tensor) -> List[torch.Tensor]:
    """
    function to broaden a batch of spectra

    Args:
        spectra_stk (List[np.ndarray]): List of spectra
        sigma (float): Broadening of the spectra in eV.
        energies (torch.Tensor): Fixed energy axis so that all 
            spectra will have the same shape.

    Returns:
        A list of broadened spectra.
    """
    broadened_spectra = []
    for spec in tqdm(spectra_stk):
        E = spec[:, 0]
        osc = spec[:, 1]
        gE = spectrum(E, osc, sigma, energies)
        #broadened_spec = torch.vstack((x, gE)).T
        broadened_spectra.append(gE)
        
    return broadened_spectra
