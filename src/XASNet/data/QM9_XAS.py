import os
from tqdm import tqdm
import os.path as osp
from typing import *

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)

import numpy as np
import torch
from torch import Tensor

from torch_geometric.utils import scatter

from torch_geometric.data import Data, Dataset
from torch_geometric.data.collate import collate

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem import rdMolTransforms 

class QM9_XAS(Dataset):
    """
    The QM9-XAS dataset. The dataset is similar to the original QM9 dataset 
    with addition of XAS spectra as the labels of graphs.
    """ 
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root: str, 
                 raw_dir: str, 
                 spectra: List[Tensor] = []):
        """
        Args:
            root: The path to the processed QM9-XAS dataset.
            raw_dir: The directory of raw qm9 data.
            spectra: List of all spectra for the subset of qm9 structures. 
        """
        self.root = root
        self.dir = raw_dir
           
        if osp.exists(root):
            self.data_list = torch.load(root)
        else:
            self.download()
            self.spectra = spectra
            self.process()

    @property
    def raw_file_names(self) -> List[str]:
        return ['gdb9.sdf', 'gbd9.sdf.csv']

    @property
    def raw_paths(self) -> List[str]:
        files = self.raw_file_names

        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.dir, f) for f in files]

    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.dir)
            extract_zip(file_path, self.dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.dir)
            os.rename(osp.join(self.dir, '3195404'),
                      osp.join(self.dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.dir)
            extract_zip(path, self.dir)
            os.unlink(path)

    @staticmethod
    def collate(
            data_list: List[Data]) -> Tuple[Data, Optional[Dict[str, torch.Tensor]]]:
        r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
        to the internal storage format of
        :class:`~torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices

    def process(self):
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        self.data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i == len(self.spectra):
                break
            #obtain the number of atoms 
            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []

            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            row, col, edge_type, bl = [], [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]
                bl += 2 * [rdMolTransforms.GetBondLength(conf, 
                                                        bond.GetBeginAtomIdx(), 
                                                        bond.GetEndAtomIdx())]
                

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = torch.nn.functional.one_hot(
              edge_type, num_classes=len(bonds)
               ).to(torch.float)
            edge_length = torch.tensor(bl, dtype=torch.float)
            #edge_length = (edge_length - torch.mean(edge_length)) / torch.std(edge_length)
            edge_length = (edge_length - edge_length.min()) / (edge_length.max() - edge_length.min())
            

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]
            edge_length = edge_length[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N).tolist()

            #calculate pairwise distances
            idx_t, idx_s = edge_index[0], edge_index[1]
            V_ts = pos[:, None, :] - pos[None, :, :]
            D_ij = torch.sqrt(torch.sum(V_ts**2, dim=1))
            #D_ij = (D_ij - D_ij.min()) / (D_ij.max() - D_ij.min())
            
            #V_ts = pos[:, None, :] - pos[None, :, :]
            #D_ij = np.linalg.norm(np.linalg.norm(V_ts, axis=1), axis=1) 
            #D_ij = torch.tensor(D_ij, dtype=torch.float)
            #D_ij = (D_ij - torch.mean(D_ij)) / torch.std(D_ij)

            x1 = torch.nn.functional.one_hot(
              torch.tensor(type_idx), num_classes=len(types)
            )
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                            dtype=torch.float).t().contiguous()

            x = torch.cat([x1.to(torch.float), x2, D_ij], dim=-1) 


            #adding the spectra data to the graph data
            spectrum = torch.tensor(self.spectra[i], dtype=torch.float)

            data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                      edge_attr=torch.cat([edge_attr, edge_length.reshape(-1,1)], -1), 
                                          spectrum=spectrum, idx=i)

            self.data_list.append(data)
        #self.collate(data_list)
        #torch.save(self.collate(self.data_list), self.root)

        return self.data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def len(self):
        pass
    
    def get(self):
        pass

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.int32, np.int64)):
            data = self.data_list[idx] 
        if isinstance(idx, (list, tuple, np.ndarray)):
            data = [self.data_list[i] for i in list(idx)]
        if isinstance(idx, slice):
            idx = np.arange(idx.start, min(idx.stop, len(self), idx.step))
            data = [self.data_list[i] for i in idx]   
        
        return data