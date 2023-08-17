import os.path as osp
import numpy as np


def save_split(path: str, 
               ndata: int,
               ntrain: int,
               nval: int,
               ntest: int,
               save_split: bool,
               shuffle: bool = False,
               seed: int = 42, 
               print_nsample: bool = False
               ) -> None:
    """function to split the indexes of the qm9 dataset. 
    The indexes are saved to split_qm9.npz

    Args:
        path (str): path to save the split file
        ndata (int): number of structures in the whole dataset
        ntrain (int): size of train data
        nval (int): size of validation data
        save_split (bool): sacing the split file
        print_nsample (bool, optional): whether to print the size of train, test and val data.

    Returns:
        idxs: a dictionary of splitted indexes
    """
    
    assert path.endswith('.npz')

    if osp.exists(path):
        idxs = np.load(path)
        return idxs
    else:
        assert ndata >= ntrain + nval
    
    nsamples = {
        'train' : ntrain,
        'val' : nval,
        'test' : ntest
    }
    
    if print_nsample:
        print(nsamples)
    
    random_state = np.random.RandomState(seed=seed)

    all_idx = np.arange(ndata)

    if shuffle:
        all_idx = random_state.permutation(all_idx)

    idxs = {
        'train' : all_idx[:ntrain],
        'val' : all_idx[ntrain : ntrain+nval],
        'test' : all_idx[ntrain+nval:ntrain+nval+ntest]
    }

    if save_split:
        np.savez(path, **idxs)

    return idxs