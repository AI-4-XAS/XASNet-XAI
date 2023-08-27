# XASNet - Graph Neural Network models to predict X-ray absorption spectra

![generated molecules](./images/XASNet.png)

XASNet is a graph neural network (GNN) model to predict X-ray absorption spectra (XAS) of small molecules while maintaing the explainibility of the predicted spectra. It can be chosen based on different GNN architectures, i.e. [GraphNet](https://arxiv.org/abs/1806.01261), [graph convolutional neural network (GCN)](https://arxiv.org/abs/1509.09292), [multi-head graph attention network (GATv2)](https://arxiv.org/abs/1710.10903). XASNet can be trained on datasets of 3d molecules with variable sizes composed of the first- and second row of main group elements H, C, N, O, and F. Here, we trained the GNN models on custom-generated carbon K-edge XAS dataset of 65k small organic molecules (subset of original QM9), denoted as QM9-XAS. 

To explain the predictions, feature attributions are employed to determine the respective contributions of various atoms in the molecules to the peaks observed in the XAS spectrum. Here, we also developed a method which assigns the ground-truth contributions of various atoms in a molecule to a peak in the TDDFT spectrum. The developed data pipeline produces atoms labels denoting whether a particular atom conztibute to an XAS peak.


## <div align="center">Documentation</div>

Quickstart installation and usage example is given below. Training, prediction and explainability of XAS spectra are given in the example.  


### Content

+ [Installation](/README.md#installation)
+ [Python](/README.md#Python)
  + [Dataset preparation](/README.md#dataset-preparation)
  + [Model training and validation](/README.md#model-training)
  + [Prediction and explainability with ground truth data](/README.md#prediction-and-explainability-with-ground-truth-data)

# Installation

To install `XASNet-XAI`, download this repository and use pip.

```
git clone https://github.com/Amirktb1994/XASNet-XAI
conda create -n xasnet-xai numpy
conda activate xasnet-xai
pip install ./XASNet-XAI
```

# Python

## Dataset preparation

The raw and processed QM9-XAS dataset can be downloaded form 

The labels of [QM9-XAS dataset](https://doi.org/10.5281/zenodo.8276902) was used for training, validation and test. The labels of graphs in QM9-XAS are the correponding XAS spectra for QM9 structures. Python environment can be used according to the following to prepare the QM9-XAS dataset, 

```python
from XASNet.data import QM9_XAS

# load or create QM9-XAS graph dataset
root = 'path-to-save/load-QM9-XAS-dataset' 
qm9_spec = QM9_XAS(
    root=root,
    raw_dir='./', # path to save/load the raw data necessary to build the graph dataset
    spectra=xas_spectra # XAS spectra of all structures in QM9-XAS
)

# save the dataset if it doesn't exists
if not osp.exists(root):
    torch.save(qm9_spec, root)

```

## Model training and validation

`GNNTrainer` can be used to train and validate the GNN models. It can also be used for performing XAS spectra predictions with the trained models.

```python
from XASNet.models import XASNet_GNN
from XASNet.trainer import GNNTrainer

# load the GNN model
trainer = GNNTrainer(model=gnn_model, 
                     model_name="model-name",
                     device=device,
                     metric_path="./metrics")

trainer.train_val(
  train_loader, # train data loader 
  val_loader, # val data loader
  optimizer, # optimizer, i.e. AdamW
  loss_fn, # loss function
  scheduler, # learning rate scheduler  
  num_epochs, # number of epochs
  write_every=1, # frequency to write train/val outcome
  train_graphnet=True # whether the trained model is GraphNet
  )
```

## Prediction and explainability with ground truth data

```python
from XASNet.utils import GraphDataProducer
from XASNet.utils import (
    GroundTruthGenerator,
    OrcaAnlyser,
    Contributions
)

# loading test dataset
root = 'path-to-qm9xas-dataset'
test_qm9xas = QM9_XAS(root=root,
             raw_dir='./raw/')



```