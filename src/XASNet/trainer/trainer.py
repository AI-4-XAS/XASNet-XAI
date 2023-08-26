from typing import Any
import torch
import numpy as np
import pandas as pd
from .metrics import MetricManager
from tqdm import tqdm
import os
import os.path as osp
import json
import matplotlib.pyplot as plt

class GNNTrainer():
    """
    Class to train and validate GNN models. It can also run predictions 
    with the already saved trained model. 
    """
    def __init__(
        self, 
        model: Any, 
        model_name: str, 
        device: str,
        metric_path: str
        ):
        """
        Args:
            model (Any): GNN model to train. It can be GATv2, GCN or GraphNet model.
            model_name (str): The name of the model to save.
            device (str): Device to train on, i.e. cpu or cuda.
            metric_path (str): path to save the metrics data.
        """
        self.model = model
        self.model_name = model_name 
        self._save_model_params()

        self.metrics = MetricManager()
        # load the previous metrics from last training
        if osp.exists(osp.join(metric_path, "train_metrics.csv")):
            for mode in self.metrics.modes:
                df = pd.read_csv(
                    osp.join(metric_path, f"{mode}_metrics.csv")
                    )
                for col in list(df.columns):
                    self.metrics.outputs[mode][col] = list(df[col])
        else:
            pass

        self.device = device

    def train_val(
        self, 
        train_loader, 
        val_loader, 
        optimizer,
        loss_fn, 
        scheduler,
        epochs,
        write_every: int = 50,
        train_graphnet: bool =False
        ):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        prev_loss = torch.tensor(float('inf'), device=self.device)

        start.record()
        for epoch in tqdm(range(epochs), position=0, leave=True):
            train_loss = 0
            num_molecules_train = 0
            val_loss = 0
            num_molecules_val = 0
            self.model.train()
            for batch in train_loader:
                batch = batch.to(self.device)

                x, edge_index, batch_seg = batch.x, \
                    batch.edge_index, batch.batch

                optimizer.zero_grad()

                if train_graphnet:
                    pred = self.model(batch)
                else:
                    pred = self.model(x, edge_index, batch_seg)

                loss = loss_fn(
                    pred.view(-1, 1),
                    batch.spectrum.view(-1, 1)
                )
                loss.backward()
                train_loss += loss
                num_molecules_train += batch.num_graphs
                optimizer.step()
            scheduler.step()
            
            avg_train_loss = train_loss / num_molecules_train


            with torch.no_grad():
                self.model.eval()
                for batch in val_loader:
                    batch = batch.to(self.device)
                    x, edge_index, batch_seg = batch.x, \
                        batch.edge_index, batch.batch
                    if train_graphnet:
                        pred = self.model(batch)
                    else:
                        pred = self.model(x, edge_index, batch_seg)
                    loss = loss_fn(
                        pred.view(-1, 1), 
                        batch.spectrum.view(-1, 1)
                    )
                    val_loss += loss
                    num_molecules_val += batch.num_graphs
            avg_val_loss = val_loss / num_molecules_val

            if avg_val_loss < prev_loss:
                self._save_model()
                prev_loss = avg_val_loss

            if epoch % int(write_every) == 0:
                end.record()
                torch.cuda.synchronize()
                time=f"{start.elapsed_time(end)/6e4:.2f} mins"
                lr=round(float(scheduler.get_lr()[0]), 5)
                self.metrics.store_metrics(
                    mode='train',
                    epoch=epoch,
                    time=time,
                    loss=round(float(avg_train_loss), 2),
                    lr=lr
                    )
                self.metrics.store_metrics(
                    mode='val',
                    epoch=epoch,
                    time=time,
                    loss=round(float(avg_val_loss), 2),
                    lr=lr
                    )
                self.save_metrics()
                
                print(f"time = {time} mins")
                print(f"epoch {epoch} | average train loss = {avg_train_loss:.2f}",
                    f" and average validation loss = {avg_val_loss:.2f}",
                    f" |learning rate = {lr:.5f}")


    def predict(
        self, 
        molecule_graph, 
        model_path: str
        ):
        if model_path:
            assert model_path.endswith('.pt')
            self.model.load_state_dict(torch.load(model_path))

        molecule_graph = molecule_graph.to(self.device)
        x, edge_index = molecule_graph.x, molecule_graph.edge_index
        batch_seg = torch.tensor(np.repeat(0, x.shape[0]), device=self.device)

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            molecule_graph_pred = self.model(x, edge_index, batch_seg)
            x_pred = np.linspace(270, 300, 100)
            y_pred = molecule_graph_pred.cpu().numpy().flatten()
            y_true = molecule_graph.spectrum.cpu().numpy().flatten()
            
            df = pd.DataFrame(np.c_[x_pred, y_true, y_pred],
                              columns=['Energies', 'DFT/ROCIS', 'SpectraGNN'])

        fig, ax = plt.subplots()
        ax.plot(df.Energies, df.SpectraGNN, color='r', label='prediction')
        ax.plot(df.Energies, df['DFT/ROCIS'], color='black', label='DFT/ROCIS')
        ax.legend()
        ax.set_yticklabels([])
        ax.set_xlabel('Energies (eV)')
        ax.set_ylabel('Intensity (arb. units)')

    def save_metrics(self, path: str = './metrics/'):
        if not osp.exists(path):
            os.mkdir(path)
        for mode in self.metrics.modes:
            df = pd.DataFrame(self.metrics.outputs[mode])
            df.to_csv(osp.join(path, f"{mode}_metrics.csv"), index=False)
        

    def _save_model_params(self, path: str = './metrics/'):
        if not osp.exists(path):
            os.mkdir(path)

        params = {}
        for k, v in self.model.__dict__.items():
            if isinstance(v, (str, int, float, list)):
                params[k] = v
        
        with open(osp.join(path, 'params.json'), 'w') as fout:
            json.dump(params, fout)
       
    def _save_model(self):
        if not osp.exists('./best_model'):
            os.mkdir('./best_model')
        path = osp.join('./best_model', self.model_name + '.pt')
        torch.save(self.model.cpu().state_dict(), path)
        self.model.to(self.device)