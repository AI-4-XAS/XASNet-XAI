import argparse
import os
import os.path as osp
import torch 
import numpy as np
from .broadening import group_broadening
from ..attribution_gnn1.QM9_SpecData import QM9_SpecData
from ..attribution_gnn1.split import save_split
from torch_geometric.loader import DataLoader
from SpectraGNN import SpectraGNN
from .trainer_old import batch_train, batch_val

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='muted', font_scale=1.5)

from pylab import rc, rcParams
rc('text', usetex=False)
rc('axes', linewidth=2)
rc('font', weight='normal')

params = {'legend.fontsize': 20,
          'figure.figsize': (8, 6),}
rcParams.update(params)




parser = argparse.ArgumentParser(description="training the spectragnn model based on qm9 spectra dataset")

parser.add_argument('--path_to_data', type=str, metavar='S',
help='path to the data folder')
parser.add_argument('--cuda', action='store_true', default=False,
help='whether using gpu')
parser.add_argument('--gnn_name', type=str, default='gat', metavar='G',
help='name of the graph interaction layer')
parser.add_argument('--in_channels', type=int, nargs='+', metavar='N',
help='input channels to the model')
parser.add_argument('--out_channels', type=int, nargs='+', metavar='N',
help='output channels to the model')
parser.add_argument('--nlayers', type=int, metavar='N',
help='number of gnn layers')
parser.add_argument('--learning_rate', type=float, metavar='N',
help='optimizer lr')
parser.add_argument('--scheduler', action='store_true', default=False,
help='whether using scheduler in lr')
parser.add_argument('--path_to_model', type=str, default='./best_model', metavar='S',
help='path to save and load the model')
parser.add_argument('--num_epochs', type=int, metavar='N',
help='number of epochs for training')

args = parser.parse_args()



#device
if args.cuda and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# qm9_spectra dataset 
qm9_spec = QM9_SpecData(args.path_to_data, 
                       './data', spectra=broadened_spectra_stk)
if not osp.exists('./data/qm9_spec.pt'):
    torch.save(qm9_spec, './data/qm9_spec.pt')


# splitting the data
idxs = save_split(
    path=args.path_to_data,
    ndata=len(qm9_spec),
    ntrain=8000,
    nval=1500,
    save_split=True,
    shuffle=True, 
    print_nsample=True
)

train_qm9 = qm9_spec[idxs['train']]
val_qm9 = qm9_spec[idxs['val']]
test_qm9 = qm9_spec[idxs['test']]

#data loaders 
train_loader = DataLoader(train_qm9, batch_size=64, shuffle=True)
val_loader = DataLoader(val_qm9, batch_size=64, shuffle=True)
test_loader = DataLoader(test_qm9, batch_size=64)

# model, optimizer and loss function
model = SpectraGNN(
    gnn_name=args.gnn_name,
    in_channels=args.in_channels,
    out_channels=args.out_channels,
    num_targets=100,
    num_layers=args.nlayers,
    heads=3
)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
loss_fn = torch.nn.L1Loss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                 milestones=np.arange(500, 5000, 500).tolist(),
                                                 gamma=0.8)

#function to save the model
def save_model(model, device):
    if not osp.exists(args.path_to_model):
        os.mkdir(args.path_to_model)
    model_name = 'spectragnn.pt'
    path = os.path.join('./best_model', model_name)
    torch.save(model.cpu().state_dict(), path)
    model.to(device)

# load the best model 
model.load_state_dict(torch.load(args.path_to_model))

#timer
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# plotting the loss functon
def loss_plot(epochs: int, 
              loss_values: dict, 
              ylabel: str = None,
              plot_test_loss: bool = False,
              save: bool = False
             ):
    assert isinstance(loss_values, dict)
    fig, ax = plt.subplots()
    sns.lineplot(epochs, loss_values['train'], ax=ax, label='train')
    if plot_test_loss:
        sns.lineplot(epochs, loss_values['validation'], ax=ax, label='validation') 
    ax.set_xlabel('Epochs')
    ax.set_ylabel('L1 loss')
    if save:
        fig.savefig('./train_test.png', dpi=200)


if __name__ == "__main__":
    train_loss_values = []
    val_loss_values = []
    epochs = []
    prev_loss = torch.tensor(float('inf'), device=device)
    start.record()
    for epoch in range(args.num_epochs):
        train_avg_loss = batch_train(model, train_loader, optimizer, 
        loss_fn, scheduler, device)
        val_avg_loss = batch_val(model, val_loader, 
        loss_fn, device)

        if val_avg_loss < prev_loss:
            #print('saving the model')
            save_model(model, device)           
            prev_loss = val_avg_loss
        
        if epoch % 50 == 0:
            end.record()
            torch.cuda.synchronize()
            epochs.append(epoch)
            train_loss_values.append(train_avg_loss.cpu().detach().numpy())
            val_loss_values.append(val_avg_loss.cpu().detach().numpy())
            print(f"time = {start.elapsed_time(end)/6e4:.2f} mins")
            print(f"epoch {epoch} | average train loss = {train_avg_loss:.2f}",
                f" and average validation loss = {val_avg_loss:.2f}",
                f" |learning rate = {scheduler.get_lr()[0]:.5f}")
