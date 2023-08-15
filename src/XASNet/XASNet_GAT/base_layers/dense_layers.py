import torch
from torch.nn import Linear, ReLU, ELU, SiLU

def he_kaiming_init(tensor):
    tensor = torch.nn.init.orthogonal_(tensor)
    if len(tensor.shape) == 3:
        axis=[0, 1]
    else:
        axis=1
    mean, var = torch.var_mean(tensor, dim=axis, keepdim=True)
    tensor = (tensor - mean) / (var) ** 0.5

    fan_in = tensor.shape[1]

    tensor *= (1 / fan_in) ** 0.5
    return tensor


class LinearLayer(torch.nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        bias: bool = False,
        activation: str = None,
        ):
        super().__init__()

        activations = {'relu' : ReLU(inplace=True), 
        'elu' : ELU(inplace=True), 
        'silu' : SiLU()}
        if activation is not None:
            assert activation.lower() in activations.keys()

        self.linear = Linear(in_channels, out_channels, bias=bias)
        if activation is None:
            self._activation = torch.nn.Identity()
        else:
            self._activation = activations[activation]
        
        self.reset_parameters()
        self.weight = self.linear.weight
        self.bias = self.linear.bias

    def reset_parameters(self):
        he_kaiming_init(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        out = self._activation(x)
        return out

class Residual_block(torch.nn.Module):
    def __init__(
        self,
        units: int,
        n_layers: int,
        activation: str = None
        ):
        super().__init__()

        all_layers = []

        for i in range(n_layers):
            all_layers.append(
                LinearLayer(units, units, 
                bias=False, activation=activation)
            )
        
        self.res_layers = torch.nn.Sequential(*all_layers)

    def forward(self, inputs):
        x = self.res_layers(inputs)
        x = inputs + x
        return x