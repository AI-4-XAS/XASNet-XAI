import torch

def standardize(kernel):
    if len(kernel.shape) == 3:
        axis=[0, 1]
    else:
        axis=1
    var, mean = torch.var_mean(kernel, dim=axis, keepdim=True)
    kernel = (kernel - mean) / (var) ** 0.5
    return kernel


def kaiming_orthogonal_init(tensor):
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numul()
    else:
        fan_in = tensor.shape[1]
    
    with torch.no_grad():
        tensor = standardize(tensor)
        tensor *= (1/fan_in)**0.5
    return tensor