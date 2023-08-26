import numpy as np

class MetricManager():
    """
    Class for managing the training of the model.
    """
    def __init__(self, modes = ['train', 'val']):
        self.modes = modes
        self.outputs = {}

        for mode in modes:
            self.outputs[mode] = {}
            self.outputs[mode]['epoch'] = []
            self.outputs[mode]['time'] = []
            self.outputs[mode]['loss'] = []
            self.outputs[mode]['lr'] = []

    def store_metrics(
        self, 
        mode: str, 
        epoch: int, 
        time: float, 
        loss: float, 
        lr: float
        ):

       self.outputs[mode]['epoch'].append(epoch)
       self.outputs[mode]['time'].append(time)
       self.outputs[mode]['loss'].append(loss) 
       self.outputs[mode]['lr'].append(lr)

    def best_metric(self, mode: str = 'val'):
        best_results = {}

        i = np.array(self.outputs[mode]['loss']).argmax()

        for key in self.outputs[mode].keys():
            best_results[key] = self.outputs[mode][key][i]

        return best_results