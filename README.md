# XASNet - Graph Neural Network models to predict X-ray absorption spectra

![generated molecules](./images/XASNet.png)

XASNet is a graph neural network model to predict X-ray absorption spectra (XAS) of small molecules while maintaing the explainibility of the predicted spectra. It can be chosen based on different architectures, i.e. [GraphNet](https://arxiv.org/abs/1806.01261), [graph convolutional neural network (GCN)](https://arxiv.org/abs/1509.09292), [multi-head graph attention network (GATv2)](https://arxiv.org/abs/1710.10903). XASNet can be trained on datasets of 3d molecules with variable sizes composed of the first- and second row of main group elements H, C, N, O, and F. To explain the predictions, feature attributions are employed to determine the respective contributions of various atoms in the molecules to the peaks observed in the XAS spectrum.

