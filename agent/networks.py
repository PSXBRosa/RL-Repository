import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, json_file):
        super().__init__()

        label = {
        "Dense"   : nn.Linear,
        "Conv2d"  : nn.Conv2d,
        "relu"    : nn.ReLU,
        "softmax" : nn.Softmax
        }

        self._l         = json_file["config"]["layers"]
        self._name      = json_file["config"]["name"]
        self._hparams   = json_file["config"]["hyperparameters"]
        self._layers    = []
        for layer in self._l:
            class_name  = layer["class_name"]
            input_shape = layer["config"]["input_shape"]
            units       = layer["config"]["units"]
            activation  = layer["config"]["activation"]
            
            if class_name == "Dense":
                l1 = label[class_name](input_shape, units)
            elif class_name == "Conv2d":
                kernel  = layer["config"]["kernel"]
                stride  = layer["config"]["stride"]
                padding = layer["config"]["padd"]
                l1 = label[class_name](input_shape, units, kernel, stride, padding)
            self._layers.append(l1)

            if activation != "linear":
                if activation == "softmax":
                    l2 = label[activation]()
                else:
                    l2 = label[activation]()
                self._layers.append(l2)

        self._nn = nn.ModuleList(self._layers)

    def forward(self, x):
        x = x.view(x.size()[0],-1) # Flatten
        for layer in self._nn:
            x = layer(x)
        return x.squeeze(0)