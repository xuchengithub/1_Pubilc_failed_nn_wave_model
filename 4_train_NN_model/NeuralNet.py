import torch.nn as nn
import torch.nn.functional as F
import torch

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_hidden_layers, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # Create remaining hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(0, num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        input = x.view(x.size(0), -1)
        output = self.input_layer(input)
        input = self.relu(output)

        for layer in self.hidden_layers:
            output = layer(input)
            input = self.relu(output)

        output = self.output_layer(input)
        out = torch.sigmoid(output)
        # no activation and no softmax at the end
        return out
