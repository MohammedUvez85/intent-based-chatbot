
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.output_layer(x)

        return x
