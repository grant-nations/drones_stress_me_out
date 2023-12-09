import torch.nn as nn


class DemoNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        """
        Initialize the demographic model that encodes collaborator demographic data

        :param input_size: The number of expected demographic data features
        :param output_size: The number of features in the latent representation of the demographic data
        :param hidden_size: The number of features in the hidden state h
        """

        super(DemoNN, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = self.stack(x)
        return x
