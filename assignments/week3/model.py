import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    Our MLP class
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU(),
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.layer = torch.nn.Linear(input_size, hidden_size)
        # self.hidden_layers = torch.nn.ModuleList()
        initializer(self.layer.weight)
        self.activation = activation
        # input_dim = 32
        # self. hidden_units = [16, 8, 4]
        # A fully-connected network (FCN) with len(hidden_units) hidden layers
        self.hidden1 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden3 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden4 = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout1 = torch.nn.Dropout(0.5)  # 0.5
        self.dropout2 = torch.nn.Dropout(0.3)  # 0.25
        self.dropout3 = torch.nn.Dropout(0.15)  # 0.125
        self.dropout4 = torch.nn.Dropout(0.1)

        """
        for _ in range(1, hidden_count + 1):
            self.hidden_layers += [torch.nn.Linear(hidden_size, hidden_size)]
        """

        self.out = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> None:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = self.layer(x)
        act = self.activation
        x1 = self.dropout1(x)
        x1 = act(x)

        x1 = self.hidden1(x1)
        x1 = self.dropout2(x1)
        x1 = act(x1)

        x1 = self.hidden2(x1)
        x1 = self.dropout3(x1)
        x1 = act(x1)

        x1 = self.hidden3(x1)
        x1 = self.dropout4(x1)
        x1 = act(x)

        x1 = self.hidden4(x1)
        x1 = act(x)

        """
        x1 = act(x)
        for layer in self.hidden_layers:
            x = layer(x1)
            x1 = act(x)
        """
        # x_out = self.layer(x1)
        return self.out(x1)
