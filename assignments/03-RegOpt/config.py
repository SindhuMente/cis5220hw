from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor

# Normalize


class CONFIG:
    """
    Our configuration of a scheduler
    """

    """config 1"""
    batch_size = 64
    num_epochs = 6
    initial_learning_rate = 0.001
    initial_weight_decay = initial_learning_rate / num_epochs

    # You can pass arguments to the learning rate scheduler
    # constructor here.
    lrs_kwargs = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "initial_learning_rate": initial_learning_rate,
        "initial_weight_decay": initial_weight_decay,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
        ]
    )
