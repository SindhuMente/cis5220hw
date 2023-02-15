from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    '''
    Our implentation of a scheduler
    '''
    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        batch_size=1,
        num_epochs=1,
        initial_learning_rate=1,
        initial_weight_decay=1,
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.initial_learning_rate = initial_learning_rate
        self.initial_weight_decay = initial_weight_decay

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:

        """****** Baseline ******"""
        # return [i for i in self.base_lrs]

        """Worked somewhat ok"""
        return [i + (i - 1) * 0.0001 for i in self.base_lrs]

        # return [0.05, 0.01, 0.001, 0.01, 0.02, 0.03]