from abc import ABC
from typing import Any
from typing import Optional

from amp.models import model


class Discriminator(model.Model, ABC):
    """ Perform classification on a sequence."""

    def output_tensor(self, input_: Optional[Any] = None):
        raise NotImplementedError

    def output_tensor_with_dense_input(self, input_: Optional[Any] = None):
        raise NotImplementedError

    def __call__(self, input_: Optional[Any] = None):
        raise NotImplementedError

    def freeze_layers(self):
        raise NotImplementedError

    def unfreeze_layers(self):
        raise NotImplementedError

    def train_for_n_epochs_with_batch_size(self, epochs: int, batch_size: int, **kwargs):
        raise NotImplementedError
