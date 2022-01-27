from abc import ABC
from typing import Any
from typing import Optional

from amp.models import model

class Decoder(model.Model, ABC):
    def output_tensor(self, input_: Optional[Any] = None):
        raise NotImplementedError

    def __call__(self, input_: Optional[Any] = None):
        raise NotImplementedError
