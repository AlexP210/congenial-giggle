import abc
import typing
import torch
from torch import nn

class LossFunctionBase(abc.ABC):

    def __init__(self, cfg):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, batch:torch.Tensor, model) -> typing.Tuple[torch.Tensor, typing.Dict[str, typing.Any]]:
        pass