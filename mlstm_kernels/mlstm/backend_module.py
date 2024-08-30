import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class mLSTMBackendConfig:
    pass


class mLSTMBackend(nn.Module):
    config_class = mLSTMBackendConfig

    def __init__(self, config: mLSTMBackendConfig):
        super().__init__()
        self.config = config

    def forward(
        self, *args, **kwargs
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        pass


