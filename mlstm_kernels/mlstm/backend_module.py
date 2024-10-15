import torch
from torch import nn
from dataclasses import dataclass
from typing import Literal

from . import get_mlstm_kernel


@dataclass
class mLSTMBackendConfig:
    # TODO: mbeck: is there a way to make this dynamic?
    kernel_name: Literal[
        "recurrent_sequence--sequence_torch_autograd",
        "chunkwise--torch_autograd",
        "chunkwise--torch_ownbw",
        "chunkwise--max_triton",
        "chunkwise--max_triton_v1",
        "chunkwise--max_triton_v2",
        "chunkwise--max_triton_v3",
        "chunkwise--triton",
        "parallel--torch_autograd",
        "parallel--torch_ownbw",
        "parallel--triton",
    ] = "chunkwise--triton"
    chunk_size: int = 64
    autocast_kernel_dtype: Literal["float32", "bfloat16", "float16"] = "float16"


class mLSTMBackend(nn.Module):
    config_class = mLSTMBackendConfig

    def __init__(self, config: mLSTMBackendConfig):
        super().__init__()
        self.config = config
        self.backend_fn = get_mlstm_kernel(config.kernel_name)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        f: torch.Tensor,
        c_initial: torch.Tensor = None,
        n_initial: torch.Tensor = None,
        m_initial: torch.Tensor = None,
        return_last_states: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        return self.backend_fn(
            q=q,
            k=k,
            v=v,
            i=i,
            f=f,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
            return_last_states=return_last_states,
            chunk_size=self.config.chunk_size,
            autocast_kernel_dtype=getattr(torch, self.config.autocast_kernel_dtype),
        )
