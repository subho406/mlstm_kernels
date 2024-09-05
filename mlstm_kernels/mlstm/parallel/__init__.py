from .torch_fwbw import mlstm_parallel_torch_autograd
from .torch_fwbw import mlstm_parallel_torch_ownbw
from .stable_torch_fwbw import (
    mlstm_parallel_torch_autograd as mlstm_parallel_stable_torch_autograd,
)
from .stable_torch_fwbw import (
    mlstm_parallel_torch_ownbw as mlstm_parallel_stable_torch_ownbw,
)
from .triton_fwbw.triton_fwbw import mlstm_parallel_triton

registry = {
    "torch_autograd": mlstm_parallel_torch_autograd,
    "torch_ownbw": mlstm_parallel_torch_ownbw,
    "stable_torch_autograd": mlstm_parallel_stable_torch_autograd,
    "stable_torch_ownbw": mlstm_parallel_stable_torch_ownbw,
    "triton": mlstm_parallel_triton,
}
