from .torch_fwbw import mlstm_fw as mlstm_torch_autograd
from .torch_fwbw import mlstm_fwbw as mlstm_torch_ownbw
from .triton_fwbw import mlstm_fwbw as mlstm_triton

registry = {
    "torch_autograd": mlstm_torch_autograd,
    "torch_ownbw": mlstm_torch_ownbw,
    "triton": mlstm_triton,
}