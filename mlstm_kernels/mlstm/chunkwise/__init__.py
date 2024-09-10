from .torch_fwbw import mlstm_chunkwise_torch_autograd
from .torch_fwbw import mlstm_chunkwise_torch_ownbw
from .max_triton_fwbw import mlstm_chunkwise_max_triton
from .max_triton_fwbw_v1 import mlstm_chunkwise_max_triton_v1
from .max_triton_fwbw_v2 import mlstm_chunkwise_max_triton_v2
from .triton_fwbw import mlstm_fwbw as mlstm_chunkwise_triton

registry = {
    "torch_autograd": mlstm_chunkwise_torch_autograd,
    "torch_ownbw": mlstm_chunkwise_torch_ownbw,
    "max_triton": mlstm_chunkwise_max_triton,
    "max_triton_v1": mlstm_chunkwise_max_triton_v1,
    "max_triton_v2": mlstm_chunkwise_max_triton_v2,
    "triton": mlstm_chunkwise_triton,  # TODO integrate newest version
}
