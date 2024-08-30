from .torch_fwbw import mlstm_chunkwise_torch_autograd 
from .torch_fwbw import mlstm_chunkwise_torch_ownbw
from .max_triton_fwbw import mlstm_chunkwise_max_triton
from .triton_fwbw import mlstm_fwbw as mlstm_chunkwise_triton

registry = {
    "torch_autograd": mlstm_chunkwise_torch_autograd,
    "torch_ownbw": mlstm_chunkwise_torch_ownbw,
    "max_triton": mlstm_chunkwise_max_triton,
    "triton": mlstm_chunkwise_triton, # TODO integrate newest version
}
