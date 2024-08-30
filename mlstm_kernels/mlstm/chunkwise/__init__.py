from .torch_fwbw import mlstm_chunkwise_fw as mlstm_torch_autograd 
from .torch_fwbw import mlstm_chunkwise_fwbw as mlstm_torch_ownbw
from .max_triton_fwbw.triton_fwbw import mlstm_chunkwise_fwbw as max_mlstm_fwbw
from .triton_fwbw import mlstm_fwbw

registry = {
    "torch_autograd": mlstm_torch_autograd,
    "torch_ownbw": mlstm_torch_ownbw,
    "triton_max": max_mlstm_fwbw,
    "triton": mlstm_fwbw,
}
