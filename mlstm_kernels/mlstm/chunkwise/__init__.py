from .torch_fwbw import mlstm_chunkwise_torch_autograd
from .torch_fwbw import mlstm_chunkwise_torch_ownbw
from .max_triton_fwbw import mlstm_chunkwise_max_triton
from .max_triton_fwbw_v1 import mlstm_chunkwise_max_triton_v1
from .max_triton_fwbw_v2 import mlstm_chunkwise_max_triton_v2
from .max_triton_fwbw_v3 import mlstm_chunkwise_max_triton_v3
from .triton_fwbw import mlstm_fwbw as mlstm_chunkwise_triton
from .triton_fwbw import mlstm_fwbw as mlstm_chunkwise_triton_stable

registry = {
    "torch_autograd": mlstm_chunkwise_torch_autograd,
    "torch_ownbw": mlstm_chunkwise_torch_ownbw,
    "max_triton": mlstm_chunkwise_max_triton,  # fgate cumsum in dtype, intermediate states in dtype (kernel_dtype, e.g. float16, bfloat16)
    "max_triton_v1": mlstm_chunkwise_max_triton_v1,  # fgate cumsum in float32, intermediate states in float32
    "max_triton_v2": mlstm_chunkwise_max_triton_v2,  # fgate cumsum in float32, intermediate states in dtype (kernel_dtype, e.g. float16, bfloat16)
    "max_triton_v3": mlstm_chunkwise_max_triton_v3,  # complete fgate grad in float32, intermediate states in float32
    "triton": mlstm_chunkwise_triton,  # TODO integrate newest version
    "triton_stable": mlstm_chunkwise_triton_stable,
}
