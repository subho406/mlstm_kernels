from .max_triton_fwbw import mlstm_chunkwise_max_triton
from .max_triton_fwbw_v1 import mlstm_chunkwise_max_triton_v1
from .max_triton_fwbw_v2 import mlstm_chunkwise_max_triton_v2
from .max_triton_fwbw_v3 import mlstm_chunkwise_max_triton_v3
from .max_triton_fwbw_v3noslice import mlstm_chunkwise_max_triton_v3noslice
from .max_triton_fwbw_v5xlchunksize import mlstm_chunkwise_max_triton_v5xlchunksize
from .torch_fwbw import mlstm_chunkwise_torch_autograd, mlstm_chunkwise_torch_ownbw
from .torch_fwbw_v4 import (
    mlstm_chunkwise_torch_autograd_v4,
    mlstm_chunkwise_torch_ownbw_v4,
)
from .triton_fwbw import mlstm_fwbw as mlstm_chunkwise_stable_triton
from .triton_fwbw import mlstm_fwbw as mlstm_chunkwise_triton

registry = {
    "torch_autograd": mlstm_chunkwise_torch_autograd,
    "torch_ownbw": mlstm_chunkwise_torch_ownbw,
    "max_triton": mlstm_chunkwise_max_triton,  # fgate cumsum in dtype, intermediate states in dtype (kernel_dtype, e.g. float16, bfloat16)
    "max_triton_v1": mlstm_chunkwise_max_triton_v1,  # fgate cumsum in float32, intermediate states in float32
    "max_triton_v2": mlstm_chunkwise_max_triton_v2,  # fgate cumsum in float32, intermediate states in dtype (kernel_dtype, e.g. float16, bfloat16)
    "max_triton_v3": mlstm_chunkwise_max_triton_v3,  # complete fgate grad in float32, intermediate states in float32
    "max_triton_v3noslice": mlstm_chunkwise_max_triton_v3noslice,  # complete fgate grad in float32, intermediate states in float32, no slicing before kernel calls
    "max_triton_v5xlchunksize": mlstm_chunkwise_max_triton_v5xlchunksize,  # new work partitioning, enable larger chunk sizes, by using flash-attention work partitioning for intra chunks
    "triton": mlstm_chunkwise_triton,  # TODO integrate newest version
    "stable_triton": mlstm_chunkwise_stable_triton,
}
