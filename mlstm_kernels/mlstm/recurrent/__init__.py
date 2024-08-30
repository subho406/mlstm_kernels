from .torch_fw import recurrent_step_fw as mlstm_recurrent_step_torch_autograd
from .torch_fw import mlstm_recurrent_sequence_torch_autograd
from .triton_fw import recurrent_step_fw as mlstm_recurrent_step_triton
from .triton_fused_fw import recurrent_step_fw as mlstm_recurrent_step_fused_triton

registry_step = {
    "step_torch_autograd": mlstm_recurrent_step_torch_autograd,
    "sequence_torch_autograd": mlstm_recurrent_sequence_torch_autograd,
    "step_triton": mlstm_recurrent_step_triton,
    "step_fused_triton": mlstm_recurrent_step_fused_triton,
}