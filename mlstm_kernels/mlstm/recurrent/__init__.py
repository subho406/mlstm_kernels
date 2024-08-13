from .torch_fw import recurrent_step_fw 
from .triton_fw import recurrent_step_fw as recurrent_step_fw_triton
from .triton_fused_fw import recurrent_step_fw as recurrent_step_fw_triton_fused

registry = {
    "torch": recurrent_step_fw,
    "triton": recurrent_step_fw_triton,
    "triton_fused": recurrent_step_fw_triton_fused,
}