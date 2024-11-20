from .native_sequence import (
    mlstm_recurrent_sequence__native_fw,
    mlstm_recurrent_sequence__triton_step_fused_fw,
    mlstm_recurrent_sequence__triton_step_fw,
)
from .native_step import mlstm_recurrent_step__native
from .triton_step import mlstm_recurrent_step__triton
from .triton_step_fused import mlstm_recurrent_step__triton_fused

registry_step = {
    "native": mlstm_recurrent_step__native,
    "triton": mlstm_recurrent_step__triton,
    "triton_fused": mlstm_recurrent_step__triton_fused,
}
