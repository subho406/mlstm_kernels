from .native_sequence import (
    mlstm_recurrent_sequence__native_fw,
    mlstm_recurrent_sequence__triton_alternate_step_fw,
    mlstm_recurrent_sequence__triton_step_fused_fw,
)
from .native_step import mlstm_recurrent_step__native
from .triton_step_alternate import mlstm_recurrent_step__triton_alternate
from .triton_step_fused import mlstm_recurrent_step__triton_fused

registry_step = {
    "native": mlstm_recurrent_step__native,
    # "triton_alternate": mlstm_recurrent_step__triton_alternate,
    "triton_fused": mlstm_recurrent_step__triton_fused,
}


registry_sequence = {
    "native_sequence__native": mlstm_recurrent_sequence__native_fw,
    # "native_sequence__triton_alternate_step": mlstm_recurrent_sequence__triton_alternate_step_fw,
    "native_sequence__triton_step_fused": mlstm_recurrent_sequence__triton_step_fused_fw,
}
