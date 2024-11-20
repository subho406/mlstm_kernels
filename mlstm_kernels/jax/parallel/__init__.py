from .native import mlstm_parallel__native_autograd, mlstm_parallel__native_custbw
from .native_stablef import (
    mlstm_parallel__native_stablef_autograd,
    mlstm_parallel__native_stablef_custbw,
)

registry = {
    "native_autograd": mlstm_parallel__native_autograd,
    "native_custbw": mlstm_parallel__native_custbw,
    "native_stablef_autograd": mlstm_parallel__native_stablef_autograd,
    "native_stablef_custbw": mlstm_parallel__native_stablef_custbw,
}
