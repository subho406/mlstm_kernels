import logging
from typing import Optional
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


def wrap_pad_inputs(backend, padded_chunk_size: int | None = 64):
    import torch

    if padded_chunk_size is None:
        return backend

    def _backend(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, i: torch.Tensor, f: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        B, N, S, H = q.shape  # (B, N, S, H)
        # padding to sequence lengths of 64 for kernels
        if S % padded_chunk_size != 0:
            # not optimized but usually only relevant in evaluation
            S_padded = ((S - 1) // padded_chunk_size + 1) * padded_chunk_size
            S_unpadded = S
            S = S_padded
            # actually these should not give NaNs as the model is causal.
            q_pad = q.new_zeros(B, N, S, q.shape[3])
            k_pad = k.new_zeros(B, N, S, k.shape[3])
            v_pad = (
                v.new_zeros(B, N, S, v.shape[3])
                + torch.arange(v.shape[3], dtype=v.dtype, device=v.device)[None, None, None, :]
            )
            # causality might have a problem here regarding NaNs? (D_matrix normalization)
            i_pad = i.new_zeros(B, N, S)
            f_pad = f.new_zeros(B, N, S)
            q_pad[:, :, :S_unpadded] = q
            k_pad[:, :, :S_unpadded] = k
            v_pad[:, :, :S_unpadded] = v
            i_pad[:, :, :S_unpadded] = i.reshape(B, N, S_unpadded)
            f_pad[:, :, :S_unpadded] = f.reshape(B, N, S_unpadded)
        else:
            S_unpadded = S
            q_pad = q
            k_pad = k
            v_pad = v
            i_pad = i.reshape(B, N, S_unpadded)
            f_pad = f.reshape(B, N, S_unpadded)
        h_state = backend(q_pad, k_pad, v_pad, i_pad, f_pad, **kwargs)
        return h_state[:, :, :S_unpadded]

    return _backend


def get_kernel(name: str, padded_chunk_size: int | None = 64) -> Callable:
    """
    Get a kernel function by name.

    Naming convention:
    name = "<module_name>--<backend_name>"

    module_name: The name of the module containing the kernel function.
        Example: "mlstm_chunkwise", "mlstm_parallel", "mlstm_recurrent", "flash_attention", "flash_linear_attention"

    backend_name: The name of the kernel function as defined in the registry in the __init__.py file of the module.
    """

    from .baselines.flash_attention import registry as flash_attention_registry
    from .baselines.flash_linear_attention import registry as flash_linear_attention_gla_registry
    from .mlstm.chunkwise import registry as mlstm_chunkwise_registry
    from .mlstm.parallel import registry as mlstm_parallel_registry
    from .mlstm.recurrent import registry_sequence as mlstm_recurrent_sequence_registry

    module_backend_registry = {
        "mlstm_recurrent_sequence": mlstm_recurrent_sequence_registry,
        "mlstm_chunkwise": mlstm_chunkwise_registry,
        "mlstm_parallel": mlstm_parallel_registry,
        "flash_attention": flash_attention_registry,
        "flash_linear_attention": flash_linear_attention_gla_registry,
    }

    module_name, backend_name = name.split("--")

    if module_name not in module_backend_registry:
        raise ValueError(
            f"Unknown module name: {module_name}. Available module names: {list(module_backend_registry.keys())}"
        )

    if backend_name not in module_backend_registry[module_name]:
        raise ValueError(
            f"Unknown backend name: {backend_name}. Available backend names: {list(module_backend_registry[module_name].keys())}"
        )

    # LOGGER.info(f"Calling backend: {module_name} {backend_name} with {padded_chunk_size}")
    backend = module_backend_registry[module_name][backend_name]
    return wrap_pad_inputs(backend, padded_chunk_size=padded_chunk_size)


def get_whole_registry(padded_chunk_size: int | None = 64):
    from .baselines.flash_attention import registry as flash_attention_registry
    from .baselines.flash_linear_attention import registry as flash_linear_attention_gla_registry
    from .mlstm.chunkwise import registry as mlstm_chunkwise_registry
    from .mlstm.parallel import registry as mlstm_parallel_registry
    from .mlstm.recurrent import registry_step as mlstm_recurrent_step_registry

    module_backend_registry = {
        "mlstm_recurrent_step": mlstm_recurrent_step_registry,
        "mlstm_chunkwise": mlstm_chunkwise_registry,
        "mlstm_parallel": mlstm_parallel_registry,
        "flash_attention": flash_attention_registry,
        "flash_linear_attention": flash_linear_attention_gla_registry,
    }

    _complete_kernel_registry = {
        module_name + "--" + backend_name: wrap_pad_inputs(backend, padded_chunk_size=padded_chunk_size)
        for module_name in module_backend_registry
        for backend_name, backend in module_backend_registry[module_name].items()
    }
    return _complete_kernel_registry
