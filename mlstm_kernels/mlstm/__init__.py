import torch
from typing import Callable


def mlstm_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = False,
) -> (
    torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    pass


def _create_module_backend_registry() -> dict[str, dict[str, Callable]]:
    from .chunkwise import registry as mlstm_chunkwise_registry
    from .parallel import registry as mlstm_parallel_registry
    from .recurrent import registry_step as mlstm_recurrent_step_registry
    from .recurrent import registry_sequence as mlstm_recurrent_sequence_registry

    module_backend_registry = {
        "recurrent_step": mlstm_recurrent_step_registry,
        "recurrent_sequence": mlstm_recurrent_sequence_registry,
        "chunkwise": mlstm_chunkwise_registry,
        "parallel": mlstm_parallel_registry,
    }
    return module_backend_registry


def get_mlstm_kernel(name: str) -> Callable:
    """
    Get a kernel function by name.

    Naming convention:
    name = "<module_name>--<backend_name>"

    module_name: The name of the module containing the kernel function.
        Example: "chunkwise", "parallel", "recurrent"

    backend_name: The name of the kernel function as defined in the registry in the __init__.py file of the module.
    """

    module_backend_registry = _create_module_backend_registry()

    module_name, backend_name = name.split("--")

    if module_name not in module_backend_registry:
        raise ValueError(
            f"Unknown module name: {module_name}. Available module names: {list(module_backend_registry.keys())}"
        )

    if backend_name not in module_backend_registry[module_name]:
        raise ValueError(
            f"Unknown backend name: {backend_name}. Available backend names: {list(module_backend_registry[module_name].keys())}"
        )

    return module_backend_registry[module_name][backend_name]


def get_available_mlstm_kernels() -> list[str]:
    module_backend_registry = _create_module_backend_registry()

    backend_names = [
        f"{module_key}--{kernel_key}"
        for module_key in module_backend_registry.keys()
        for kernel_key in module_backend_registry[module_key].keys()
    ]
    return backend_names
