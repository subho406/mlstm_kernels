from typing import Callable


def get_kernel(name: str) -> Callable:
    """
    Get a kernel function by name.

    Naming convention:    
    name = "<module_name>--<backend_name>"

    module_name: The name of the module containing the kernel function.
        Example: "mlstm_chunkwise", "mlstm_parallel", "mlstm_recurrent", "flash_attention", "flash_linear_attention"

    backend_name: The name of the kernel function as defined in the registry in the __init__.py file of the module.
    """

    from mlstm_kernels.baselines.flash_attention import registry as flash_attention_registry
    from mlstm_kernels.mlstm.chunkwise import registry as mlstm_chunkwise_registry
    from mlstm_kernels.mlstm.parallel import registry as mlstm_parallel_registry
    from mlstm_kernels.mlstm.recurrent import registry_step as mlstm_recurrent_step_registry
    from mlstm_kernels.baselines.flash_linear_attention import registry as flash_linear_attention_gla_registry

    module_backend_registry = {
        "mlstm_recurrent_step": mlstm_recurrent_step_registry,
        "mlstm_chunkwise": mlstm_chunkwise_registry,
        "mlstm_parallel": mlstm_parallel_registry,
        "flash_attention": flash_attention_registry,
        "flash_linear_attention": flash_linear_attention_gla_registry,
    }
    
    module_name, backend_name = name.split("--")

    if module_name not in module_backend_registry:
        raise ValueError(f"Unknown module name: {module_name}. Available module names: {list(module_backend_registry.keys())}")
    
    if backend_name not in module_backend_registry[module_name]:
        raise ValueError(f"Unknown backend name: {backend_name}. Available backend names: {list(module_backend_registry[module_name].keys())}")
    
    return module_backend_registry[module_name][backend_name]