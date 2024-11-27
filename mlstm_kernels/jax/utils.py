import jax
import jax.numpy as jnp
import numpy as np
import triton.language as tl


def dtype2str(dtype: jnp.dtype) -> str:
    if dtype == jnp.float32:
        return "fp32"
    elif dtype == jnp.float16:
        return "fp16"
    elif dtype == jnp.float64:
        return "fp64"
    elif dtype == jnp.bfloat16:
        return "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def jax2triton_dtype(dtype):
    """
    Converts a JAX dtype to a Triton dtype.

    Args:
        dtype: JAX dtype.

    Returns:
        Triton dtype.
    """
    if hasattr(dtype, "dtype"):
        dtype = dtype.dtype
    return getattr(tl, str(dtype))


def to_numpy(tensor: jnp.ndarray) -> np.ndarray:
    return jax.device_get(tensor)
