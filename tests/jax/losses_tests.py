import jax
import jax.numpy as jnp


def loss_layernorm_offset_quadratic(
    input_tensor: jax.Array, seed: int = 0, eps: float = 1e-5
) -> jax.Array:
    rng_key = jax.random.PRNGKey(seed)
    rng_key, rng_key_offset = jax.random.split(rng_key)
    offset = jax.random.normal(rng_key_offset, input_tensor.shape)
    assert len(input_tensor.shape) == 4

    input_tensor_scaled = jax.nn.standardize(input_tensor, axis=-1)

    loss = jnp.sum((input_tensor_scaled + offset) ** 2)
    return loss
