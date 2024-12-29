import triton

from ..utils.kernels import is_power_of_2


def get_head_dim_block_size(head_dim: int, min_block_size: int = 64) -> int:

    assert (
        is_power_of_2(head_dim) or head_dim % min_block_size == 0
    ), f"head_dim must be a power of 2 or multiple of {min_block_size}. Got {head_dim}."

    return min(min_block_size, triton.next_power_of_2(head_dim))