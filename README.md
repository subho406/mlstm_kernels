# mLSTM Kernels

In this repository we collect implementations of the different mLSTM formulations.

## External kernel interface with names

```python
def mlstm_interface(
    q: torch.Tensor, # (B, NH, S, DHQK)
    k: torch.Tensor, # (B, NH, S, DHQK)
    v: torch.Tensor, # (B, NH, S, DHHV)
    i: torch.Tensor, # (B, NH, S)
    f: torch.Tensor, # (B, NH, S)
    c_initial: torch.Tensor = None, # (B, NH, DHQK, DHHV)
    n_initial: torch.Tensor = None, # (B, NH, DHQK)
    m_initial: torch.Tensor = None, # (B, NH, 1)
    return_last_states: bool = False,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.bfloat16,
    chunk_size: int = 64,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # (B, NH, S, DHHV) | ((B, NH, S, DHHV), ((B, NH, DHQK, DHHV), (B, NH, DHQK), (B, NH)))
    """
    Returns:
        torch.Tensor: matH outputs (no n and m values, no last states)
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: matH, (matC_last, vecN_last, scaM_last)
    """

    pass

```

```python
def mlstm_step_interface(
    q: torch.Tensor,  # (B, NH, DHQK)
    k: torch.Tensor,  # (B, NH, DHQK)
    v: torch.Tensor,  # (B, NH, DHHV)
    i: torch.Tensor,  # (B, NH, 1)
    f: torch.Tensor,  # (B, NH, 1)
    c: torch.Tensor,  # (B, NH, DHQK, DHHV)
    n: torch.Tensor,  # (B, NH, DHQK)
    m: torch.Tensor,  # (B, NH, 1)
    eps: float = 1e-6,
    **kwargs,
) -> tuple[
    torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:  # vecH, (matC_state_new (B, NH, DHQK, DHHV), vecN_state_new (B, NH, DHQK), vecM_state_new (B, NH, 1))
    pass
```

## Kernel variants

The mLSTM repo contains the following kernel variants:

- `chunkwise`: chunkwise kernels like flash linear attention (sub-quadratic)
- `parallel`: parallel kernels like flash attention (quadratic)
- `recurrent`: recurrent kernels (mostly for inference) (linear)

Not all variants support all features of the interface. Only the chunkwise and recurrent support passing the initial states and returning the last states.

### Kernel naming

#### External names of kernel functions in chunkwise, parallel and recurrent modules

- Python function: `mlstm_[recurrent|parallel|chunkwise]_[specifier]_[triton|torch]_[[autograd|ownbw]]`
- Registry name (within module): `[specifier]_[triton|torch]_[[autograd|ownbw]]`

### Available Kernels

You can view all available kernels for the mLSTM by calling

```python
from mlstm_kernels.mlstm import get_available_mlstm_kernels

get_available_mlstm_kernels()
```


## Running the unit tests

The unit tests cross-check the different kernel implementations on numerical deviations for different dtypes.
You can run all of them with the following command:

```bash
pytest -s tests/torch
# make sure you are in a JAX GPU environment
pytest -s tests/jax
```

The `-s` disables the log capturing so you see the results directly on the command line.
Each test will log the outputs to a new folder with the timestamp as name in the `test_outputs/` directory.

Example:
Each test starts with the line
`Test chunkwise-triton_xl_chunk target=triton_chunkwise_xl_chunk vs. baseline=native_parallel_stablef_custbw with S=256, B=1, NH=2, DHQK=64, DHHV=128, DTYPE=torch.float32`.

This test tests the chunkwise triton kernel `triton_chunkwise_xl_chunk` against the `native_parallel_stablef_custbw` baseline and runs the `triton_chunkwise_xl_chunk` in dtype float32. It will compare the errors against the baseline in the same dtype (i.e. float32 here) and in float64 if specified.

## Profiling Kernels with Nsight Systems & Nsight Compute

### Nsight Systems

Documentation: <https://docs.nvidia.com/nsight-systems/UserGuide/#cli-profiling>

Command:

```bash
PYTHONPATH=. nsys profile -t cuda,osrt,nvtx,cudnn,cublas -w true -o ./nvidia_nsight/nsys_mlstm_xlchunksize python scripts/run_training_kernel_benchmarks_with_profile.py
```

### Nsight Compute

Documentation: <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>

Command:

```bash
PYTHONPATH=. ncu -o kernel_prof -f -c 1 -k mlstm_chunkwise__parallel_fw_Hintra_kernel --set=full python ./scripts/run_training_kernel_benchmarks_with_profile.py
```

## Running kernel benchmarks with baselines

To run the benchmarks including all baselines, you have to install:
```bash
pip install mamba_ssm causal_conv1d fla
```
For `FlashAttention3`, you have to clone the original repo `https://github.com/Dao-AILab/flash-attention`:
```bash
# clone FlashAttention
cd ..
git clone https://github.com/Dao-AILab/flash-attention
# Apply CONDA ENV patch
git apply ../mlstm_kernels/flash_attention.patch
# Install flash attention 3
cd hopper
PYTHONPATH=. python3 setup.py install
cd ..
# Install regular flash attention 2
python3 install -e .
# Go back to this repo
cd ../mlstm_kernels
```