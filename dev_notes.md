# Dev Notes

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
python3 pip install -e .
# Go back to this repo
cd ../mlstm_kernels
```