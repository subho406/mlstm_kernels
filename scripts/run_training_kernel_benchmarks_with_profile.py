#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from functools import partial
from pathlib import Path

import torch
from dacite import from_dict
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from mlstm_kernels.utils.benchmark.benchmarks.training_kernel_benchmarks import (
    create_training_kernel_benchmark,
)
from mlstm_kernels.utils.benchmark.param_handling import BenchmarkConfig
from mlstm_kernels.utils.benchmark.run_benchmark import (
    run_and_record_benchmarks,
    run_benchmarks,
)
from mlstm_kernels.utils.benchmark.utils import setup_output_folder

run_training_benchmarks = partial(
    run_benchmarks, benchmark_creator=create_training_kernel_benchmark
)


def _benchmark_to_profile(output_folder: Path):
    B = 8
    S = 8192

    cfg_yaml = f"""
vary_type: grid
vary_params:

fixed_params:
  batch_size: {B}
  sequence_length: {S}
  rep: 25
  warmup: 5

x_axis_param: "batch_size"

kernel_specs:
  - kernel_name: "mlstm_chunkwise__xl_chunk"
    dtype: bfloat16
    fwbw: True
    use_torch_compile: True
    additional_params:
      head_dim_qk: 256
      head_dim_v: 512
      num_heads: 4
      chunk_size_inter: 128
      chunk_size_intra: 128

      siz_b_L_parallel: 64
      siz_b_L_loop: 64
      siz_b_DH_parallel: 128
      siz_b_DH_loop: 64

      num_warps_intra: 4
      num_warps_inter: 4
      num_stages_intra: 1
      num_stages_inter: 1
      recompute_states_in_bw: False

benchmark_name: "compare to flash_attention"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)


def run_multiple_benchmarks(output_dir: str = "./outputs_kernel_benchmarks_profiler"):
    output_folder = setup_output_folder(output_dir)

    # _sequence_length_benchmark(output_folder, batch_size=1, num_heads=16, head_dim=256)
    # _batch_size_benchmark(output_folder, seq_len=8192, num_heads=16, head_dim=256)
    # _sequence_length_benchmark(output_folder, batch_size=1, num_heads=8, head_dim=512)
    # _batch_size_benchmark(output_folder, seq_len=8192, num_heads=8, head_dim=512)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
    sort_by_keyword = "cuda_time_total"
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        use_cuda=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            output_folder / "tensorboard"
        ),
    ) as prof:
        _benchmark_to_profile(output_folder)

    print(
        prof.key_averages().table(
            sort_by=sort_by_keyword, row_limit=50, max_name_column_width=100
        )
    )


if __name__ == "__main__":
    run_multiple_benchmarks()
