import argparse
import logging
import os
from pathlib import Path

import torch
from dacite import from_dict
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile

from mlstm_kernels.utils.benchmark.benchmarks.huggingface_model_benchmark import (
    create_hf_model_benchmark,
)
from mlstm_kernels.utils.benchmark.param_handling import BenchmarkConfig
from mlstm_kernels.utils.benchmark.run_benchmark import (
    run_and_record_benchmarks,
)
from mlstm_kernels.utils.benchmark.utils import setup_output_folder

# run_hf_model_benchmarks = partial(
#     run_model_benchmarks, benchmark_creator=create_hf_model_benchmark
# )

WARMUP_STEPS = 5


def _benchmark_to_profile(output_folder: Path, profiler=None):
    batch_size = 1
    prefill_length = 0
    weight_dtype = "bfloat16"
    use_torch_compile_model = True
    generation_length = 10

    cfg_yaml = f"""
vary_type: grid
vary_params:
fixed_params:
  generation_length: {generation_length}
  batch_size: {batch_size}
  prefill_length: {prefill_length}

  rep: 5
  warmup: {WARMUP_STEPS}
  benchmark_fn_context_manager: "inference_mode"

x_axis_param: "generation_length"

kernel_specs:
#   - model_name: "mlstm_simple"
#     weight_dtype: {weight_dtype}
#     use_torch_compile_model: {use_torch_compile_model}
#     additional_params:
#       use_torch_compile_generate: False
#       use_cuda_graphs_model: True
#       use_cuda_graphs_generate: False
#       inference_state_dtype: bfloat16
#       embedding_dim: 4096
#       num_heads: 8
#       num_blocks: 32
#       vocab_size: 50304

#       chunkwise_kernel: chunkwise--triton_xl_chunk
#       sequence_kernel: native_sequence__triton_step_fused
#       step_kernel: triton_fused

#       weight_mode: "fused"

#       chunk_size: 128
#       autocast_kernel_dtype: bfloat16

#   - model_name: "xlstm"
#     weight_dtype: {weight_dtype}
#     use_torch_compile_model: {use_torch_compile_model}
#     additional_params:
#       use_cuda_graphs_model: True
#       use_cuda_graphs_generate: False
#       inference_state_dtype: bfloat16
#       embedding_dim: 4096
#       num_heads: 8
#       num_blocks: 32 #3 #32
#       vocab_size: 50304
#       weight_mode: "fused" # or "single"

#       chunkwise_kernel: chunkwise--triton_xl_chunk
#       sequence_kernel: native_sequence__triton_step_fused
#       step_kernel: triton_fused

#       chunk_size: 128
#       autocast_kernel_dtype: bfloat16

#   - model_name: "llama3"
#     weight_dtype: {weight_dtype}
#     use_torch_compile_model: {use_torch_compile_model}
#     additional_params:
#       use_cuda_graphs_generate: True

#   - model_name: "falcon_mamba"
#     weight_dtype: {weight_dtype}
#     use_torch_compile_model: False
#     additional_params:
#       use_cuda_graphs_model: False

  - model_name: "codestral_mamba"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False
    additional_params:
      use_cuda_graphs_model: False

#   - model_name: "zamba2"
#     weight_dtype: {weight_dtype}
#     use_torch_compile_model: {use_torch_compile_model}

benchmark_name: "Look_at_trace"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(
        cfg,
        create_hf_model_benchmark,
        output_folder,
        benchmark_type="model",
        profiler=profiler,
    )


def run_multiple_benchmarks(
    output_dir: str = "./outputs_kernel_benchmarks_profiler",
    output_folder_suffix: str | None = None,
):
    output_folder = setup_output_folder(
        output_dir, name_suffix=output_folder_suffix, log_level=logging.DEBUG
    )
    logging.getLogger("matplotlib").setLevel(
        logging.WARNING
    )  # Suppress matplotlib debug logging.
    trace_folder = output_folder / "tensorboard"
    trace_folder.mkdir(parents=True, exist_ok=False)

    # _sequence_length_benchmark(output_folder, batch_size=1, num_heads=16, head_dim=256)
    # _batch_size_benchmark(output_folder, seq_len=8192, num_heads=16, head_dim=256)
    # _sequence_length_benchmark(output_folder, batch_size=1, num_heads=8, head_dim=512)
    # _batch_size_benchmark(output_folder, seq_len=8192, num_heads=8, head_dim=512)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
    sort_by_keyword = "cuda_time_total"
    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            skip_first=WARMUP_STEPS,  # Do not profile warm-up steps
            wait=1,  # First step for actual runtime without profiler
            warmup=1,  # First step warms up profiler, usually has extra overhead
            active=2,  # Profile 2 steps
            repeat=1,  # Only do once.
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_folder),
    ) as prof:
        _benchmark_to_profile(output_folder, profiler=prof)

    try:
        print(
            prof.key_averages().table(
                sort_by=sort_by_keyword, row_limit=50, max_name_column_width=100
            )
        )
    except AssertionError as e:
        # If no profile data is available, the above will throw an assertion error.
        print(e)
        print("No profiling data available.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_suffix",
        type=str,
        required=False,
        help="Suffix that is appended to the output folder of the benchmark results.",
    )

    args = parser.parse_args()
    print(args)
    run_multiple_benchmarks(output_folder_suffix=args.folder_suffix)
