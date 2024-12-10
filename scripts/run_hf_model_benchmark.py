#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import argparse
import logging
import pprint
import typing
from pathlib import Path
from typing import Literal

from dacite import from_dict
from omegaconf import OmegaConf

from mlstm_kernels.utils.benchmark.benchmarks.huggingface_model_benchmark import (
    create_hf_model_benchmark,
)
from mlstm_kernels.utils.benchmark.param_handling import BenchmarkConfig
from mlstm_kernels.utils.benchmark.run_benchmark import run_and_record_benchmarks
from mlstm_kernels.utils.benchmark.utils import setup_output_folder

LOGGER = logging.getLogger(__name__)


def _throughput_benchmark(
    output_folder: Path,
    prefill_length: int = 0,
    generation_length: int = 100,
    use_torch_compile_model: bool = True,
    weight_dtype: str = "bfloat16",
):
    cfg_yaml = f"""
vary_type: grid
vary_params:
  batch_size: [1, 4, 8, 16, 32, 64]
fixed_params:
  prefill_length: {prefill_length}
  generation_length: {generation_length}

  rep: 4
  warmup: 2

  benchmark_fn_context_manager: "inference_mode" #"no_grad" #"inference_mode"
  benchmark_type: "forward"

x_axis_param: "batch_size"

kernel_specs:
  # Note: not implemented for mlstm_simple
  # - model_name: "mlstm_simple"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: True
  #   additional_params:
  #     use_cuda_graphs_generate: True
  #     use_cuda_graphs_model: False

  #     use_torch_compile_generate: False
  #     inference_state_dtype: bfloat16
  #     embedding_dim: 4096
  #     num_heads: 8
  #     num_blocks: 32 #3 #32
  #     vocab_size: 50304
  #     weight_mode: "fused"

  #     chunkwise_kernel: chunkwise--triton_xl_chunk
  #     sequence_kernel: native_sequence__triton_step_fused
  #     step_kernel: triton_fused

  #     chunk_size: 128
  #     autocast_kernel_dtype: bfloat16

  - model_name: "xlstm"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: True
    additional_params:
      use_cuda_graphs_generate: True
      use_cuda_graphs_model: False

      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304
      weight_mode: "fused"

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  # Note: we can use torch.compile for llama models since we use a static cache
  - model_name: "llama2"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: True
    additional_params:
      use_cuda_graphs_generate: False
      use_cuda_graphs_model: False

  - model_name: "llama3"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: True
    additional_params:
      use_cuda_graphs_generate: False
      use_cuda_graphs_model: False

  # Note: torch.compile is not well supported for codestral and falcon models
  # Runtimes do not differ and are sometimes slower
  - model_name: "codestral_mamba"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False
    additional_params:
      use_cuda_graphs_generate: True
      use_cuda_graphs_model: False

  - model_name: "falcon_mamba"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False
    additional_params:
      use_cuda_graphs_generate: True
      use_cuda_graphs_model: False


benchmark_name: "hf_7B_throughput__pfl{prefill_length}_gl{generation_length}_tc{use_torch_compile_model}_weightdtype{weight_dtype}"
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
        setup_model_on_every_param_combination=False,
    )


def _generation_time_benchmark(
    output_folder: Path,
    prefill_length: int = 0,
    batch_size: int = 1,
    use_torch_compile_model: bool = True,
    weight_dtype: str = "bfloat16",
):
    cfg_yaml = f"""
vary_type: grid
vary_params:
  generation_length: [64, 128, 512, 1024, 2048, 4096, 8192, 16384]
fixed_params:
  batch_size: {batch_size}
  prefill_length: {prefill_length}

  rep: 1
  warmup: 1
  benchmark_fn_context_manager: "inference_mode"

x_axis_param: "generation_length"

kernel_specs:
  - model_name: "mlstm_simple"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: True
    additional_params:
      use_cuda_graphs_generate: False
      use_cuda_graphs_model: True

      use_torch_compile_generate: False
      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304
      weight_mode: "fused"

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "xlstm"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: True
    additional_params:
      use_cuda_graphs_generate: False
      use_cuda_graphs_model: True

      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304
      weight_mode: "fused"

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  # Note: we can use torch.compile for llama models since we use a static cache
  # Cuda graphs on model are not supported for llama models, yet.
  # When applying cuda_graph on generate not much difference was observed.
  # - model_name: "llama2"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: True
  #   additional_params:
  #     use_cuda_graphs_generate: False
  #     use_cuda_graphs_model: False

  # - model_name: "llama3"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: True
  #   additional_params:
  #     use_cuda_graphs_generate: False
  #     use_cuda_graphs_model: False

  # # Note: torch.compile is not well supported for codestral and falcon models
  # # Runtimes do not differ and are sometimes slower
  # - model_name: "codestral_mamba"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False
  #   additional_params:
  #     use_cuda_graphs_generate: False
  #     use_cuda_graphs_model: True

  # - model_name: "falcon_mamba"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False
  #   additional_params:
  #     use_cuda_graphs_generate: False
  #     use_cuda_graphs_model: True

benchmark_name: "hf_7B_generation_time__pfl{prefill_length}_bs{batch_size}_tc{use_torch_compile_model}_weightdtype{weight_dtype}"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )
    LOGGER.info(f"Running benchmark with config:\n{pprint.pformat(cfg)}")
    run_and_record_benchmarks(
        cfg,
        create_hf_model_benchmark,
        output_folder,
        benchmark_type="model",
        setup_model_on_every_param_combination=False,
    )


def _time_to_first_token_benchmark(
    output_folder: Path,
    batch_size: int = 8,
    generation_length: int = 1,
    use_torch_compile_model: bool = True,
    weight_dtype: str = "bfloat16",
):
    # We add all models with and without cuda graphs on the generate function.
    cfg_yaml = f"""
vary_type: grid
vary_params:
  prefill_length: [128, 512, 1024, 2048, 4096, 8192, 16384]
fixed_params:
  batch_size: {batch_size}
  generation_length: {generation_length}

  rep: 4
  warmup: 2 #1
  benchmark_fn_context_manager: "inference_mode"

x_axis_param: "prefill_length"

kernel_specs:
  - model_name: "xlstm"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}
    additional_params:
      use_cuda_graphs_generate: True
      use_cuda_graphs_model: False

      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304
      weight_mode: "fused"

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "mlstm_simple"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}
    additional_params:
      use_torch_compile_generate: False

      use_cuda_graphs_generate: True
      use_cuda_graphs_model: False

      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304
      weight_mode: "fused"

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "xlstm"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}
    additional_params:
      use_cuda_graphs_generate: False
      use_cuda_graphs_model: False

      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304
      weight_mode: "fused"

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "mlstm_simple"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: True
    additional_params:
      use_torch_compile_generate: False

      use_cuda_graphs_generate: False
      use_cuda_graphs_model: False

      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304
      weight_mode: "fused"

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "llama2"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: True
    additional_params:
      use_cuda_graphs_generate: True
      use_cuda_graphs_model: False

  - model_name: "llama3"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: True
    additional_params:
      use_cuda_graphs_generate: True
      use_cuda_graphs_model: False

  - model_name: "llama2"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: True
    additional_params:
      use_cuda_graphs_generate: False
      use_cuda_graphs_model: False

  - model_name: "llama3"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: True
    additional_params:
      use_cuda_graphs_generate: False
      use_cuda_graphs_model: False

  # Note: this should work, but not tested extensively for torch.compile and cuda graphs
  # - model_name: "ministral8b"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False
  #   additional_params:
  #     use_cuda_graphs_generate: False
  #     use_cuda_graphs_model: False

  # Note: torch.compile does not work for mamba models
  # We can use cuda graph on generate since we generate not more than 101 tokens
  - model_name: "codestral_mamba"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False
    additional_params:
      use_cuda_graphs_generate: True
      use_cuda_graphs_model: False

  - model_name: "codestral_mamba"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False
    additional_params:
      use_cuda_graphs_generate: False
      use_cuda_graphs_model: False

  - model_name: "falcon_mamba"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False
    additional_params:
      use_cuda_graphs_generate: True
      use_cuda_graphs_model: False

  - model_name: "falcon_mamba"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False
    additional_params:
      use_cuda_graphs_generate: False
      use_cuda_graphs_model: False

  - model_name: "zamba2"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False
    additional_params:
      use_cuda_graphs_generate: False
      use_cuda_graphs_model: False

benchmark_name: "hf_7B_timtofirsttok__bs{batch_size}_gl{generation_length}_tc{use_torch_compile_model}_weightdtype{weight_dtype}"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )
    LOGGER.info(f"Running benchmark with config:\n{pprint.pformat(cfg)}")
    run_and_record_benchmarks(
        cfg,
        create_hf_model_benchmark,
        output_folder,
        benchmark_type="model",
        setup_model_on_every_param_combination=False,
    )


BenchmarkType = Literal["ttft", "gen_time", "throughput"]


def run_multiple_benchmarks(
    output_dir: str = "./outputs_kernel_benchmarks",
    benchmark_type: BenchmarkType = "ttft",
    use_torch_compile: bool = True,
    output_folder_suffix: str | None = None,
):
    full_folder_suffix = (
        f"{benchmark_type}__{output_folder_suffix}"
        if output_folder_suffix
        else benchmark_type
    )
    output_folder = setup_output_folder(output_dir, name_suffix=full_folder_suffix)

    if benchmark_type == "throughput":
        for prefill_length in [2048, 4096, 8192]:
            _throughput_benchmark(
                output_folder,
                prefill_length=prefill_length,
                generation_length=0,
                use_torch_compile_model=use_torch_compile,
                weight_dtype="bfloat16",
            )
    elif benchmark_type == "ttft":
        batch_sizes = [1, 8]
        generation_lengths = [1, 101]
        for batch_size in batch_sizes:
            for generation_length in generation_lengths:
                _time_to_first_token_benchmark(
                    output_folder,
                    batch_size=batch_size,
                    generation_length=generation_length,
                    use_torch_compile_model=use_torch_compile,
                    weight_dtype="bfloat16",
                )
    elif benchmark_type == "gen_time":
        batch_sizes = [1]
        prefill_lengths = [0]
        for batch_size in batch_sizes:
            for prefill_length in prefill_lengths:
                _generation_time_benchmark(
                    output_folder,
                    prefill_length=prefill_length,
                    batch_size=batch_size,
                    use_torch_compile_model=use_torch_compile,
                    weight_dtype="bfloat16",
                )
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help=f"The benchmark type. One of {typing.get_args(BenchmarkType)}",
    )
    parser.add_argument(
        "--folder_suffix",
        type=str,
        required=False,
        help="Suffix that is appended to the output folder of the benchmark results.",
    )
    parser.add_argument(
        "--use_torch_compile",
        type=str,
        required=False,
        default="1",
        help="Whether to use torch compile for the benchmark.",
    )

    args = parser.parse_args()
    print(args)

    run_multiple_benchmarks(
        output_folder_suffix=args.folder_suffix,
        benchmark_type=args.benchmark,
        use_torch_compile=bool(args.use_torch_compile == "1"),
    )

# Run commands:
# PYTHONPATH=. python scripts/run_hf_model_benchmark.py --benchmark ttft --folder_suffix version_0
# PYTHONPATH=. python scripts/run_hf_model_benchmark.py --benchmark gen_time --folder_suffix version_0
