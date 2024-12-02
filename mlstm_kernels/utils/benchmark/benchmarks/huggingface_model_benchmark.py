import contextlib
import logging
import typing
from dataclasses import dataclass
from typing import Any

import torch

from ..param_handling import ModelSpec
from .interface import BenchmarkFnContextManagerCfgType, ModelBenchmarkInterface

LOGGER = logging.getLogger(__name__)

ministral8b_config = {
    "architectures": ["MistralForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "max_position_embeddings": 32768,
    "model_type": "mistral",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_theta": 100000000.0,
    "sliding_window": 32768,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.46.0.dev0",
    "use_cache": True,
    "vocab_size": 131072,
}

codestral_mamba_config = {
    "_name_or_path": "/raid/pablo/codestral-hf-good/",
    "architectures": ["Mamba2ForCausalLM"],
    "bos_token_id": 0,
    "chunk_size": 256,
    "conv_kernel": 4,
    "eos_token_id": 0,
    "expand": 2,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.1,
    "intermediate_size": 8192,
    "layer_norm_epsilon": 1e-05,
    "model_type": "mamba2",
    "n_groups": 8,
    "norm_before_gate": True,
    "num_heads": 128,
    "num_hidden_layers": 64,
    "pad_token_id": 0,
    "rescale_prenorm_residual": False,
    "residual_in_fp32": True,
    "rms_norm": True,
    "state_size": 128,
    "tie_word_embeddings": False,
    "time_step_floor": 0.0001,
    "time_step_init_scheme": "random",
    "time_step_limit": (0.0, float("inf")),
    "time_step_max": 0.1,
    "time_step_min": 0.001,
    "time_step_rank": 256,
    "time_step_scale": 1.0,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.44.0.dev0",
    "use_bias": False,
    "use_cache": True,
    "use_conv_bias": True,
    "vocab_size": 32768,
}

llama_2_config = {
    "_name_or_path": "meta-llama/Llama-2-7b-hf",
    "architectures": ["LlamaForCausalLM"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "transformers_version": "4.31.0.dev0",
    "use_cache": True,
    "vocab_size": 32000,
}

llama_3_1_config = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.43.0.dev0",
    "use_cache": True,
    "vocab_size": 128256,
}

falcon_mamba_config = {
    "_name_or_path": "./",
    "architectures": ["FalconMambaForCausalLM"],
    "bos_token_id": 0,
    "conv_kernel": 4,
    "eos_token_id": 11,
    "expand": 16,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.1,
    "intermediate_size": 8192,
    "layer_norm_epsilon": 1e-05,
    "model_type": "falcon_mamba",
    "num_hidden_layers": 64,
    "pad_token_id": 11,
    "rescale_prenorm_residual": False,
    "residual_in_fp32": True,
    "state_size": 16,
    "tie_word_embeddings": False,
    "time_step_floor": 0.0001,
    "time_step_init_scheme": "random",
    "time_step_max": 0.1,
    "time_step_min": 0.001,
    "time_step_rank": 256,
    "time_step_scale": 1.0,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.43.0.dev0",
    "use_bias": False,
    "use_cache": True,
    "use_conv_bias": True,
    "vocab_size": 65024,
}

zamba_config = {
    "add_bias_linear": False,
    "architectures": ["Zamba2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "conv_dimension": 4,
    "eos_token_id": 2,
    "expansion_factor": 2,
    "ffn_hidden_size": 14336,
    "ft_lora": False,
    "gated_linear_unit": True,
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "kv_channels": 112,
    "layers_block_type": [
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
    ],
    "lora_rank": 128,
    "mamba_headdim": 64,
    "mamba_ngroups": 2,
    "max_position_embeddings": 4096,
    "model_type": "zamba2",
    "num_attention_heads": 32,
    "num_hidden_layers": 81,
    "num_key_value_heads": 32,
    "num_logits_to_keep": 1,
    "num_mem_blocks": 2,
    "num_query_groups": 32,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000,
    "sliding_window": None,
    "state_size": 64,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.43.0.dev0",
    "use_cache": True,
    "use_mamba_kernels": True,
    "use_mem_rope": True,
    "use_shared_attention_lora": False,
    "use_shared_block_lora": True,
    "vocab_size": 32000,
}


@dataclass
class HFModelBenchmark(ModelBenchmarkInterface):
    model_name: str = "mLSTM"

    embedding_dim: int = 4096
    num_heads: int = 8
    num_blocks: int = 32
    vocab_size: int = 50304
    # feedforward
    ffn_proj_factor: float = 2.6667
    ffn_round_up_to_multiple_of: int = 64
    # mlstm
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    # mlstm backend
    chunkwise_kernel: str = "chunkwise--triton_xl_chunk"
    sequence_kernel: str = "native_sequence__triton_step_fused"
    step_kernel: str = "triton_fused"
    mode: str = "inference"
    chunk_size: int = 128

    inference_state_dtype: str = "float32"
    autocast_kernel_dtype: str = "bfloat16"

    # benchmark
    amp_enabled: bool = True
    amp_dtype: str = "bfloat16"
    weight_dtype: str = "float32"

    use_torch_compile_model: bool = True
    use_torch_compile_generate: bool = False  # unused for now

    apply_overrides_to_hf_model: bool = False

    batch_size: int = 1
    prefill_length: int = 128
    generation_length: int = 1

    def get_hf_model_registry(self) -> dict:
        from transformers.models.falcon_mamba import FalconMambaConfig
        from transformers.models.llama import LlamaConfig
        from transformers.models.mamba2 import Mamba2Config
        from transformers.models.mistral import MistralConfig
        from transformers.models.xlstm import xLSTMConfig
        from transformers.models.zamba import ZambaConfig

        hf_model_registry = {
            "llama2": (LlamaConfig, llama_2_config),
            "llama3": (LlamaConfig, llama_3_1_config),
            "ministral8b": (MistralConfig, ministral8b_config),
            "codestral_mamba": (Mamba2Config, codestral_mamba_config),
            "falcon_mamba": (FalconMambaConfig, falcon_mamba_config),
            "zamba2": (ZambaConfig, zamba_config),
            "xlstm": (xLSTMConfig, {}),
        }

        return hf_model_registry

    def get_hf_model_config(self, model_name: str) -> Any:
        hf_model_registry = self.get_hf_model_registry()

        if model_name not in hf_model_registry:
            raise ValueError(
                f"Unknown model name: {model_name}, available models: {hf_model_registry.keys()}"
            )

        model_config_class, model_config_dict = hf_model_registry[model_name]

        model_config = model_config_class(**model_config_dict)

        if self.apply_overrides_to_hf_model:
            if model_name == "llama3" or model_name == "llama2":
                model_config.vocab_size = self.vocab_size
                model_config.hidden_size = self.embedding_dim
                model_config.num_attention_heads = self.num_heads
                model_config.num_hidden_layers = self.num_blocks
            elif model_name == "falcon_mamba":
                model_config.vocab_size = self.vocab_size
                model_config.hidden_size = self.embedding_dim
                model_config.num_hidden_layers = self.num_blocks
            elif model_name == "codestral_mamba":
                model_config.vocab_size = self.vocab_size
                model_config.hidden_size = self.embedding_dim
                model_config.num_hidden_layers = self.num_blocks
            elif model_name == "ministral8b":
                model_config.vocab_size = self.vocab_size
                model_config.hidden_size = self.embedding_dim
                model_config.num_attention_heads = self.num_heads
                model_config.num_hidden_layers = self.num_blocks
            elif model_name == "zamba2":
                model_config.vocab_size = self.vocab_size
                model_config.hidden_size = self.embedding_dim
                # we cannot override this because there is a fixed block map in the model
                # we would have override these too
                # model_config.num_hidden_layers = self.num_blocks
            else:
                raise NotImplementedError(
                    f"Overrides for model {model_name} are not implemented."
                )

        if model_name == "xlstm":
            model_config.vocab_size = self.vocab_size
            model_config.embedding_dim = self.embedding_dim
            model_config.num_heads = self.num_heads
            model_config.num_blocks = self.num_blocks
            model_config.ffn_proj_factor = self.ffn_proj_factor
            model_config.ffn_round_up_to_multiple_of = self.ffn_round_up_to_multiple_of
            model_config.qk_dim_factor = self.qk_dim_factor
            model_config.v_dim_factor = self.v_dim_factor
            model_config.chunkwise_kernel = self.chunkwise_kernel
            model_config.sequence_kernel = self.sequence_kernel
            model_config.step_kernel = self.step_kernel
            model_config.chunk_size = self.chunk_size
            model_config.mode = self.mode
            model_config.inference_state_dtype = self.inference_state_dtype
            model_config.autocast_kernel_dtype = self.autocast_kernel_dtype

        return model_config

    def get_hf_model_torch_compile_config(self, model_name: str) -> dict:
        hf_model_torch_compile_config = {
            "llama2": dict(disable=True),
            "llama3": dict(dynamic=False, fullgraph=True, mode="reduce-overhead"),
            "ministral8b": dict(dynamic=False, fullgraph=True, mode="reduce-overhead"),
            "codestral_mamba": dict(
                dynamic=False, fullgraph=False, mode="reduce-overhead"
            ),
            "xlstm": dict(dynamic=False, fullgraph=True, mode="reduce-overhead"),
            "falcon_mamba": dict(disable=True),
            "zamba2": dict(
                dynamic=False, fullgraph=False, mode="reduce-overhead"
            ),  # same as mamba2
        }
        return hf_model_torch_compile_config[model_name]

    def setup_model(self) -> None:
        from transformers import AutoModelForCausalLM

        self.hf_model_config = self.get_hf_model_config(self.model_name)

        self.model = AutoModelForCausalLM.from_config(self.hf_model_config)

        self.model = self.model.to(
            dtype=getattr(torch, self.weight_dtype), device=torch.device(self.device)
        )

        LOGGER.info(f"Setting up model: {self.model}")
        LOGGER.info(f"Model config: {self.hf_model_config}")

        def count_num_params(model):
            return sum(p.numel() for p in model.parameters())

        LOGGER.info(f"Model number of parameters: {count_num_params(self.model)}")

        self.model.generation_config.cache_implementation = "static"

        if self.use_torch_compile_model:
            self.model.forward = torch.compile(
                self.model.forward, dynamic=False, fullgraph=False, mode="default"
            )

    def setup_benchmark(self) -> None:
        if self.model is None:
            self.setup_model()

        # we need to input a first token to pass the batch size to huggingface
        # generate()
        if self.prefill_length == 0:
            self.prefill_length = 1

        # setup prefill inputs
        if self.prefill_length > 0:
            self.prefill_tokens = torch.randint(
                low=0,
                high=self.hf_model_config.vocab_size,
                size=(self.batch_size, self.prefill_length),
            ).to(device=torch.device(self.device), dtype=torch.long)
        else:
            self.prefill_tokens = None

        pf_shape = (
            self.prefill_tokens.shape if self.prefill_tokens is not None else None
        )
        LOGGER.info(
            f"Prefill tokens shape: {pf_shape}, Generating {self.generation_length} tokens."
        )

        assert self.benchmark_fn_context_manager in typing.get_args(
            BenchmarkFnContextManagerCfgType
        ), f"Invalid benchmark_fn_context_manager: {self.benchmark_fn_context_manager}"

        if self.benchmark_fn_context_manager == "none":
            benchmark_fn_context_manager = contextlib.nullcontext
        elif self.benchmark_fn_context_manager == "no_grad":
            benchmark_fn_context_manager = torch.no_grad
        elif self.benchmark_fn_context_manager == "inference_mode":
            benchmark_fn_context_manager = torch.inference_mode

        def benchmark_fn():
            with benchmark_fn_context_manager():
                outputs = self.model.generate(
                    inputs=self.prefill_tokens,
                    max_new_tokens=self.generation_length,
                    min_new_tokens=self.generation_length,
                    do_sample=False,
                    use_cache=True,
                )
            assert (
                outputs.shape
                == (self.batch_size, self.prefill_length + self.generation_length)
            ), f"Unexpected output shape: {outputs.shape}, expected: {(self.batch_size, self.prefill_length + self.generation_length)}"

        self.benchmark_fn = benchmark_fn

    def available_kernels(self) -> list[str]:
        hf_models = list(self.get_hf_model_registry().keys())
        return hf_models


def create_hf_model_benchmark(
    model_spec: ModelSpec, param_dict: dict
) -> HFModelBenchmark:
    from .model_benchmarks import mLSTMSimpleModelBenchmark

    hf_model_benchmark = HFModelBenchmark()
    mlstm_simple_benchmark = mLSTMSimpleModelBenchmark()

    model_name = model_spec.model_name
    if model_name in hf_model_benchmark.available_kernels():
        hf_model_benchmark.model_name = model_spec.model_name
        hf_model_benchmark.amp_enabled = model_spec.amp_enabled
        hf_model_benchmark.amp_dtype = model_spec.amp_dtype
        hf_model_benchmark.weight_dtype = model_spec.weight_dtype
        hf_model_benchmark.use_torch_compile_model = model_spec.use_torch_compile_model

        hf_model_benchmark.set_params(param_dict)
        return hf_model_benchmark

    elif model_name in mlstm_simple_benchmark.available_kernels():
        mlstm_simple_benchmark.model_name = model_spec.model_name
        mlstm_simple_benchmark.amp_enabled = False  # model_spec.amp_enabled
        mlstm_simple_benchmark.amp_dtype = model_spec.amp_dtype
        mlstm_simple_benchmark.weight_dtype = model_spec.weight_dtype
        mlstm_simple_benchmark.use_torch_compile_model = (
            model_spec.use_torch_compile_model
        )

        mlstm_simple_benchmark.set_params(param_dict)

        return mlstm_simple_benchmark
