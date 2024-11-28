import logging
from dataclasses import dataclass
from typing import Any

import torch

from ..param_handling import ModelSpec
from .interface import BenchmarkInterface

LOGGER = logging.getLogger(__name__)

llama_3_1_cfg = {
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


@dataclass
class HFModelBenchmark(BenchmarkInterface):
    model_name: str = "mLSTM"

    embedding_dim: int = 256
    num_heads: int = 4
    num_blocks: int = 2
    vocab_size: int = 1000
    # feedforward
    ffn_proj_factor: float = 2.6667
    ffn_round_up_to_multiple_of: int = 64
    # mlstm
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    # mlstm backend
    chunkwise_kernel: str = "chunkwise--triton_xl_chunk"
    sequence_kernel: str = "native_sequence__native"
    step_kernel: str = "native"
    mode: str = "inference"
    chunk_size: int = 128

    inference_state_dtype: str = "float32"
    autocast_kernel_dtype: str = "bfloat16"

    # benchmark
    amp_enabled: bool = True
    amp_dtype: str = "bfloat16"
    weight_dtype: str = "float32"

    use_torch_compile_model: bool = False
    use_torch_compile_generate: bool = False

    apply_overrides_to_hf_model: bool = False

    batch_size: int = 1
    prefill_length: int = 128
    generation_length: int = 1

    def get_hf_model_registry(self) -> dict:
        from transformers.models.falcon_mamba import FalconMambaConfig
        from transformers.models.llama import LlamaConfig
        from transformers.models.xlstm import xLSTMConfig

        hf_model_registry = {
            "llama": (LlamaConfig, llama_3_1_cfg),
            "falcon_mamba": (FalconMambaConfig, falcon_mamba_config),
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
            if model_name == "llama":
                model_config.vocab_size = self.vocab_size
                model_config.hidden_size = self.embedding_dim
                model_config.num_attention_heads = self.num_heads
                model_config.num_hidden_layers = self.num_blocks
            elif model_name == "falcon_mamba":
                model_config.vocab_size = self.vocab_size
                model_config.hidden_size = self.embedding_dim
                model_config.num_hidden_layers = self.num_blocks

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

    def setup_benchmark(self) -> None:
        from transformers import AutoModelForCausalLM
        from transformers.tokenization_utils_base import BatchEncoding

        hf_model_config = self.get_hf_model_config(self.model_name)

        self.model = AutoModelForCausalLM.from_config(hf_model_config)

        self.model = self.model.to(
            dtype=getattr(torch, self.weight_dtype), device=torch.device(self.device)
        )

        LOGGER.info(f"Setting up model: {self.model}")

        # setup prefill inputs
        if self.prefill_length > 0:
            self.prefill_tokens = torch.randint(
                low=0, high=self.vocab_size, size=(self.batch_size, self.prefill_length)
            ).to(device=torch.device(self.device))
        else:
            self.prefill_tokens = None

        if self.use_torch_compile_model:
            self.model = torch.compile(self.model)

        def benchmark_fn():
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs=self.prefill_tokens,
                    max_new_tokens=self.generation_length,
                    min_new_tokens=self.generation_length,
                    do_sample=False,
                    use_cache=True,
                )

        self.benchmark_fn = benchmark_fn


def create_hf_model_benchmark(
    model_spec: ModelSpec, param_dict: dict
) -> HFModelBenchmark:
    hf_model_benchmark = HFModelBenchmark()

    hf_model_benchmark.model_name = model_spec.model_name
    hf_model_benchmark.amp_enabled = model_spec.amp_enabled
    hf_model_benchmark.amp_dtype = model_spec.amp_dtype
    hf_model_benchmark.weight_dtype = model_spec.weight_dtype
    hf_model_benchmark.use_torch_compile_model = model_spec.use_torch_compile_model

    hf_model_benchmark.set_params(param_dict)

    return hf_model_benchmark
