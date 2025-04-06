#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import contextlib
import logging
import typing
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch._dynamo.cache_size
from vllm import LLM, SamplingParams

from ..param_handling import ModelSpec
from .interface import BenchmarkFnContextManagerCfgType, ModelBenchmarkInterface

LOGGER = logging.getLogger(__name__)

BenchmarkType = Literal["generate", "forward"]


@dataclass
class VLLMModelBenchmark(ModelBenchmarkInterface):
    model_name: str = "mLSTM"

    embedding_dim: int = 4096
    num_heads: int = 8
    num_blocks: int = 32
    vocab_size: int = 50304

    use_torch_compile_model: bool = False  # True
    use_torch_compile_generate: bool = False  # unused for now
    use_cuda_graphs_model: bool = False
    use_cuda_graphs_generate: bool = False
    cuda_graphs_warmups: int = 3

    apply_overrides_to_hf_model: bool = False

    benchmark_type: BenchmarkType = "generate"

    batch_size: int = 1
    prefill_length: int = 128
    generation_length: int = 1

    def get_hf_model_registry(self) -> dict:
        from .huggingface_model_configs import hf_model_registry

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

        return model_config

    def setup_model(self) -> None:
        self.hf_model_config = self.get_hf_model_config(self.model_name)

        # self.model = AutoModelForCausalLM.from_config(self.hf_model_config)

        self.model = LLM(
            self.hf_model_config._name_or_path,
            device=torch.device(self.device),
            hf_overrides={"max_seq_len": 32768},
            dtype=getattr(torch, self.weight_dtype),
        )

        LOGGER.info(f"Setting up model: {self.model}")
        LOGGER.info(f"Model config: {self.hf_model_config}")

        if self.use_torch_compile_generate:
            LOGGER.warning(
                "torch.compile() in Huggingface generate() is not supported. Not compiling generate()."
            )

    def setup_benchmark(self) -> None:
        if self.model is None:
            self.setup_model()

        if self.benchmark_type == "generate":
            self._setup_generate_benchmark()
        elif self.benchmark_type == "forward":
            self._setup_forward_benchmark()
        else:
            raise ValueError(f"Unknown benchmark type: {self.benchmark_type}")

        LOGGER.debug("Setup benchmark done.")

    def _setup_forward_benchmark(self):
        assert (
            self.generation_length == 0
        ), "Generation length must be 0 for forward pass benchmark."
        assert (
            self.prefill_length > 0
        ), "Prefill length must be greater than 0 for forward pass benchmark."

        self.prefill_tokens = torch.randint(
            low=0,
            high=self.hf_model_config.vocab_size,
            size=(self.batch_size, self.prefill_length),
            # device=torch.device(self.device),
            dtype=torch.long,
        )
        # Allow caching compiling of all generation steps.
        if self.use_torch_compile_model:
            LOGGER.info("Free up cache for torch compile.")
            torch.compiler.reset()

        benchmark_fn_context_manager = self._get_benchmark_fn_context_manager()

        LOGGER.info(f"Prefill tokens shape: {self.prefill_tokens.shape}.")

        def benchmark_fn():
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.CUDNN_ATTENTION
            ):
                outputs = self.model.generate(
                    # vllm does not work with tensors somehow
                    prompt_token_ids=self.prefill_tokens.cpu().numpy().tolist(),
                    sampling_params=SamplingParams(
                        min_tokens=1,
                        max_tokens=1,
                        ignore_eos=True,
                        detokenize=False,
                        temperature=0.0,
                    ),
                )
                assert outputs is not None, "Forward pass did not return any output."

        self.benchmark_fn = benchmark_fn

    def _setup_generate_benchmark(self):
        if self.prefill_length == 0:
            self.prefill_length = 1

        # setup prefill inputs
        if self.prefill_length > 0:
            self.prefill_tokens = torch.randint(
                low=0,
                high=self.hf_model_config.vocab_size,
                size=(self.batch_size, self.prefill_length),
            ).to(
                dtype=torch.long
            )  # .to(device=torch.device(self.device), dtype=torch.long)
        else:
            self.prefill_tokens = None

        # Allow caching compiling of all generation steps.
        if self.use_torch_compile_model:
            LOGGER.info("Free up cache for torch compile.")
            torch.compiler.reset()

        pf_shape = (
            self.prefill_tokens.shape if self.prefill_tokens is not None else None
        )
        LOGGER.info(
            f"Prefill tokens shape: {pf_shape}, Generating {self.generation_length} tokens."
        )

        benchmark_fn_context_manager = self._get_benchmark_fn_context_manager()

        def benchmark_fn():
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.CUDNN_ATTENTION
            ):
                with benchmark_fn_context_manager():
                    # Use static cache for Transformer models for compile support.
                    generate_kwargs = {}
                    _ = self.model.generate(
                        prompt_token_ids=self.prefill_tokens.cpu().numpy().tolist(),
                        sampling_params=SamplingParams(
                            min_tokens=self.generation_length,
                            max_tokens=self.generation_length,
                            ignore_eos=True,
                            detokenize=False,
                            temperature=0.0,
                        ),
                        **generate_kwargs,
                    )

        self.benchmark_fn = benchmark_fn

    def available_kernels(self) -> list[str]:
        hf_models = list(self.get_hf_model_registry().keys())
        return hf_models


def create_vllm_model_benchmark(
    model_spec: ModelSpec, param_dict: dict
) -> VLLMModelBenchmark:
    vllm_model_benchmark = VLLMModelBenchmark()

    model_name = model_spec.model_name
    if model_name in vllm_model_benchmark.available_kernels():
        vllm_model_benchmark.model_name = model_spec.model_name
        vllm_model_benchmark.amp_enabled = model_spec.amp_enabled
        vllm_model_benchmark.amp_dtype = model_spec.amp_dtype
        vllm_model_benchmark.weight_dtype = model_spec.weight_dtype
        vllm_model_benchmark.use_torch_compile_model = (
            model_spec.use_torch_compile_model
        )

        vllm_model_benchmark.set_params(param_dict)
        return vllm_model_benchmark

    else:
        raise ValueError(f"Unknown model name: {model_name}")
