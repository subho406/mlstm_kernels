from dataclasses import dataclass
from itertools import product
from typing import Any, Literal


@dataclass
class KernelSpec:
    kernel_name: str
    fwbw: bool
    dtype: Literal["float16", "float32", "float64", "bfloat16"]
    use_torch_compile: bool = False

    additional_params: dict[str, Any] = None

    def to_string(self) -> str:
        str_name = f"{self.kernel_name}__{self.dtype}__{'fwbw' if self.fwbw else 'fw'}"
        if self.additional_params is not None:
            str_name += "_"
            for k, v in self.additional_params.items():
                str_name += f"_{k}-{v}"
        return str_name


@dataclass
class BenchmarkConfig:
    vary_type: Literal["grid", "sequence"]
    vary_params: dict[str, list[Any]]

    fixed_params: dict[str, Any]

    kernel_specs: list[KernelSpec]

    benchmark_name: str

    x_axis_param: str = None

    def _get_vary_param_dicts(self) -> list[dict[str, Any]]:
        vary_dicts = []
        if self.vary_type == "grid":
            for val_tuple in product(*self.vary_params.values()):
                vary_dict = dict(zip(self.vary_params.keys(), val_tuple))
                vary_dicts.append(vary_dict)
        elif self.vary_type == "sequence":
            num_vals = len(list(self.vary_params.values())[0])
            for k, v in self.vary_params.items():
                if len(v) != num_vals:
                    raise ValueError(f"All vary parameters must have the same length. Not matching for {k}")

            for val_tuple in zip(*self.vary_params.values()):
                vary_dict = dict(zip(self.vary_params.keys(), val_tuple))
                vary_dicts.append(vary_dict)
        else:
            raise ValueError(f"Unknown vary type: {self.vary_type}")
        return vary_dicts

    def get_param_dicts(self) -> list[dict[str, Any]]:
        """
        Get all the parameter dictionaries for the benchmark.
        """
        param_dicts = []
        vary_dicts = self._get_vary_param_dicts()
        for vary_dict in vary_dicts:
            param_dict = self.fixed_params.copy()
            for k, v in vary_dict.items():
                param_dict[k] = v
            param_dicts.append(param_dict)
        return param_dicts

    def get_plot_title(self) -> str:
        title = self.benchmark_name
        for k, v in self.fixed_params.items():
            title += f"__{k}-{v}"
        return title
