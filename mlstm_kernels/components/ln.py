# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(
        self,
        ndim: int = -1,
        weight: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        residual_weight: bool = True,
        force_float32_reductions: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.force_float32_reductions = force_float32_reductions
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = input.shape
        in_dtype = input.dtype

        input = input.to(dtype=self.weight.dtype)

        gn_in_1 = input.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )
        out = out.to(dtype=in_dtype)
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out


class RMSNorm(LayerNorm):
    """RMSNorm with optional bias according to https://arxiv.org/abs/1910.07467.
    Inspired by https://github.com/mistralai/mistral-src/blob/main/mistral/model.py.
    """

    def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ..., S,..., D)
        # apply rms norm over the last dimension, i.e. H dimension
        in_dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(in_dtype)

    def _apply_weights_and_biases(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_proxy is not None and self.bias is not None:
            return x * self.weight_proxy + self.bias
        elif self.weight_proxy is not None and self.bias is None:
            return x * self.weight_proxy
        elif self.weight_proxy is None and self.bias is None:
            return x
        else:
            raise ValueError("RMSNorm: combination use_weight=False and use_bias=True not possible.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        x = self._rms_normalize(input)
        return self._apply_weights_and_biases(x)

    def extra_repr(self):
        s = f"weight.shape={self.weight.shape}"
        if self.bias is not None:
            s += f" bias.shape={self.bias.shape}"
        return s


class MultiHeadRMSNorm(RMSNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # TODO change input shape to be (B, S, NH, DH) as input already, keep it now for bw compat
        # x: (B, NH, S, DH)
        B, NH, S, DH = input.shape
        x = self._rms_normalize(input)
        x = x.transpose(1, 2).reshape(B, S, NH * DH)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, D)
        out = self._apply_weights_and_biases(x)
        out = out.reshape(B, S, NH, DH).transpose(1, 2)
        return out
