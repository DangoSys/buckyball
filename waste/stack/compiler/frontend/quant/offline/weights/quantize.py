"""Named offline weight artifacts, independent of backend memory layouts."""

from collections.abc import Mapping

import torch

from stack.compiler.frontend.quant.offline.weights.strategy.fp32toint8 import (
    quantize_int8,
)


def quantize_weights(
    parameters: Mapping[str, torch.Tensor],
    weights: Mapping[str, Mapping],
) -> dict:
    """Return quantized weights, scales, and channel axes without file I/O.

    Each weight specifies source_dtype, target_dtype, method, and axis.
    Unselected parameters and buffers retain their values and dtypes.
    Tensor shapes and element order are unchanged.
    """
    if not weights:
        raise ValueError("No weights selected for quantization")
    unknown = weights.keys() - parameters.keys()
    if unknown:
        raise ValueError(f"Unknown weight names: {sorted(unknown)}")

    tensors, scales, channel_axes = {}, {}, {}
    error_energy = 0.0
    elements = 0
    for name, parameter in parameters.items():
        if name in weights:
            config = weights[name]
            if set(config) != {
                "source_dtype",
                "target_dtype",
                "method",
                "axis",
            }:
                raise ValueError(
                    f"{name}: expected source_dtype, target_dtype, method, and axis"
                )
            scheme = (
                config["source_dtype"],
                config["target_dtype"],
                config["method"],
            )
            if scheme != ("float32", "int8", "symmetric"):
                raise ValueError(f"{name}: unsupported weight quantization {scheme}")
            axis = config["axis"]
            tensors[name], scales[name] = quantize_int8(parameter, axis)
            channel_axes[name] = None if axis is None else axis % parameter.ndim
            scale_shape = [1] * parameter.ndim
            if axis is not None:
                scale_shape[channel_axes[name]] = parameter.shape[axis]
            restored = tensors[name].float() * scales[name].reshape(scale_shape)
            original = parameter.detach().cpu().double()
            error = original - restored.double()
            error_energy += error.square().sum().item()
            elements += parameter.numel()
        else:
            tensors[name] = parameter.detach().cpu().contiguous()

    print(f"Weight quantization MSE: {error_energy / elements:.6e}", flush=True)
    return {
        "format": "buddy.offline.symmetric_int8",
        "version": 1,
        "tensors": tensors,
        "scales": scales,
        "axes": channel_axes,
    }


def dequantize_weights(artifact: dict) -> dict[str, torch.Tensor]:
    """Reconstruct FP32 weights for offline accuracy evaluation."""
    if artifact["format"] != "buddy.offline.symmetric_int8" or artifact["version"] != 1:
        raise ValueError("Unsupported offline weight artifact")
    tensors = dict(artifact["tensors"])
    for name, axis in artifact["axes"].items():
        values = tensors[name]
        shape = [1] * values.ndim
        if axis is not None:
            shape[axis] = values.shape[axis]
        tensors[name] = values.float() * artifact["scales"][name].reshape(shape)
    return tensors
