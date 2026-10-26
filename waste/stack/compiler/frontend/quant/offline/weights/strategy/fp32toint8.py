"""Offline symmetric INT8 quantization of FP32 weight tensors."""

import torch


def quantize_int8(
    weight: torch.Tensor, axis: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return INT8 values and FP32 scales; axis=None selects one tensor scale."""
    if weight.dtype != torch.float32:
        raise ValueError("INT8 weight quantization requires FP32 input")
    if not weight.numel() or not torch.isfinite(weight).all():
        raise ValueError("Weights must be nonempty and finite")
    if axis is not None and not -weight.ndim <= axis < weight.ndim:
        raise ValueError("Quantization axis is outside the weight dimensions")

    weight = weight.detach().cpu()
    if axis is None:
        maximum = weight.abs().amax()
        scale_shape = [1] * weight.ndim
    else:
        axis %= weight.ndim
        maximum = weight.movedim(axis, 0).reshape(weight.shape[axis], -1).abs().amax(1)
        scale_shape = [1] * weight.ndim
        scale_shape[axis] = weight.shape[axis]

    # A zero channel is represented exactly by q=0 and scale=1.
    scales = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
    if (scales == 0).any():
        raise ValueError("Nonzero weight scale cannot be represented in FP32")
    values = torch.round(weight / scales.reshape(scale_shape))
    values = values.clamp(-127, 127).to(torch.int8).contiguous()
    return values, scales.contiguous()
