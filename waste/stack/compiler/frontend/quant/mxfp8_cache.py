import torch


def _shape(values):
    if values.ndim < 1 or values.shape[-1] == 0 or values.shape[-1] % 32:
        raise ValueError("MXFP8 requires a positive last dimension divisible by 32")
    return (*values.shape[:-1], values.shape[-1] // 32)


@torch.library.custom_op("buckyball::mxfp8_encode", mutates_args=())
def encode(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = _shape(values)
    if values.dtype != torch.float32 or not torch.isfinite(values).all():
        raise ValueError("MXFP8 requires finite FP32 values")
    blocks = values.reshape(*shape, 32)
    maximum = blocks.abs().amax(-1)
    exponent = torch.where(maximum == 0, 0, torch.frexp(maximum)[1] - 9)
    exponent = exponent + (torch.ldexp(maximum, -exponent) > 448)
    exponent = exponent.clamp(-127, 119)
    scaled = torch.ldexp(blocks, -exponent[..., None]).contiguous()
    bits = scaled.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign = (bits >> 24) & 128
    element_exponent = ((bits >> 23) & 255) - 120
    fraction = bits & 0x7FFFFF
    high, low = fraction >> 20, fraction & 0xFFFFF
    rounded = high + ((low > 0x80000) | ((low == 0x80000) & ((high & 1) != 0)))
    codes = torch.where(
        element_exponent <= 0,
        (scaled.abs() * 512).round().to(torch.int64),
        element_exponent * 8 + rounded,
    )
    codes = (codes.clamp_max(126) | sign).to(torch.int8).reshape(values.shape)
    return codes, (exponent + 127).to(torch.int8)


@encode.register_fake
def _encode_fake(values):
    shape = _shape(values)
    return values.new_empty(values.shape, dtype=torch.int8), values.new_empty(
        shape, dtype=torch.int8
    )


@torch.library.custom_op("buckyball::mxfp8_decode", mutates_args=())
def decode(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    shape = _shape(codes)
    if codes.dtype != torch.int8 or scales.dtype != torch.int8 or scales.shape != shape:
        raise ValueError("MXFP8 requires matching INT8 codes and scales")
    unsigned = codes.to(torch.int16) & 255
    scale = scales.to(torch.int16) & 255
    if torch.any((unsigned & 127) == 127) or torch.any(scale == 255):
        raise ValueError("MXFP8 NaN encoding")
    exponent, fraction = (unsigned >> 3) & 15, (unsigned & 7).float()
    elements = torch.where(
        exponent == 0, fraction / 512, torch.ldexp(8 + fraction, exponent - 10)
    )
    elements = torch.copysign(elements, torch.where((unsigned & 128) != 0, -1.0, 1.0))
    result = torch.ldexp(elements.reshape(*shape, 32), scale[..., None] - 127).reshape(
        codes.shape
    )
    if not torch.isfinite(result).all():
        raise ValueError("MXFP8 decoded value overflows FP32")
    return result


@decode.register_fake
def _decode_fake(codes, scales):
    if scales.shape != _shape(codes):
        raise ValueError("MXFP8 code/scale shape mismatch")
    return codes.new_empty(codes.shape, dtype=torch.float32)
