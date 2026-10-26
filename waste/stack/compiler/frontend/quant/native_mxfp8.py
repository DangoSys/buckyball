"""MXFP8 storage and offline activation calibration for native CPU/GPU models."""

import torch
from torch import nn

from stack.compiler.frontend.quant import mxfp8_cache


@torch.library.custom_op("buddy_native::mxfp8_encode_fixed", mutates_args=())
def encode_fixed(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    exponent = (scales.to(torch.int16) & 255) - 127
    blocks = values.reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    scaled = torch.ldexp(blocks, -exponent[..., None]).contiguous()
    magnitude = scaled.abs().clamp_max(448)
    bits = magnitude.view(torch.int32)
    element_exponent = (bits >> 23) - 120
    fraction = bits & 0x7FFFFF
    high, low = fraction >> 20, fraction & 0xFFFFF
    rounded = high + ((low > 0x80000) | ((low == 0x80000) & ((high & 1) != 0)))
    codes = torch.where(
        element_exponent <= 0,
        (magnitude * 512).round().to(torch.int32),
        element_exponent * 8 + rounded,
    )
    sign = (scaled.view(torch.int32) >> 24) & 128
    return (codes | sign).to(torch.int8).reshape(values.shape)


@encode_fixed.register_fake
def _encode_fake(values, scales):
    return torch.empty_like(values, dtype=torch.int8)


@torch.library.custom_op("buddy_native::mxfp8_decode", mutates_args=())
def decode(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return mxfp8_cache.decode(
        codes,
        scales.expand(*codes.shape[:-1], codes.shape[-1] // 32).contiguous(),
    )


@decode.register_fake
def _decode_fake(codes, scales):
    return torch.empty_like(codes, dtype=torch.float32)


@torch.library.custom_op("buddy_native::mxfp8_linear", mutates_args=())
def matmul(
    activation: torch.Tensor,
    activation_scales: torch.Tensor,
    weights: torch.Tensor,
    weight_scales: torch.Tensor,
) -> torch.Tensor:
    # Round the dot product once, before the next MXFP8 quantization boundary.
    return nn.functional.linear(
        decode(activation, activation_scales).double(),
        decode(weights, weight_scales).double(),
    ).float()


@matmul.register_fake
def _matmul_fake(activation, activation_scales, weights, weight_scales):
    return activation.new_empty(
        (activation.shape[0], weights.shape[0]), dtype=torch.float32
    )


@torch.library.custom_op("buddy_native::attention_matmul", mutates_args=())
def attention_matmul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.bmm(lhs.double(), rhs.double()).float()


@attention_matmul.register_fake
def _attention_matmul_fake(lhs, rhs):
    return lhs.new_empty((lhs.shape[0], lhs.shape[1], rhs.shape[2]))


@torch.library.custom_op("buddy_native::attention_softmax", mutates_args=())
def attention_softmax(value: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.softmax(value.double(), dim=axis).float()


@attention_softmax.register_fake
def _attention_softmax_fake(value, axis):
    return torch.empty_like(value)


def attention(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    softcap=None,
    **kwargs,
):
    if dropout != 0.0:
        raise ValueError("MXFP8 attention is an inference kernel")
    batch, heads, length, width = query.shape
    keys = key.shape[2]
    groups = module.num_key_value_groups
    key = (
        key[:, :, None]
        .expand(batch, key.shape[1], groups, keys, width)
        .reshape(batch * heads, keys, width)
    )
    value = (
        value[:, :, None]
        .expand(batch, value.shape[1], groups, keys, width)
        .reshape(batch * heads, keys, width)
    )
    query = query.reshape(batch * heads, length, width)
    scores = (
        attention_matmul(query, key.transpose(1, 2)).reshape(batch, heads, length, keys)
        * scaling
    )
    if softcap is not None:
        scores = torch.tanh(scores / softcap) * softcap
    if attention_mask is not None:
        scores = scores + attention_mask
    probabilities = attention_softmax(scores, -1)
    output = attention_matmul(probabilities.reshape(batch * heads, length, keys), value)
    return (
        output.reshape(batch, heads, length, width).transpose(1, 2).contiguous(),
        probabilities,
    )


def register(compiler):
    from buddy.compiler.graph.operation import Op, OpType
    from stack.compiler.frontend.ops import reductions
    from stack.compiler.frontend.ops.native_mxfp8 import codec, concat
    from stack.compiler.frontend.ops.native_mxfp8 import matmul as lower_matmul
    from stack.compiler.frontend.ops.reductions import bmm, gelu, mean, softmax, tanh

    class NativeMXFP8EncodeOp(Op):
        def __init__(self):
            super().__init__()
            self._op_type = OpType.BroadcastType

    class NativeMXFP8DecodeOp(Op):
        def __init__(self):
            super().__init__()
            self._op_type = OpType.BroadcastType

    class NativeMXFP8MatmulOp(Op):
        def __init__(self):
            super().__init__()
            self._op_type = OpType.ReduceType

    class NativeAttentionMatmulOp(Op):
        pass

    class NativeAttentionSoftmaxOp(Op):
        pass

    compiler._ops_map.update(
        {
            "buddy_native.mxfp8_encode_fixed.default": NativeMXFP8EncodeOp,
            "buddy_native.mxfp8_decode.default": NativeMXFP8DecodeOp,
            "buddy_native.mxfp8_linear.default": NativeMXFP8MatmulOp,
            "buddy_native.attention_matmul.default": NativeAttentionMatmulOp,
            "buddy_native.attention_softmax.default": NativeAttentionSoftmaxOp,
        }
    )
    compiler._ops_registry.update(
        {
            "NativeMXFP8EncodeOp": codec,
            "NativeMXFP8DecodeOp": codec,
            "CatOp": concat,
            "NativeAttentionMatmulOp": reductions.bmm,
            "NativeAttentionSoftmaxOp": reductions.softmax,
            "MeanOp": mean,
            "BatchMatmulOp": bmm,
            "SoftmaxOp": softmax,
            "GeluOp": gelu,
            "TanhOp": tanh,
            "NativeMXFP8MatmulOp": lower_matmul,
        }
    )


def scale_for(maximum):
    exponent = torch.where(maximum == 0, 0, torch.frexp(maximum)[1] - 9)
    exponent += torch.ldexp(maximum, -exponent) > 448
    return (exponent.clamp(-127, 119) + 127).to(torch.int8)


class Linear(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.in_features, self.out_features = (
            source.in_features,
            source.out_features,
        )
        codes, scales = mxfp8_cache.encode(source.weight.detach().float())
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)
        self.register_buffer(
            "activation_scales",
            torch.full((1, self.in_features // 32), 127, dtype=torch.int8),
        )
        self.bias = source.bias
        self.calibrating = True
        self.register_buffer(
            "maximum", torch.zeros(self.in_features // 32), persistent=False
        )

    def forward(self, values):
        shape = values.shape
        values = values.reshape(-1, self.in_features)
        if self.calibrating:
            maximum = (
                values.detach()
                .reshape(-1, self.in_features // 32, 32)
                .abs()
                .amax((0, 2))
            )
            self.maximum = torch.maximum(self.maximum, maximum)
            result = nn.functional.linear(
                values, decode(self.codes, self.scales), self.bias
            )
        else:
            result = matmul(
                encode_fixed(values, self.activation_scales),
                self.activation_scales,
                self.codes,
                self.scales,
            )
            if self.bias is not None:
                result = result + self.bias
        return result.reshape(*shape[:-1], self.out_features)


class Embedding(nn.Module):
    def __init__(self, source, scale=1.0):
        super().__init__()
        self.scale = scale
        codes, scales = mxfp8_cache.encode(source.weight.detach().float())
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)
        self.num_embeddings, self.embedding_dim = (
            source.num_embeddings,
            source.embedding_dim,
        )

    def forward(self, tokens):
        return (
            decode(
                nn.functional.embedding(tokens, self.codes),
                nn.functional.embedding(tokens, self.scales),
            )
            * self.scale
        )


def prepare(model, *, embedding_scales=None):
    """Replace supported weight consumers before collecting activation ranges."""
    embedding_scales = {} if embedding_scales is None else embedding_scales
    for name, module in list(model.named_modules()):
        if not name:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        if type(module) is nn.Linear:
            setattr(parent, child_name, Linear(module))
        elif isinstance(module, nn.Embedding):
            setattr(
                parent,
                child_name,
                Embedding(module, embedding_scales.get(name, 1.0)),
            )
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register("buddy_mxfp8", attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        "buddy_mxfp8", ALL_MASK_ATTENTION_FUNCTIONS["eager"]
    )
    model.set_attn_implementation("buddy_mxfp8")
    return model


def finish_calibration(model):
    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            module.activation_scales.copy_(scale_for(module.maximum)[None, :])
            module.calibrating = False
            scales[name] = (module.activation_scales.to(torch.int16) & 255).tolist()
            del module.maximum
    return scales


class CacheCalibration:
    """Observe per-head KV blocks while the model executes calibration samples."""

    def __init__(self):
        self.maximum = {}
        self.key_minimum = {}
        self.key_maximum = {}

    def update(self, key, value, layer_idx):
        maximum = torch.stack(
            [
                t.detach()
                .reshape(*t.shape[:-1], t.shape[-1] // 32, 32)
                .abs()
                .amax((0, 2, 4))
                for t in (key, value)
            ]
        )
        low, high = key.detach().amin((0, 2)), key.detach().amax((0, 2))
        if layer_idx in self.maximum:
            low = torch.minimum(low, self.key_minimum[layer_idx])
            high = torch.maximum(high, self.key_maximum[layer_idx])
            maximum = torch.maximum(maximum, self.maximum[layer_idx])
        self.maximum[layer_idx] = maximum
        self.key_minimum[layer_idx], self.key_maximum[layer_idx] = low, high
        return key, value

    def parameters(self):
        centers = {}
        for layer, maximum in self.maximum.items():
            low, high = self.key_minimum[layer], self.key_maximum[layer]
            centers[layer] = (low + high) * 0.5
            radius = (high - low) * 0.5
            maximum[0] = radius.reshape(*radius.shape[:-1], -1, 32).amax(-1)
        return self.maximum, centers


class Cache:
    """Explicit fixed-capacity cache; only MXFP8 bytes persist between calls."""

    def __init__(self, codes, positions, layouts, scales, centers, capacity):
        self.codes = codes
        self.positions = positions
        self.layouts = layouts
        self.scales = scales
        self.centers = centers
        self.capacity = capacity
        self.updated = []

    def update(self, key, value, layer_idx):
        offset, heads, width = self.layouts[layer_idx]
        size = 2 * heads * self.capacity * width
        old = self.codes[offset : offset + size].reshape(
            2, 1, heads, self.capacity, width
        )
        scales = self.scales[layer_idx]
        # A shared K translation cancels in softmax(Q K^T). Keep it centered.
        key = key - self.centers[layer_idx].reshape(1, heads, 1, width)
        keys = encode_fixed(key, scales[0].reshape(1, heads, 1, width // 32))
        values = encode_fixed(value, scales[1].reshape(1, heads, 1, width // 32))
        current = torch.stack((keys, values))
        update = torch.arange(self.capacity, device=key.device)[
            None, None, None, :, None
        ] == self.positions.reshape(1, 1, 1, 1, 1)
        packed = torch.where(update, current, old)
        self.updated.append(packed.flatten())
        return decode(packed[0], scales[0].reshape(1, heads, 1, width // 32)), decode(
            packed[1], scales[1].reshape(1, heads, 1, width // 32)
        )


def calibrate(model, dataset, *, samples, batch_size):
    from torch.utils.data import DataLoader, Subset

    if not 0 < samples <= len(dataset):
        raise ValueError("Calibration sample count must be within the dataset size")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))[
        :samples
    ]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size)
    cache = CacheCalibration()
    completed = 0
    with torch.no_grad():
        for tokens in loader:
            positions = torch.arange(tokens.shape[1])[None, :]
            visible = positions[:, :, None] >= positions[:, None, :]
            mask = torch.where(visible[:, None], 0.0, torch.finfo(torch.float32).min)
            model(
                input_ids=tokens,
                position_ids=positions,
                attention_mask={
                    "full_attention": mask,
                    "sliding_attention": mask,
                },
                past_key_values=cache,
                use_cache=True,
            )
            completed += tokens.shape[0]
            print(f"Calibrated {completed}/{samples} sequences", flush=True)
    return finish_calibration(model), cache
