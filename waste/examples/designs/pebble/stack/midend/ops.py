"""Pebble's banked INT8 operator emission."""

from copy import deepcopy

from buddy_mlir import ir
from buddy_mlir.dialects import tosa as dialect

from stack.compiler.frontend.operation import (
    Int8MatmulOp,
    QuantizedDepthwiseConv2dOp,
    QuantizedConv2dOp,
    QuantizedMatmulOp,
)
from stack.compiler.frontend.layout.provenance import has_nhwc_layout
from stack.compiler.frontend.ops import linalg, tosa
from stack.compiler.frontend.ops.int8 import _rescale


def contraction(node, symbols):
    linear = isinstance(node, Int8MatmulOp)
    if linear:
        activation, weight, activation_scale, weight_scale, bias = node.args
        operation = QuantizedMatmulOp()
    else:
        activation, weight = node.args[:2]
        (
            bias,
            stride,
            padding,
            dilation,
            transposed,
            _,
            groups,
            activation_scale,
            weight_scale,
        ) = node.args[2:]
        source = symbols[("__buddy_ops_by_name__", 0)][activation]
        input_shape = list(source.tensor_meta["shape"])
        depthwise = (
            groups > 1 and groups == input_shape[1] == node.tensor_meta["shape"][1]
        )
        if (
            transposed
            or list(dilation) != [1, 1]
            or stride[0] != stride[1]
            or padding[0] != padding[1]
            or (groups != 1 and not depthwise)
        ):
            raise ValueError(
                "Pebble requires ordinary or depthwise convolution with unit dilation"
            )
        operation = QuantizedDepthwiseConv2dOp() if depthwise else QuantizedConv2dOp()
        operation._input_shape = input_shape
        operation._weight_shape = list(
            ir.RankedTensorType(symbols[(weight, 0)].type).shape
        )
        operation._output_shape = list(node.tensor_meta["shape"])
        operation._output_nhwc = node.tensor_meta.get("activation_layout") == "nhwc"
        operation._stride, operation._padding = stride[0], padding[0]
        value = symbols[(activation, 0)]
        if not has_nhwc_layout(value):
            n, c, h, w = input_shape
            value = dialect.TransposeOp(
                ir.RankedTensorType.get([n, h, w, c], ir.IntegerType.get_signless(8)),
                value,
                tosa._create_permutation_attr([0, 2, 3, 1]),
            ).result
            symbols = {**symbols, (activation, 0): value}
    operation.name = node.name
    operation._arguments = [activation, weight]
    operation.tensor_meta = dict(node.tensor_meta)
    operation.trace_meta = deepcopy(node.trace_meta)
    channels = node.tensor_meta["shape"][1]
    operation._bias_i32 = [0] * channels
    operation._dequant_scale = [1.0] * channels
    operation._requant_scale = [1.0] * channels
    operation._final_output = True
    operation._activation = 0
    operation._output_scale = 1.0
    operation._lut_i8 = [0]
    result = (linalg.quantized_matmul_op if linear else linalg.quantized_conv2d_op)(
        operation, symbols
    )
    return _rescale(
        result,
        activation_scale,
        weight_scale,
        bias,
        symbols,
        channel_axis=3 if not linear and operation._output_nhwc else 1,
    )


def permute(node, symbols):
    # Preserve NHWC storage across a projection's logical NCHW boundary.
    if list(node.args[1]) == [0, 3, 1, 2]:
        value = symbols[(str(node.args[0]), 0)]
        n, c, h, w = node.tensor_meta["shape"]
        if list(ir.RankedTensorType(value.type).shape) != [n, h, w, c]:
            raise ValueError("Pebble projection boundary requires NHWC storage")
        return value
    return tosa.permute_op(node, symbols)


ops_registry = {
    **tosa.ops_registry,
    "Int8Conv2dOp": contraction,
    "Int8MatmulOp": contraction,
    "PermuteOp": permute,
}
