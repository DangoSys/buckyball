"""Native signed INT8 contractions with INT32 accumulators."""

import math as python_math

from buddy_mlir import ir
from buddy_mlir.dialects import arith, linalg, math, tensor, tosa


def _maps(values):
    return ir.ArrayAttr.get([ir.AffineMapAttr.get(value) for value in values])


def _iterators(parallel, reduction):
    return ir.ArrayAttr.get(
        [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * parallel
        + [ir.Attribute.parse("#linalg.iterator_type<reduction>")] * reduction
    )


def _zero(shape):
    dtype = ir.IntegerType.get_signless(32)
    empty = tensor.EmptyOp(shape, dtype)
    zero = arith.ConstantOp(dtype, 0).result
    fill = linalg.FillOp([empty.result.type], [zero], [empty.result])
    linalg.fill_builtin_region(fill.operation)
    return fill.result


def _multiply_accumulate(op):
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    body = ir.Block.create_at_start(op.region, [i8, i8, i32])
    with ir.InsertionPoint(body):
        lhs = arith.ExtSIOp(i32, body.arguments[0]).result
        rhs = arith.ExtSIOp(i32, body.arguments[1]).result
        product = arith.MulIOp(lhs, rhs).result
        linalg.YieldOp([arith.AddIOp(body.arguments[2], product).result])


def _rescale(
    accumulator,
    activation_scale,
    weight_scale,
    bias,
    symbols,
    *,
    channel_axis=1,
):
    from buddy.compiler.ops.tosa import _create_shape_operand, _gen_arith_binary_op

    shape = list(ir.RankedTensorType(accumulator.type).shape)
    rank = len(shape)
    f32 = ir.F32Type.get()
    if isinstance(activation_scale, str):
        activation_scale = symbols[(activation_scale, 0)]
    else:
        ty = ir.RankedTensorType.get([1] * rank, f32)
        activation_scale = tosa.ConstOp(
            ir.DenseElementsAttr.get_splat(ty, ir.FloatAttr.get(f32, activation_scale))
        ).result
    activation_shape = list(ir.RankedTensorType(activation_scale.type).shape)
    if channel_axis == 3 and len(activation_shape) == 4 and activation_shape[1] != 1:
        activation_scale = tosa.ReshapeOp(
            activation_scale,
            _create_shape_operand([1, 1, 1, activation_shape[1]]),
        ).result
    scale = symbols[(weight_scale, 0)]
    count = python_math.prod(ir.RankedTensorType(scale.type).shape)
    broadcast_shape = [1] * rank
    broadcast_shape[channel_axis] = count
    scale = tosa.ReshapeOp(scale, _create_shape_operand(broadcast_shape)).result
    combined_scale = _gen_arith_binary_op(activation_scale, scale, tosa.MulOp).result
    restored = (
        accumulator
        if ir.RankedTensorType(accumulator.type).element_type == f32
        else tosa.CastOp(ir.RankedTensorType.get(shape, f32), accumulator).result
    )
    result = _gen_arith_binary_op(restored, combined_scale, tosa.MulOp).result
    if bias is not None:
        value = symbols[(bias, 0)]
        bias_shape = [1] * rank
        bias_shape[channel_axis] = shape[channel_axis]
        value = tosa.ReshapeOp(value, _create_shape_operand(bias_shape)).result
        result = _gen_arith_binary_op(result, value, tosa.AddOp).result
    return result


def conv2d(node, symbols):
    from stack.compiler.frontend.layout.provenance import has_nhwc_layout

    value, weight = [symbols[(name, 0)] for name in node.args[:2]]
    input_nhwc = has_nhwc_layout(value)
    output_nhwc = node.tensor_meta.get("activation_layout") == "nhwc"
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
    if transposed:
        raise NotImplementedError("INT8 transpose convolution is not supported")
    input_shape = list(ir.RankedTensorType(value.type).shape)
    stored_shape = list(ir.RankedTensorType(weight.type).shape)
    permutation = node.tensor_meta.get("weight_permutation", [0, 1, 2, 3])
    weight_shape = node.tensor_meta.get(
        "weight_shape", [stored_shape[permutation.index(i)] for i in range(4)]
    )
    shape = list(node.tensor_meta["shape"])
    if output_nhwc:
        shape = [shape[0], shape[2], shape[3], shape[1]]
    ph, pw = padding
    if ph or pw:
        low = [0, ph, pw, 0] if input_nhwc else [0, 0, ph, pw]
        padded_shape = [size + 2 * pad for size, pad in zip(input_shape, low)]
        op = tensor.PadOp(
            ir.RankedTensorType.get(padded_shape, ir.IntegerType.get_signless(8)),
            value,
            [],
            [],
            low,
            low,
        )
        block = ir.Block.create_at_start(op.region, [ir.IndexType.get()] * 4)
        with ir.InsertionPoint(block):
            tensor.YieldOp(arith.ConstantOp(ir.IntegerType.get_signless(8), 0).result)
        value = op.result
    n, f, h, w, c, kh, kw = [ir.AffineExpr.get_dim(i) for i in range(7)]
    channel = (
        ir.AffineExpr.get_floor_div(f, weight_shape[0] // groups) * weight_shape[1] + c
    )
    if node.tensor_meta.get("weight_layout") == "blocked_conv":
        lanes = node.tensor_meta["output_lanes"]
        d0, d1, d2, d3 = [ir.AffineExpr.get_dim(i) for i in range(4)]
        indices = [
            ir.AffineExpr.get_floor_div(d0, lanes),
            d1,
            d2 * weight_shape[3] + d3,
            ir.AffineExpr.get_mod(d0, lanes),
        ]
        unpacked = tensor.EmptyOp(weight_shape, ir.IntegerType.get_signless(8))
        unpack = linalg.GenericOp(
            [unpacked.result.type],
            [weight],
            [unpacked.result],
            _maps([ir.AffineMap.get(4, 0, indices), ir.AffineMap.get_identity(4)]),
            _iterators(4, 0),
        )
        block = ir.Block.create_at_start(
            unpack.region, [ir.IntegerType.get_signless(8)] * 2
        )
        with ir.InsertionPoint(block):
            linalg.YieldOp([block.arguments[0]])
        weight = unpack.result
        permutation = [0, 1, 2, 3]
    weight_indices = [[f, c, kh, kw][i] for i in permutation]
    height = h * stride[0] + kh * dilation[0]
    width = w * stride[1] + kw * dilation[1]
    maps = [
        ir.AffineMap.get(
            7,
            0,
            (
                [n, height, width, channel]
                if input_nhwc
                else [n, channel, height, width]
            ),
        ),
        ir.AffineMap.get(7, 0, weight_indices),
        ir.AffineMap.get(7, 0, [n, h, w, f] if output_nhwc else [n, f, h, w]),
    ]
    output = _zero(shape)
    op = linalg.GenericOp(
        [output.type], [value, weight], [output], _maps(maps), _iterators(4, 3)
    )
    _multiply_accumulate(op)
    op.operation.attributes["buddy.int8_conv"] = ir.DictAttr.get(
        {
            "stride": ir.DenseI64ArrayAttr.get(stride),
            "dilation": ir.DenseI64ArrayAttr.get(dilation),
            "groups": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), groups),
            "weight_permutation": ir.DenseI64ArrayAttr.get(permutation),
        }
    )
    result = _rescale(
        op.result,
        activation_scale,
        weight_scale,
        bias,
        symbols,
        channel_axis=3 if output_nhwc else 1,
    )
    if output_nhwc:
        result.owner.attributes["buddy.layout"] = ir.StringAttr.get("nhwc")
    return result


def matmul(node, symbols):
    activation, weight, activation_scale, weight_scale, bias = node.args
    lhs, rhs = symbols[(activation, 0)], symbols[(weight, 0)]
    shape = list(node.tensor_meta["shape"])
    m, n, k = [ir.AffineExpr.get_dim(i) for i in range(3)]
    maps = [
        ir.AffineMap.get(3, 0, [m, k]),
        ir.AffineMap.get(
            3, 0, [n, k] if node.tensor_meta.get("transpose_b") else [k, n]
        ),
        ir.AffineMap.get(3, 0, [m, n]),
    ]
    output = _zero(shape)
    op = linalg.GenericOp(
        [output.type], [lhs, rhs], [output], _maps(maps), _iterators(2, 1)
    )
    _multiply_accumulate(op)
    op.operation.attributes["buddy.int8_matmul"] = ir.UnitAttr.get()
    return _rescale(op.result, activation_scale, weight_scale, bias, symbols)


def quantize_activation(node, symbols):
    value = symbols[(node.args[0], 0)]
    shape = list(ir.RankedTensorType(value.type).shape)
    f32, i8 = ir.F32Type.get(), ir.IntegerType.get_signless(8)
    output = tensor.EmptyOp(shape, i8)
    identity = ir.AffineMap.get_identity(len(shape))
    op = linalg.GenericOp(
        [output.result.type],
        [value],
        [output.result],
        _maps([identity, identity]),
        _iterators(len(shape), 0),
    )
    multiplier = 1.0 / node.args[1]
    op.operation.attributes["buckyball.quant_f32_to_i8"] = ir.BoolAttr.get(True)
    op.operation.attributes["nchw_to_nhwc"] = ir.BoolAttr.get(False)
    op.operation.attributes["scale"] = ir.FloatAttr.get(f32, multiplier)
    block = ir.Block.create_at_start(op.region, [f32, i8])
    with ir.InsertionPoint(block):
        scale = arith.ConstantOp(f32, ir.FloatAttr.get(f32, multiplier)).result
        value = math.RoundEvenOp(arith.MulFOp(block.arguments[0], scale).result).result
        minimum = arith.ConstantOp(f32, ir.FloatAttr.get(f32, -128.0)).result
        maximum = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 127.0)).result
        value = arith.MinimumFOp(
            arith.MaximumFOp(value, minimum).result, maximum
        ).result
        linalg.YieldOp([arith.FPToSIOp(i8, value).result])
    return op.result
