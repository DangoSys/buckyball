"""Stable floating reductions between quantized integer contractions."""

import math as scalar_math

from buddy_mlir import ir
from buddy_mlir.dialects import arith, linalg, math, tensor


def _filled(shape, dtype, value):
    empty = tensor.EmptyOp(shape, dtype)
    scalar = arith.ConstantOp(dtype, value).result
    result = linalg.FillOp([empty.result.type], [scalar], [empty.result])
    linalg.fill_builtin_region(result.operation)
    return result.result


def _generic(inputs, output, maps, iterators):
    return linalg.GenericOp(
        [output.type],
        inputs,
        [output],
        ir.ArrayAttr.get([ir.AffineMapAttr.get(m) for m in maps]),
        ir.ArrayAttr.get(
            [
                ir.Attribute.parse("#linalg.iterator_type<" + item + ">")
                for item in iterators
            ]
        ),
    )


def _cast(value, dtype):
    ty = ir.RankedTensorType(value.type)
    identity = ir.AffineMap.get_identity(ty.rank)
    empty = tensor.EmptyOp(ty.shape, dtype).result
    op = _generic([value], empty, [identity, identity], ["parallel"] * ty.rank)
    body = ir.Block.create_at_start(op.region, [ty.element_type, dtype])
    with ir.InsertionPoint(body):
        converted = (
            arith.ExtFOp(dtype, body.arguments[0]).result
            if ir.FloatType(dtype).width > ir.FloatType(ty.element_type).width
            else arith.TruncFOp(dtype, body.arguments[0]).result
        )
        linalg.YieldOp([converted])
    return op.result


def _sum_f64(value, axes, keepdim=True):
    ty = ir.RankedTensorType(value.type)
    shape = list(ty.shape)
    dims = [ir.AffineExpr.get_dim(i) for i in range(ty.rank)]
    output_shape = (
        [1 if i in axes else size for i, size in enumerate(shape)]
        if keepdim
        else [size for i, size in enumerate(shape) if i not in axes]
    )
    results = (
        [
            ir.AffineExpr.get_constant(0) if i in axes else dim
            for i, dim in enumerate(dims)
        ]
        if keepdim
        else [dim for i, dim in enumerate(dims) if i not in axes]
    )
    f64 = ir.F64Type.get()
    op = _generic(
        [value],
        _filled(output_shape, f64, 0.0),
        [
            ir.AffineMap.get_identity(ty.rank),
            ir.AffineMap.get(ty.rank, 0, results),
        ],
        ["reduction" if i in axes else "parallel" for i in range(ty.rank)],
    )
    body = ir.Block.create_at_start(op.region, [ty.element_type, f64])
    with ir.InsertionPoint(body):
        number = (
            body.arguments[0]
            if ty.element_type == f64
            else arith.ExtFOp(f64, body.arguments[0]).result
        )
        linalg.YieldOp([arith.AddFOp(number, body.arguments[1]).result])
    return op.result


def _mean_f64(value, axes, keepdim=True):
    summed = _sum_f64(value, axes, keepdim)
    ty = ir.RankedTensorType(summed.type)
    identity = ir.AffineMap.get_identity(ty.rank)
    op = _generic(
        [summed],
        tensor.EmptyOp(ty.shape, ty.element_type).result,
        [identity, identity],
        ["parallel"] * ty.rank,
    )
    body = ir.Block.create_at_start(op.region, [ty.element_type] * 2)
    with ir.InsertionPoint(body):
        count = scalar_math.prod(ir.RankedTensorType(value.type).shape[i] for i in axes)
        divisor = arith.ConstantOp(ty.element_type, float(count)).result
        linalg.YieldOp([arith.DivFOp(body.arguments[0], divisor).result])
    return op.result


def mean(node, symbols):
    from stack.compiler.frontend.ops.tosa import _nhwc_to_nchw
    from stack.compiler.frontend.layout.provenance import has_nhwc_layout

    value = symbols[(node.args[0], 0)]
    if has_nhwc_layout(value):
        value = _nhwc_to_nchw(value)
    rank = ir.RankedTensorType(value.type).rank
    return _cast(
        _mean_f64(value, {int(axis) % rank for axis in node.args[1]}, node.args[2]),
        ir.F32Type.get(),
    )


def bmm(node, symbols):
    lhs, rhs = [symbols[(name, 0)] for name in node.args]
    f32, f64 = ir.F32Type.get(), ir.F64Type.get()
    b, m, n, k = [ir.AffineExpr.get_dim(i) for i in range(4)]
    maps = [
        ir.AffineMap.get(4, 0, [b, m, k]),
        ir.AffineMap.get(4, 0, [b, k, n]),
        ir.AffineMap.get(4, 0, [b, m, n]),
    ]
    op = _generic(
        [lhs, rhs],
        _filled(node.tensor_meta["shape"], f64, 0.0),
        maps,
        ["parallel"] * 3 + ["reduction"],
    )
    body = ir.Block.create_at_start(op.region, [f32, f32, f64])
    with ir.InsertionPoint(body):
        a = arith.ExtFOp(f64, body.arguments[0]).result
        b = arith.ExtFOp(f64, body.arguments[1]).result
        linalg.YieldOp([math.FmaOp(a, b, body.arguments[2]).result])
    return _cast(op.result, f32)


def softmax(node, symbols):
    value = symbols[(node.args[0], 0)]
    ty = ir.RankedTensorType(value.type)
    shape = list(ty.shape)
    rank = ty.rank
    axis = int(node.args[1]) % rank
    f32, f64 = ir.F32Type.get(), ir.F64Type.get()
    dims = [ir.AffineExpr.get_dim(i) for i in range(rank)]
    reduced_shape = [1 if i == axis else size for i, size in enumerate(shape)]
    small = ir.AffineMap.get(
        rank,
        0,
        [
            ir.AffineExpr.get_constant(0) if i == axis else dim
            for i, dim in enumerate(dims)
        ],
    )
    identity = ir.AffineMap.get_identity(rank)
    maximum = _generic(
        [value],
        _filled(reduced_shape, f32, float("-inf")),
        [identity, small],
        ["reduction" if i == axis else "parallel" for i in range(rank)],
    )
    body = ir.Block.create_at_start(maximum.region, [f32, f32])
    with ir.InsertionPoint(body):
        linalg.YieldOp([arith.MaximumFOp(*body.arguments).result])
    exponent = _generic(
        [value, maximum.result],
        tensor.EmptyOp(shape, f64).result,
        [identity, small, identity],
        ["parallel"] * rank,
    )
    body = ir.Block.create_at_start(exponent.region, [f32, f32, f64])
    with ir.InsertionPoint(body):
        x = arith.ExtFOp(f64, body.arguments[0]).result
        maximum_value = arith.ExtFOp(f64, body.arguments[1]).result
        linalg.YieldOp([math.ExpOp(arith.SubFOp(x, maximum_value).result).result])
    summed = _sum_f64(exponent.result, {axis})
    output = _generic(
        [exponent.result, summed],
        tensor.EmptyOp(shape, f32).result,
        [identity, small, identity],
        ["parallel"] * rank,
    )
    body = ir.Block.create_at_start(output.region, [f64, f64, f32])
    with ir.InsertionPoint(body):
        linalg.YieldOp(
            [
                arith.TruncFOp(
                    f32,
                    arith.DivFOp(body.arguments[0], body.arguments[1]).result,
                ).result
            ]
        )
    return output.result


def layer_norm(node, symbols):
    value = symbols[(node.args[0], 0)]
    weight = symbols[(node.args[2], 0)]
    bias = symbols[(node.args[3], 0)]
    shape = list(ir.RankedTensorType(value.type).shape)
    rank = len(shape)
    normalized = len(node.args[1])
    axes = set(range(rank - normalized, rank))
    f32, f64 = ir.F32Type.get(), ir.F64Type.get()
    mean_value = _mean_f64(value, axes)
    dims = [ir.AffineExpr.get_dim(i) for i in range(rank)]
    identity = ir.AffineMap.get_identity(rank)
    small = ir.AffineMap.get(
        rank,
        0,
        [
            ir.AffineExpr.get_constant(0) if i in axes else dim
            for i, dim in enumerate(dims)
        ],
    )
    parameters = ir.AffineMap.get(rank, 0, dims[rank - normalized :])
    squared = _generic(
        [value, mean_value],
        tensor.EmptyOp(shape, f64).result,
        [identity, small, identity],
        ["parallel"] * rank,
    )
    body = ir.Block.create_at_start(squared.region, [f32, f64, f64])
    with ir.InsertionPoint(body):
        difference = arith.SubFOp(
            arith.ExtFOp(f64, body.arguments[0]).result, body.arguments[1]
        ).result
        linalg.YieldOp([arith.MulFOp(difference, difference).result])
    variance = _mean_f64(squared.result, axes)
    stat_shape = list(ir.RankedTensorType(variance.type).shape)
    stat_identity = ir.AffineMap.get_identity(rank)
    inverse = _generic(
        [variance],
        tensor.EmptyOp(stat_shape, f64).result,
        [stat_identity, stat_identity],
        ["parallel"] * rank,
    )
    body = ir.Block.create_at_start(inverse.region, [f64, f64])
    with ir.InsertionPoint(body):
        adjusted = arith.AddFOp(
            body.arguments[0], arith.ConstantOp(f64, float(node.args[4])).result
        ).result
        root = math.SqrtOp(adjusted).result
        linalg.YieldOp([arith.DivFOp(arith.ConstantOp(f64, 1.0).result, root).result])
    output = _generic(
        [value, mean_value, inverse.result, weight, bias],
        tensor.EmptyOp(shape, f32).result,
        [identity, small, small, parameters, parameters, identity],
        ["parallel"] * rank,
    )
    body = ir.Block.create_at_start(output.region, [f32, f64, f64, f32, f32, f32])
    with ir.InsertionPoint(body):
        difference = arith.SubFOp(
            arith.ExtFOp(f64, body.arguments[0]).result, body.arguments[1]
        ).result
        normalized_value = arith.MulFOp(difference, body.arguments[2]).result
        gamma = arith.ExtFOp(f64, body.arguments[3]).result
        beta = arith.ExtFOp(f64, body.arguments[4]).result
        result = math.FmaOp(normalized_value, gamma, beta).result
        linalg.YieldOp([arith.TruncFOp(f32, result).result])
    return output.result, _cast(mean_value, f32), _cast(inverse.result, f32)


def gelu(node, symbols):
    value = symbols[(node.args[0], 0)]
    ty = ir.RankedTensorType(value.type)
    f32, f64 = ir.F32Type.get(), ir.F64Type.get()
    identity = ir.AffineMap.get_identity(ty.rank)
    op = _generic(
        [value],
        tensor.EmptyOp(ty.shape, f32).result,
        [identity, identity],
        ["parallel"] * ty.rank,
    )
    body = ir.Block.create_at_start(op.region, [f32, f32])
    with ir.InsertionPoint(body):
        x = arith.ExtFOp(f64, body.arguments[0]).result
        scaled = arith.MulFOp(
            x, arith.ConstantOp(f64, scalar_math.sqrt(0.5)).result
        ).result
        erf = math.ErfOp(scaled).result
        factor = arith.AddFOp(arith.ConstantOp(f64, 1.0).result, erf).result
        half = arith.MulFOp(x, arith.ConstantOp(f64, 0.5).result).result
        result = arith.MulFOp(half, factor).result
        linalg.YieldOp([arith.TruncFOp(f32, result).result])
    return op.result


def tanh(node, symbols):
    value = symbols[(node.args[0], 0)]
    ty = ir.RankedTensorType(value.type)
    f32, f64 = ir.F32Type.get(), ir.F64Type.get()
    identity = ir.AffineMap.get_identity(ty.rank)
    op = _generic(
        [value],
        tensor.EmptyOp(ty.shape, f32).result,
        [identity, identity],
        ["parallel"] * ty.rank,
    )
    body = ir.Block.create_at_start(op.region, [f32, f32])
    with ir.InsertionPoint(body):
        extended = arith.ExtFOp(f64, body.arguments[0]).result
        result = math.TanhOp(extended).result
        linalg.YieldOp([arith.TruncFOp(f32, result).result])
    return op.result
