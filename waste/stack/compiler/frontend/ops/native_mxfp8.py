"""Portable MXFP8 codecs: E4M3FN values and one E8M0 scale per 32 values."""

from buddy_mlir import ir
from buddy_mlir.dialects import arith, linalg, math, tensor

from stack.compiler.frontend.ops.reductions import _generic


def _constant(dtype, value):
    return arith.ConstantOp(dtype, value).result


def _power(scale):
    i32, f32 = (ir.IntegerType.get_signless(32), ir.F32Type.get())
    exponent = arith.ExtUIOp(i32, scale).result
    bits = arith.ShLIOp(exponent, _constant(i32, 23)).result
    normal = arith.BitcastOp(f32, bits).result
    return arith.SelectOp(
        arith.CmpIOp(arith.CmpIPredicate.eq, exponent, _constant(i32, 0)).result,
        _constant(f32, 2.0 ** (-127)),
        normal,
    ).result


def _decode(code, scale):
    i32, f32 = (ir.IntegerType.get_signless(32), ir.F32Type.get())
    value = arith.ExtUIOp(i32, code).result
    sign = arith.AndIOp(value, _constant(i32, 128)).result
    fraction = arith.AndIOp(value, _constant(i32, 7)).result
    exponent = arith.AndIOp(
        arith.ShRUIOp(value, _constant(i32, 3)).result, _constant(i32, 15)
    ).result
    bits = arith.OrIOp(
        arith.ShLIOp(
            arith.AddIOp(exponent, _constant(i32, 120)).result,
            _constant(i32, 23),
        ).result,
        arith.ShLIOp(fraction, _constant(i32, 20)).result,
    ).result
    normal = arith.BitcastOp(f32, bits).result
    subnormal = arith.MulFOp(
        arith.UIToFPOp(f32, fraction).result, _constant(f32, 1.0 / 512)
    ).result
    magnitude = arith.SelectOp(
        arith.CmpIOp(arith.CmpIPredicate.eq, exponent, _constant(i32, 0)).result,
        subnormal,
        normal,
    ).result
    signed = arith.SelectOp(
        arith.CmpIOp(arith.CmpIPredicate.ne, sign, _constant(i32, 0)).result,
        arith.NegFOp(magnitude).result,
        magnitude,
    ).result
    return arith.MulFOp(signed, _power(scale)).result


def _encode(value, scale):
    i8, i32, f32 = (
        ir.IntegerType.get_signless(8),
        ir.IntegerType.get_signless(32),
        ir.F32Type.get(),
    )
    scaled = arith.DivFOp(value, _power(scale)).result
    magnitude = arith.MinimumFOp(
        math.AbsFOp(scaled).result, _constant(f32, 448.0)
    ).result
    bits = arith.BitcastOp(i32, magnitude).result
    exponent = arith.SubIOp(
        arith.ShRUIOp(bits, _constant(i32, 23)).result, _constant(i32, 120)
    ).result
    fraction = arith.AndIOp(bits, _constant(i32, 8388607)).result
    high = arith.ShRUIOp(fraction, _constant(i32, 20)).result
    low = arith.AndIOp(fraction, _constant(i32, 1048575)).result
    increment = arith.OrIOp(
        arith.CmpIOp(arith.CmpIPredicate.ugt, low, _constant(i32, 524288)).result,
        arith.AndIOp(
            arith.CmpIOp(arith.CmpIPredicate.eq, low, _constant(i32, 524288)).result,
            arith.CmpIOp(
                arith.CmpIPredicate.ne,
                arith.AndIOp(high, _constant(i32, 1)).result,
                _constant(i32, 0),
            ).result,
        ).result,
    ).result
    rounded = arith.AddIOp(high, arith.ExtUIOp(i32, increment).result).result
    normal = arith.AddIOp(
        arith.MulIOp(exponent, _constant(i32, 8)).result, rounded
    ).result
    subnormal = arith.FPToSIOp(
        i32,
        math.RoundEvenOp(arith.MulFOp(magnitude, _constant(f32, 512.0)).result).result,
    ).result
    code = arith.SelectOp(
        arith.CmpIOp(arith.CmpIPredicate.sle, exponent, _constant(i32, 0)).result,
        subnormal,
        normal,
    ).result
    sign = arith.AndIOp(
        arith.ShRUIOp(arith.BitcastOp(i32, value).result, _constant(i32, 24)).result,
        _constant(i32, 128),
    ).result
    return arith.TruncIOp(i8, arith.OrIOp(code, sign).result).result


def codec(node, symbols):
    values, scales = [symbols[name, 0] for name in node.args]
    shape = list(ir.RankedTensorType(values.type).shape)
    rank = len(shape)
    scale_shape = list(ir.RankedTensorType(scales.type).shape)
    dims = [ir.AffineExpr.get_dim(i) for i in range(rank)]
    scale_dims = [
        ir.AffineExpr.get_constant(0) if size == 1 else dim
        for dim, size in zip(dims[:-1], scale_shape[:-1], strict=True)
    ]
    scale_dims.append(
        ir.AffineExpr.get_floor_div(dims[-1], ir.AffineExpr.get_constant(32))
    )
    identity = ir.AffineMap.get_identity(rank)
    scale_map = ir.AffineMap.get(rank, 0, scale_dims)
    encoding = type(node).__name__ == "NativeMXFP8EncodeOp"
    dtype = ir.IntegerType.get_signless(8) if encoding else ir.F32Type.get()
    output = tensor.EmptyOp(shape, dtype).result
    operation = _generic(
        [values, scales],
        output,
        [identity, scale_map, identity],
        ["parallel"] * rank,
    )
    operation.attributes["buddy.mxfp8"] = ir.StringAttr.get(
        "encode" if encoding else "decode"
    )
    body = ir.Block.create_at_start(
        operation.region,
        [
            ir.RankedTensorType(values.type).element_type,
            ir.IntegerType.get_signless(8),
            dtype,
        ],
    )
    with ir.InsertionPoint(body):
        result = (_encode if encoding else _decode)(*body.arguments[:2])
        linalg.YieldOp([result])
    return operation.result


def concat(node, symbols):
    """LLM tensors retain logical axes, including singleton head dimensions."""
    from buddy_mlir.dialects import tosa

    values = [symbols[(name, 0)] for name in node.args[0]]
    axis = node.args[1] if len(node.args) == 2 else 0
    rank = ir.RankedTensorType(values[0].type).rank
    return tosa.ConcatOp(values, axis % rank).result


def matmul(node, symbols):
    """Decode E4M3 operands inside the reduction without expanding weight storage."""
    from stack.compiler.frontend.ops.reductions import _cast, _filled

    activation, activation_scales, weights, weight_scales = [
        symbols[(name, 0)] for name in node.args
    ]
    m, n, k = [ir.AffineExpr.get_dim(i) for i in range(3)]
    block = ir.AffineExpr.get_floor_div(k, ir.AffineExpr.get_constant(32))
    zero = ir.AffineExpr.get_constant(0)
    maps = [
        ir.AffineMap.get(3, 0, dims)
        for dims in ([m, k], [zero, block], [n, k], [n, block], [m, n])
    ]
    f64, i8 = ir.F64Type.get(), ir.IntegerType.get_signless(8)
    output = _filled(list(node.tensor_meta["shape"]), f64, 0.0)
    operation = _generic(
        [activation, activation_scales, weights, weight_scales],
        output,
        maps,
        ["parallel", "parallel", "reduction"],
    )
    operation.attributes["buddy.mxfp8"] = ir.StringAttr.get("matmul")
    body = ir.Block.create_at_start(operation.region, [i8, i8, i8, i8, f64])
    with ir.InsertionPoint(body):
        a, a_scale, w, w_scale, accumulator = body.arguments
        product = arith.MulFOp(
            arith.ExtFOp(f64, _decode(a, a_scale)).result,
            arith.ExtFOp(f64, _decode(w, w_scale)).result,
        ).result
        linalg.YieldOp([arith.AddFOp(accumulator, product).result])
    return _cast(operation.result, ir.F32Type.get())
