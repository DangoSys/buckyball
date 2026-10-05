import array
import math

from buddy_mlir import ir

from . import fp32


def reshape(value, shape):
    shape_type = ir.Type.parse(f"!tosa.shape<{len(shape)}>")
    dimensions = ir.Operation.create(
        "tosa.const_shape",
        results=[shape_type],
        attributes={
            "values": ir.DenseElementsAttr.get(
                array.array("q", shape), type=ir.IndexType.get(), shape=[len(shape)]
            )
        },
    ).result
    result_type = ir.RankedTensorType.get(shape, ir.F32Type.get())
    return ir.Operation.create(
        "tosa.reshape", operands=[value, dimensions], results=[result_type]
    ).result


def lower(node, symbols):
    result = fp32.lower(node, symbols)
    native = result.owner
    if isinstance(native, ir.OpView):
        native = native.operation
    lhs, rhs = native.operands
    repeated = rhs
    while True:
        owner = repeated.owner
        if isinstance(owner, ir.OpView):
            owner = owner.operation
        if not isinstance(owner, ir.Operation) or owner.name != "tosa.reshape":
            break
        repeated = owner.operands[0]
    # Other matmuls keep their native lowering. This pattern is a five-axis KV repeat.
    repeated_type = ir.RankedTensorType(repeated.type)
    if (
        not isinstance(owner, ir.Operation)
        or owner.name != "tosa.add"
        or repeated_type.rank != 5
    ):
        return result
    source, zeros = owner.operands
    zero_op = zeros.owner
    if isinstance(zero_op, ir.OpView):
        zero_op = zero_op.operation
    if not isinstance(zero_op, ir.Operation) or zero_op.name != "tosa.const":
        raise ValueError("grouped attention requires an explicit +0 KV broadcast")
    zero_values = ir.DenseElementsAttr(zero_op.attributes["values"])
    if not zero_values.is_splat:
        raise ValueError("grouped attention KV broadcast must contain only +0")
    zero = ir.FloatAttr(zero_values.get_splat_value()).value
    if zero != 0.0 or math.copysign(1.0, zero) != 1.0:
        raise ValueError("grouped attention KV broadcast must contain only +0")
    if ir.BoolAttr(native.attributes["fused"]).value:
        raise ValueError("grouped attention requires unfused FP32 accumulation")
    if any(size <= 0 for size in repeated_type.shape):
        raise ValueError("grouped attention requires positive static dimensions")
    batch, kv_heads, groups, length, dimension = repeated_type.shape
    source_shape = list(ir.RankedTensorType(source.type).shape)
    left_shape = list(ir.RankedTensorType(lhs.type).shape)
    right_shape = list(ir.RankedTensorType(rhs.type).shape)
    transposed = "rhs_transposed" in native.attributes
    reduction, columns = (dimension, length) if transposed else (length, dimension)
    if (
        source_shape != [batch, kv_heads, 1, length, dimension]
        or len(left_shape) != 3
        or left_shape[0] != batch * kv_heads * groups
        or left_shape[2] != reduction
        or right_shape != [batch * kv_heads * groups, length, dimension]
    ):
        raise ValueError(
            "grouped attention requires contiguous head-major Q/P and KV shapes"
        )
    rows = left_shape[1]
    if rows <= 0:
        raise ValueError("grouped attention requires a positive static row count")
    grouped_left = reshape(lhs, [batch * kv_heads, groups * rows, reduction])
    grouped_right = reshape(source, [batch * kv_heads, length, dimension])
    attributes = {"fused": ir.BoolAttr.get(False)}
    if transposed:
        attributes["rhs_transposed"] = ir.UnitAttr.get()
    grouped_type = ir.RankedTensorType.get(
        [batch * kv_heads, groups * rows, columns], ir.F32Type.get()
    )
    grouped = ir.Operation.create(
        "buckyball.fp32_matmul",
        operands=[grouped_left, grouped_right],
        results=[grouped_type],
        attributes=attributes,
    ).result
    shape = list(ir.RankedTensorType(result.type).shape)
    native.erase()
    # With finite operands, RNE and initial +0, removing the repeat's +0 changes
    # only signed-zero products, which cannot change an unfused dot's result.
    return reshape(grouped, shape)


def apply(graph):
    fp32.apply(graph, fused=False)
    graph._ops_registry["MatmulOp"] = lower
    graph._ops_registry["BatchMatmulOp"] = lower
