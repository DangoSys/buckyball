from buddy_mlir import ir


def lower(node, symbols):
    lhs, rhs = [symbols[(name, 0)] for name in node.args]
    result = ir.RankedTensorType.get(node.tensor_meta["shape"], ir.F32Type.get())
    return ir.Operation.create(
        "buckyball.fp32_matmul", operands=[lhs, rhs], results=[result]
    ).result


def apply(graph):
    graph._ops_registry["MatmulOp"] = lower
    graph._ops_registry["BatchMatmulOp"] = lower
