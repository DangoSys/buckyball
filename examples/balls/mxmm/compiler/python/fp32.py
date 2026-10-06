import array

from buddy_mlir import ir


def lower(node, symbols):
    lhs, rhs = [symbols[(name, 0)] for name in node.args]
    result = ir.RankedTensorType.get(node.tensor_meta["shape"], ir.F32Type.get())
    attributes = {"fused": ir.BoolAttr.get(node.fused)}
    owner = rhs.owner
    if isinstance(owner, ir.OpView):
        owner = owner.operation
    reshape = None
    if isinstance(owner, ir.Operation) and owner.name == "tosa.reshape":
        original = ir.RankedTensorType(owner.operands[0].type)
        current = ir.RankedTensorType(rhs.type)
        if (
            original.rank == 4
            and original.shape[0] == 1
            and list(current.shape) == list(original.shape[1:])
        ):
            reshape = current
            owner = owner.operands[0].owner
            if isinstance(owner, ir.OpView):
                owner = owner.operation
    if isinstance(owner, ir.Operation) and owner.name == "tosa.transpose":
        rank = ir.RankedTensorType(owner.operands[0].type).rank
        permutation = list(ir.DenseI32ArrayAttr(owner.attributes["perms"]))
        expected = list(range(rank))
        expected[-2:] = [rank - 1, rank - 2]
        if permutation == expected:
            rhs = owner.operands[0]
            if reshape is not None:
                shape = list(reshape.shape)
                shape[-2:] = [shape[-1], shape[-2]]
                tensor = ir.RankedTensorType.get(shape, ir.F32Type.get())
                shape_type = ir.Type.parse(f"!tosa.shape<{len(shape)}>")
                dimensions = ir.Operation.create(
                    "tosa.const_shape",
                    results=[shape_type],
                    attributes={
                        "values": ir.DenseElementsAttr.get(
                            array.array("q", shape),
                            type=ir.IndexType.get(),
                            shape=[len(shape)],
                        )
                    },
                ).result
                rhs = ir.Operation.create(
                    "tosa.reshape", operands=[rhs, dimensions], results=[tensor]
                ).result
            attributes["rhs_transposed"] = ir.UnitAttr.get()
    return ir.Operation.create(
        "buckyball.fp32_matmul",
        operands=[lhs, rhs],
        results=[result],
        attributes=attributes,
    ).result


def apply(graph, *, fused):
    for node in graph._body:
        if node.__class__.__name__ in ("MatmulOp", "BatchMatmulOp"):
            node.fused = fused
    graph._ops_registry["MatmulOp"] = lower
    graph._ops_registry["BatchMatmulOp"] = lower
