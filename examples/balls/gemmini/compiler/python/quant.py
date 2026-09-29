import numpy as np
import torch
from buddy.compiler.graph.operation import Op, OpType, Conv2dOp, AddMMOp, TOp, PermuteOp
from buddy.compiler.graph.type import TensorDType
from buddy.compiler.ops import tosa as ops
from buddy_mlir import ir
from buddy_mlir.dialects import tosa
from stack.compiler.quant.rax import QuantTensor, RaxQuantPackage, write_rax


class GemminiConvOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ReduceType


def lower(node, symbols):
    input, weight, bias = [symbols[(name, 0)] for name in node.args]
    shape = list(node.tensor_meta["shape"])
    linear = len(shape) == 2
    if linear:
        m, k = ir.RankedTensorType(input.type).shape
        input = tosa.ReshapeOp(
            input,
            ops._create_shape_operand([m, k, 1, 1]),
            results=[ir.RankedTensorType.get([m, k, 1, 1], ir.F32Type.get())],
        ).result
        result_shape = [shape[0], shape[1], 1, 1]
    else:
        result_shape = shape
    attributes = {
        name: ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
        for name, value in zip(("kh", "kw", "stride", "padding"), node.geometry)
    }
    attributes.update(
        {
            name: ir.FloatAttr.get(ir.F32Type.get(), float(value))
            for name, value in zip(("inputScale", "weightScale"), node.scales)
        }
    )
    result = ir.Operation.create(
        "buckyball.gemmini_conv",
        operands=[input, weight, bias],
        results=[ir.RankedTensorType.get(result_shape, ir.F32Type.get())],
        attributes=attributes,
    ).result
    if linear:
        result = tosa.ReshapeOp(
            result,
            ops._create_shape_operand(shape),
            results=[ir.RankedTensorType.get(shape, ir.F32Type.get())],
        ).result
    return result


def apply(
    graph,
    params,
    names,
    output,
    name,
    calibration,
    quantized,
    packer,
    *,
    reorder,
    pool_result,
):
    parameters = list(graph.params)
    inputs = [graph._body[i] for i in graph._inputs]
    positions = {p.name: i for i, p in enumerate(parameters)}
    originals = {p.name: list(p.tensor_meta["shape"]) for p in parameters}
    removed, consumed = set(), set()
    occurrences = {}
    for node in list(graph._body):
        if isinstance(node, Conv2dOp):
            activation, weight, bias = node.args[:3]
            if node.args[5] != [1, 1] or node.args[6] or node.args[8] != 1:
                raise ValueError(
                    "Gemmini model convolution requires unit dilation and one group"
                )
            if len(set(node.args[3])) != 1 or len(set(node.args[4])) != 1:
                raise ValueError(
                    "Gemmini model convolution requires equal spatial strides/padding"
                )
            geometry = (*originals[weight][2:], node.args[3][0], node.args[4][0])
        elif isinstance(node, AddMMOp):
            bias, activation, transposed = node.args[:3]
            transpose = graph.node_table[transposed]
            if not (
                isinstance(transpose, TOp)
                or isinstance(transpose, PermuteOp)
                and list(transpose.args[1]) == [1, 0]
            ):
                raise ValueError(
                    "Gemmini linear requires a transposed weight parameter"
                )
            weight = transpose._parents[0]
            removed.add(transposed)
            geometry = (1, 1, 1, 0)
        else:
            continue
        index = positions[weight]
        parameter_name = names[index]
        codes, scale = quantized[parameter_name]
        if weight not in consumed:
            packed = reorder(codes)
            params[index] = torch.from_numpy(packed.copy())
            parameters[index].tensor_meta.update(
                shape=list(packed.shape), dtype=TensorDType.Int8
            )
            consumed.add(weight)
        occurrence = occurrences.get(parameter_name, 0)
        input_scale = calibration[parameter_name][occurrence][0]
        occurrences[parameter_name] = occurrence + 1
        replacement = GemminiConvOp()
        replacement._name = node.name
        replacement._arguments = [activation, weight, bias]
        replacement._parents = [activation, weight, bias]
        replacement._children = list(node._children)
        replacement._tensor_meta = dict(node.tensor_meta)
        replacement.trace_meta = node.trace_meta
        replacement.geometry = geometry
        replacement.scales = (input_scale, scale)
        graph._body[graph._body.index(node)] = replacement
        graph.node_table[node.name] = replacement
        if node.name not in parameters[index]._children:
            parameters[index]._children.append(node.name)
    for key in removed:
        node = graph.node_table[key]
        if any(
            not isinstance(graph.node_table[child], GemminiConvOp)
            for child in node._children
        ):
            raise ValueError("Weight transpose has another consumer")
        graph.node_table[node._parents[0]]._children.remove(key)
        graph._body.remove(node)
        del graph.node_table[key]
    if {names[positions[p]] for p in consumed} != set(quantized):
        raise ValueError("Gemmini quantization decisions do not match graph weights")
    graph._fake_params = [graph._body.index(p) for p in parameters]
    graph._inputs = [graph._body.index(p) for p in inputs]
    for group in graph.op_groups:
        graph.op_groups[group] = [
            graph.node_table[n.name]
            for n in graph.op_groups[group]
            if n.name not in removed
        ]
    graph._ops_registry["GemminiConvOp"] = lower
    graph._ops_registry["MaxPool2dOp"] = pool_result
    tensors, weights, floats, scales = [], [], [], []
    wo = fo = so = 0
    for node, parameter, parameter_name in zip(parameters, params, names):
        raw = parameter.detach().numpy().tobytes()
        if node.name in consumed:
            scale = np.asarray(quantized[parameter_name][1], dtype=np.float32).tobytes()
            tensors.append(
                QuantTensor(
                    parameter_name,
                    originals[node.name],
                    list(parameter.shape),
                    "i8",
                    [],
                    wo,
                    len(raw),
                    so,
                    len(scale),
                )
            )
            weights.append(raw)
            scales.append(scale)
            wo += len(raw)
            so += len(scale)
        else:
            tensors.append(
                QuantTensor(
                    parameter_name,
                    originals[node.name],
                    originals[node.name],
                    "f32",
                    [],
                    fo,
                    len(raw),
                    0,
                    0,
                )
            )
            floats.append(raw)
            fo += len(raw)
    package = RaxQuantPackage(
        tensors, b"".join(weights), b"".join(floats), b"".join(scales), {}
    )
    output.mkdir(parents=True, exist_ok=True)
    write_rax(package, output / f"{name}.rax", packer, name)
    (output / "weights.bin").write_bytes(package.weights)
    (output / "params.f32").write_bytes(package.params_f32)
    (output / "scales.bin").write_bytes(package.scales_f32)
