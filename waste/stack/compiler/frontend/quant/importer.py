from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from buddy.compiler.graph.operation import (
    AdaptiveAvgPool2dOp,
    CloneOp,
    AddMMOp,
    AddOp,
    AliasOp,
    CatOp,
    ClampMaxOp,
    ClampMinOp,
    Conv2dOp,
    DivOp,
    ExpOp,
    GetItemOp,
    HardswishOp,
    LowMemoryMaxPoolWithOffsetsOp,
    MatmulOp,
    MaxPool2dOp,
    MeanOp,
    MulOp,
    NegOp,
    PermuteOp,
    ReluOp,
    ReshapeOp,
    SiluOp,
    SplitOp,
    SplitWithSizesOp,
    TOp,
    UnsafeIndexOp,
    ViewOp,
)
from stack.compiler.frontend.operation import (
    QuantizedChannelConcatOp,
    QuantizedChannelSliceOp,
    QuantizedDepthwiseConv2dOp,
    QuantizedConv2dOp,
    QuantizedGlobalAvgPoolOp,
    QuantizedAddOp,
    QuantizedMulOp,
    QuantizedMatmulOp,
    QuantizedMaxPool2dOp,
    QuantizedResizeNearestOp,
)
from buddy.compiler.graph.type import TensorDType

from stack.compiler.frontend.quant.rax import (
    QuantTensor,
    RaxQuantPackage,
    write_payload,
)


def _quantize_operators(graph, parameter_names, arrays, weight_scales, calibration):
    def uses(node, name):
        pending = list(node.args)
        while pending:
            argument = pending.pop()
            if isinstance(argument, (list, tuple)):
                pending.extend(argument)
            elif renamed.get(str(argument), str(argument)) == name:
                return True
        return False

    param_nodes = list(graph.params)
    input_node_list = list(graph.inputs)
    original_body = list(graph._body)
    calibration_index = {}
    plans = {}
    renamed = {}
    removed = set()

    for node in original_body:
        if not isinstance(node, (AliasOp, CloneOp, AdaptiveAvgPool2dOp)) or (
            isinstance(node, AdaptiveAvgPool2dOp)
            and graph.node_table[str(node.args[0])].tensor_meta["shape"]
            != node.tensor_meta["shape"]
        ):
            continue
        if not isinstance(node.args[0], str):
            raise ValueError(f"unsupported Alias form for {node.name}")
        source = str(node.args[0])
        source_node = graph.node_table.get(source)
        if source_node is None or source_node.tensor_meta.get(
            "shape"
        ) != node.tensor_meta.get("shape"):
            raise ValueError(f"Alias shape mismatch for {node.name}")
        renamed[node.name] = renamed.get(source, source)
        removed.add(node.name)

    for index, node in enumerate(original_body):
        if not isinstance(node, (Conv2dOp, AddMMOp, MatmulOp)):
            continue
        if isinstance(node, Conv2dOp):
            if len(node.args) != 9:
                raise ValueError(f"unexpected Conv2d form for {node.name}")
            if node.args[5] != [1, 1] or node.args[6]:
                raise ValueError(f"unsupported Quantized Conv2D form for {node.name}")
            if node.args[3][0] != node.args[3][1]:
                raise ValueError(f"asymmetric Quantized Conv2D stride for {node.name}")
            padding = node.args[4]
            if len(padding) == 1:
                padding = [padding[0], padding[0]]
            if len(padding) != 2 or padding[0] != padding[1]:
                raise ValueError(f"asymmetric Quantized Conv2D padding for {node.name}")
            activation_name, weight_arg, bias_arg = node.args[:3]
            activation_name = renamed.get(str(activation_name), str(activation_name))
            weight_node = graph.node_table[str(weight_arg)]
            weight_name = parameter_names.get(weight_node.name)
            if weight_name is None:
                raise ValueError(
                    f"Quantized Conv2D weight is not a parameter for {node.name}"
                )
            weight_shape = list(arrays[weight_name].shape)
            input_channels = int(
                graph.node_table[str(activation_name)].tensor_meta["shape"][1]
            )
            groups = int(node.args[8])
            if groups == 1:
                replacement = QuantizedConv2dOp()
                reduction_size = weight_shape[1] * weight_shape[2] * weight_shape[3]
            elif (
                groups == input_channels
                and weight_shape[0] == input_channels
                and weight_shape[1] == 1
            ):
                replacement = QuantizedDepthwiseConv2dOp()
                reduction_size = weight_shape[2] * weight_shape[3]
            else:
                raise ValueError(f"unsupported grouped Conv2D for {node.name}")
        else:
            if isinstance(node, AddMMOp):
                if len(node.args) < 3:
                    raise ValueError(f"unexpected AddMM form for {node.name}")
                bias_arg, activation_name, weight_arg = node.args[:3]
            else:
                if len(node.args) != 2:
                    raise ValueError(f"unexpected Matmul form for {node.name}")
                activation_name, weight_arg = node.args
                activation_name = renamed.get(
                    str(activation_name), str(activation_name)
                )
                bias_arg = None
            weight_value = graph.node_table[str(weight_arg)]
            valid_transpose = isinstance(weight_value, TOp) or (
                isinstance(weight_value, PermuteOp)
                and len(weight_value.args) == 2
                and list(weight_value.args[1]) == [1, 0]
            )
            if not valid_transpose or len(weight_value._parents) != 1:
                raise ValueError(
                    f"Quantized MatMul weight must be one transpose for {node.name}: "
                    f"{weight_value.name}/{type(weight_value).__name__}, "
                    f"parents={weight_value._parents}, args={weight_value.args}"
                )
            weight_node = graph.node_table[weight_value._parents[0]]
            weight_arg = weight_node.name
            removed.add(weight_value.name)
            replacement = QuantizedMatmulOp()
            weight_name = parameter_names.get(weight_node.name)
            if weight_name is None:
                raise ValueError(
                    f"Quantized MatMul weight is not a parameter for {node.name}"
                )
            weight_shape = list(arrays[weight_name].shape)
            if len(weight_shape) != 2:
                raise ValueError(
                    f"Quantized MatMul weight must be rank 2 for {node.name}"
                )
            reduction_size = weight_shape[1]

        weight_name = parameter_names.get(weight_node.name)
        if weight_name is None or weight_name not in weight_scales:
            raise ValueError(f"missing quantized weight for {node.name}")
        weight_node._quantized_weight = True
        records = calibration.get(weight_name)
        occurrence = calibration_index.get(weight_name, 0)
        if records is None or occurrence >= len(records):
            raise ValueError(f"missing calibration for {weight_name}")
        calibration_index[weight_name] = occurrence + 1
        (
            calibrated_input,
            raw_output_scale,
            relu_output_scale,
            raw_channel_scales,
            hardswish_channel_scales,
        ) = records[occurrence]
        raw_channel_scales = np.asarray(raw_channel_scales, dtype=np.float32)
        hardswish_channel_scales = np.asarray(
            hardswish_channel_scales, dtype=np.float32
        )
        if (
            not np.isfinite(calibrated_input)
            or calibrated_input <= 0.0
            or not np.isfinite(raw_output_scale)
            or raw_output_scale <= 0.0
            or not np.isfinite(relu_output_scale)
            or relu_output_scale <= 0.0
            or raw_channel_scales.shape != (weight_shape[0],)
            or hardswish_channel_scales.shape != (weight_shape[0],)
            or not np.all(np.isfinite(raw_channel_scales))
            or not np.all(np.isfinite(hardswish_channel_scales))
            or np.any(raw_channel_scales <= 0.0)
            or np.any(hardswish_channel_scales <= 0.0)
        ):
            raise ValueError(f"invalid calibration scale for {node.name}")
        activation_node = graph.node_table[str(activation_name)]

        dw = np.asarray(weight_scales[weight_name], dtype=np.float32).reshape(-1)
        output_channels = weight_shape[0]
        if dw.size != output_channels:
            raise ValueError(f"weight scale count mismatch for {weight_name}")
        if not np.all(np.isfinite(dw)) or np.any(dw <= 0.0):
            raise ValueError(f"invalid weight scale for {weight_name}")
        if bias_arg is None:
            bias = np.zeros(output_channels, dtype=np.float32)
        else:
            bias_node = graph.node_table[str(bias_arg)]
            bias_name = parameter_names.get(bias_node.name)
            if bias_name is None:
                raise ValueError(
                    f"Quantized bias is not an offline parameter for {node.name}"
                )
            bias = np.asarray(arrays[bias_name], dtype=np.float32).reshape(-1)
        if bias.size != output_channels:
            raise ValueError(f"bias channel count mismatch for {node.name}")
        if not np.all(np.isfinite(bias)):
            raise ValueError(f"non-finite bias for {node.name}")
        product_bound = 128 * 128 * reduction_size
        bias_headroom = np.iinfo(np.int32).max - product_bound
        if bias_headroom <= 1:
            raise ValueError(f"INT32 accumulator cannot hold {node.name}")
        minimum_weight_scale = np.abs(bias.astype(np.float64)) / (
            (bias_headroom - 1) * float(calibrated_input)
        )
        minimum_weight_scale = np.nextafter(
            minimum_weight_scale.astype(np.float32), np.float32(np.inf)
        )
        dw = np.maximum(dw, minimum_weight_scale)
        weight_scales[weight_name] = dw
        weight_node._quantized_weight_scales = dw.copy()
        required_input_scale = float(calibrated_input)

        direct_users = [
            candidate
            for candidate in graph._body
            if candidate is not node and uses(candidate, node.name)
        ]
        relus = [
            candidate
            for candidate in graph._body
            if isinstance(candidate, (ReluOp, HardswishOp, SiluOp))
            and len(candidate.args) == 1
            and str(candidate.args[0]) == node.name
        ]
        if len(relus) > 1:
            raise ValueError(f"multiple activation users for {node.name}")
        activation = relus[0] if relus else None
        activation_nodes = [activation] if activation is not None else []
        activation_kind = (
            2
            if isinstance(activation, (HardswishOp, SiluOp))
            else 1 if isinstance(activation, ReluOp) else 0
        )
        lut_kind = (
            "silu"
            if isinstance(activation, SiluOp)
            else "hardswish" if isinstance(activation, HardswishOp) else None
        )

        silu_negs = [
            candidate for candidate in direct_users if isinstance(candidate, NegOp)
        ]
        if silu_negs:
            if activation is not None or len(silu_negs) != 1:
                raise ValueError(f"ambiguous decomposed SiLU after {node.name}")
            neg = silu_negs[0]
            neg_users = [
                candidate for candidate in original_body if uses(candidate, neg.name)
            ]
            if len(neg_users) != 1 or not isinstance(neg_users[0], ExpOp):
                raise ValueError(f"malformed decomposed SiLU exp after {node.name}")
            exp = neg_users[0]
            exp_users = [
                candidate for candidate in original_body if uses(candidate, exp.name)
            ]
            if (
                len(exp_users) != 1
                or not isinstance(exp_users[0], AddOp)
                or list(exp_users[0].args) != [exp.name, 1]
            ):
                raise ValueError(f"malformed decomposed SiLU add after {node.name}")
            add = exp_users[0]
            add_users = [
                candidate for candidate in original_body if uses(candidate, add.name)
            ]
            if (
                len(add_users) != 1
                or not isinstance(add_users[0], DivOp)
                or list(add_users[0].args) != [node.name, add.name]
                or {candidate.name for candidate in direct_users}
                != {neg.name, add_users[0].name}
            ):
                raise ValueError(f"malformed decomposed SiLU div after {node.name}")
            activation = add_users[0]
            activation_nodes = [neg, exp, add, activation]
            activation_kind = 2
            lut_kind = "silu"

        hard_swish_adds = [
            candidate
            for candidate in direct_users
            if isinstance(candidate, AddOp)
            and len(candidate.args) == 2
            and str(candidate.args[0]) == node.name
            and candidate.args[1] == 3
        ]
        if hard_swish_adds:
            if activation is not None or len(hard_swish_adds) != 1:
                raise ValueError(f"ambiguous HardSwish after {node.name}")
            add = hard_swish_adds[0]
            add_users = [
                candidate for candidate in graph._body if uses(candidate, add.name)
            ]
            if len(add_users) != 1 or not isinstance(add_users[0], ClampMinOp):
                raise ValueError(f"malformed HardSwish clamp-min after {node.name}")
            clamp_min = add_users[0]
            if list(clamp_min.args) != [add.name, 0]:
                raise ValueError(
                    f"malformed HardSwish clamp-min args after {node.name}"
                )
            clamp_min_users = [
                candidate
                for candidate in graph._body
                if uses(candidate, clamp_min.name)
            ]
            if len(clamp_min_users) != 1 or not isinstance(
                clamp_min_users[0], ClampMaxOp
            ):
                raise ValueError(f"malformed HardSwish clamp-max after {node.name}")
            clamp_max = clamp_min_users[0]
            if list(clamp_max.args) != [clamp_min.name, 6]:
                raise ValueError(
                    f"malformed HardSwish clamp-max args after {node.name}"
                )
            clamp_max_users = [
                candidate
                for candidate in graph._body
                if uses(candidate, clamp_max.name)
            ]
            if len(clamp_max_users) != 1:
                raise ValueError(f"ambiguous hard activation after {node.name}")
            after_clamp = clamp_max_users[0]
            if isinstance(after_clamp, DivOp):
                if list(after_clamp.args) != [clamp_max.name, 6]:
                    raise ValueError(f"malformed hard-sigmoid after {node.name}")
                if {candidate.name for candidate in direct_users} != {add.name}:
                    raise ValueError(
                        f"hard-sigmoid input has extra users after {node.name}"
                    )
                activation = after_clamp
                activation_nodes = [add, clamp_min, clamp_max, after_clamp]
                activation_kind = 2
                lut_kind = "hardsigmoid"
            elif isinstance(after_clamp, MulOp):
                mul = after_clamp
                if list(map(str, mul.args)) != [node.name, clamp_max.name]:
                    raise ValueError(
                        f"malformed HardSwish multiply args after {node.name}"
                    )
                if {candidate.name for candidate in direct_users} != {
                    add.name,
                    mul.name,
                }:
                    raise ValueError(
                        f"HardSwish input has extra users after {node.name}"
                    )
                mul_users = [
                    candidate for candidate in graph._body if uses(candidate, mul.name)
                ]
                if len(mul_users) != 1 or not isinstance(mul_users[0], DivOp):
                    raise ValueError(f"malformed HardSwish divide after {node.name}")
                div = mul_users[0]
                if list(div.args) != [mul.name, 6]:
                    raise ValueError(
                        f"malformed HardSwish divide args after {node.name}"
                    )
                activation = div
                activation_nodes = [add, clamp_min, clamp_max, mul, div]
                activation_kind = 2
                lut_kind = "hardswish"
            else:
                raise ValueError(f"unsupported hard activation after {node.name}")

        if (node.trace_meta is not None and node.trace_meta["tensor"]) or any(
            item.trace_meta is not None for item in activation_nodes
        ):
            activation = None
            activation_nodes = []
            activation_kind = 0
            lut_kind = None

        result_name = activation.name if activation is not None else node.name
        result_users = [
            candidate
            for candidate in graph._body
            if candidate is not activation
            and candidate is not node
            and uses(candidate, result_name)
        ]
        plans[node.name] = {
            "index": index,
            "node": node,
            "replacement": replacement,
            "activation_name": str(activation_name),
            "activation_node": activation_node,
            "weight_arg": str(weight_arg),
            "weight_shape": weight_shape,
            "dw": dw,
            "bias": bias,
            "required_input_scale": required_input_scale,
            "calibrated_output_scale": float(
                relu_output_scale if activation_kind == 1 else raw_output_scale
            ),
            "raw_channel_scales": raw_channel_scales,
            "hardswish_channel_scales": hardswish_channel_scales,
            "activation": activation,
            "activation_nodes": activation_nodes,
            "activation_kind": activation_kind,
            "lut_kind": lut_kind,
            "direct_users": direct_users,
            "result_name": result_name,
            "users": result_users,
            "padding": padding if isinstance(node, Conv2dOp) else None,
            "product_bound": product_bound,
        }

    first_conv = min(
        (plan for plan in plans.values() if isinstance(plan["node"], Conv2dOp)),
        key=lambda plan: plan["index"],
        default=None,
    )
    if first_conv is not None and first_conv["lut_kind"] == "hardswish":
        consumers = [
            candidate
            for candidate in plans.values()
            if candidate["activation_name"] == first_conv["result_name"]
        ]
        if (
            len(first_conv["users"]) != 1
            or len(consumers) != 1
            or not isinstance(consumers[0]["replacement"], QuantizedDepthwiseConv2dOp)
            or consumers[0]["weight_shape"][0] != first_conv["weight_shape"][0]
        ):
            raise ValueError(
                "the first Hardswish Conv must feed one matching depthwise Conv"
            )
        first_conv["lane_input_scales"] = first_conv["raw_channel_scales"]
        first_conv["lane_output_scales"] = first_conv["hardswish_channel_scales"]
        consumers[0]["input_channel_scales"] = first_conv["lane_output_scales"]

    for plan in plans.values():
        if plan["activation"] is not None:
            renamed[plan["result_name"]] = plan["node"].name
            removed.update(item.name for item in plan["activation_nodes"])

    special = {}
    quantized_values = {
        name
        for plan in plans.values()
        for name in (plan["node"].name, plan["result_name"])
    }
    for node in original_body:
        if not isinstance(node, GetItemOp) or len(node.args) != 2:
            continue
        source = graph.node_table.get(str(node.args[0]))
        if not isinstance(source, (SplitOp, SplitWithSizesOp)) or len(source.args) != 3:
            continue
        input_name = renamed.get(str(source.args[0]), str(source.args[0]))
        if int(source.args[2]) == 1 and input_name in quantized_values:
            quantized_values.add(node.name)
    for node in original_body:
        if node.name in removed:
            continue
        activation = None
        result_name = node.name
        arguments = None
        attributes = {}
        if isinstance(node, GetItemOp):
            if len(node.args) != 2 or not isinstance(node.args[1], int):
                continue
            source = graph.node_table[str(node.args[0])]
            index = int(node.args[1])
            if isinstance(source, (SplitOp, SplitWithSizesOp)):
                if len(source.args) != 3:
                    continue
                input_name = renamed.get(str(source.args[0]), str(source.args[0]))
                if input_name not in quantized_values:
                    continue
                if int(source.args[2]) != 1:
                    raise ValueError(
                        f"Quantized channel split must use NCHW C at {source.name}"
                    )
                if isinstance(source, SplitOp):
                    channels = int(
                        graph.node_table[str(source.args[0])].tensor_meta["shape"][1]
                    )
                    size = int(source.args[1])
                    sizes = [size] * (channels // size)
                    if channels % size:
                        sizes.append(channels % size)
                else:
                    sizes = [int(value) for value in source.args[1]]
                if index < 0 or index >= len(sizes):
                    raise ValueError(
                        f"Quantized channel split index is invalid at {node.name}"
                    )
                replacement = QuantizedChannelSliceOp()
                arguments = [input_name]
                attributes = {
                    "offset": sum(sizes[:index]),
                    "output_shape": list(node.tensor_meta["shape"]),
                }
                removed.add(source.name)
            elif isinstance(source, LowMemoryMaxPoolWithOffsetsOp):
                if index != 0 or len(source.args) != 6:
                    raise ValueError(
                        f"Quantized SPPF must select MaxPool values at {node.name}"
                    )
                kernel, stride, padding, dilation, ceil_mode = source.args[1:]
                if (
                    list(kernel) != [5, 5]
                    or list(stride) != [1, 1]
                    or list(padding) != [2, 2]
                    or list(dilation) != [1, 1]
                    or ceil_mode
                ):
                    raise ValueError(
                        f"unsupported Quantized SPPF MaxPool at {source.name}"
                    )
                input_name = renamed.get(str(source.args[0]), str(source.args[0]))
                if input_name not in quantized_values:
                    continue
                replacement = QuantizedMaxPool2dOp()
                arguments = [input_name]
                attributes = {
                    "input_shape": list(
                        graph.node_table[str(source.args[0])].tensor_meta["shape"]
                    ),
                    "output_shape": list(node.tensor_meta["shape"]),
                    "kernel": 5,
                    "stride": 1,
                    "padding": 2,
                }
                removed.add(source.name)
            else:
                continue
        elif isinstance(node, CatOp):
            if len(node.args) != 2 or int(node.args[1]) != 1:
                continue
            arguments = [renamed.get(str(value), str(value)) for value in node.args[0]]
            if len(arguments) < 2 or any(
                value not in quantized_values for value in arguments
            ):
                continue
            replacement = QuantizedChannelConcatOp()
            attributes = {"output_shape": list(node.tensor_meta["shape"])}
        elif isinstance(node, UnsafeIndexOp):
            if len(node.args) != 2:
                continue
            input_name = renamed.get(str(node.args[0]), str(node.args[0]))
            if input_name not in quantized_values:
                continue
            input_shape = list(graph.node_table[str(node.args[0])].tensor_meta["shape"])
            output_shape = list(node.tensor_meta["shape"])
            if (
                len(input_shape) != 4
                or len(output_shape) != 4
                or input_shape[0] != output_shape[0]
                or input_shape[1] != output_shape[1]
                or output_shape[2] % input_shape[2]
                or output_shape[3] % input_shape[3]
            ):
                raise ValueError(f"unsupported Quantized nearest resize at {node.name}")
            replacement = QuantizedResizeNearestOp()
            arguments = [input_name]
            attributes = {
                "output_shape": output_shape,
                "scale_h": output_shape[2] // input_shape[2],
                "scale_w": output_shape[3] // input_shape[3],
            }
        elif isinstance(node, MaxPool2dOp):
            # Pooling can sit beyond an FP32 reshape/transpose boundary.
            # Only a quantized producer supplies an INT8 scale for fusion.
            input_name = renamed.get(str(node.args[0]), str(node.args[0]))
            if input_name not in quantized_values:
                continue
            if (
                len(node.args) not in (3, 4)
                or len(node.args[1]) != 2
                or node.args[1][0] != node.args[1][1]
                or len(node.args[2]) != 2
                or node.args[2][0] != node.args[2][1]
            ):
                raise ValueError(
                    f"unsupported quantized graph MaxPool2D for {node.name}"
                )
            padding = node.args[3] if len(node.args) == 4 else [0, 0]
            if len(padding) == 1:
                padding = [padding[0], padding[0]]
            if len(padding) != 2 or padding[0] != padding[1]:
                raise ValueError(
                    f"asymmetric quantized graph MaxPool2D padding for {node.name}"
                )
            replacement = QuantizedMaxPool2dOp()
        elif isinstance(node, MeanOp):
            if (
                len(node.args) != 3
                or list(node.args[1]) != [-1, -2]
                or not node.args[2]
            ):
                continue
            replacement = QuantizedGlobalAvgPoolOp()
        elif (
            isinstance(node, MulOp)
            and len(node.args) == 2
            and all(isinstance(arg, str) for arg in node.args)
        ):
            if any(
                renamed.get(str(arg), str(arg)) not in quantized_values
                for arg in node.args
            ):
                continue
            replacement = QuantizedMulOp()
        elif (
            isinstance(node, AddOp)
            and len(node.args) == 2
            and all(isinstance(arg, str) for arg in node.args)
        ):
            if any(
                renamed.get(str(arg), str(arg)) not in quantized_values
                for arg in node.args
            ):
                continue
            replacement = QuantizedAddOp()
        else:
            continue
        if isinstance(replacement, (QuantizedAddOp, QuantizedMulOp)):
            users = [
                candidate
                for candidate in original_body
                if candidate.name not in removed and uses(candidate, node.name)
            ]
            if (node.trace_meta is not None and node.trace_meta["tensor"]) or any(
                isinstance(candidate, ReluOp) and candidate.trace_meta is not None
                for candidate in users
            ):
                continue
            if not any(
                isinstance(
                    candidate,
                    (
                        ReluOp,
                        Conv2dOp,
                        AddMMOp,
                        MatmulOp,
                        MeanOp,
                        MaxPool2dOp,
                        AddOp,
                        MulOp,
                        CatOp,
                    ),
                )
                for candidate in users
            ):
                continue
            relus = [candidate for candidate in users if isinstance(candidate, ReluOp)]
            if relus:
                if len(users) != 1 or len(relus) != 1:
                    raise ValueError(
                        f"Quantized INT8 elementwise ReLU must be its only user: {node.name}"
                    )
                activation = relus[0]
                result_name = activation.name
                renamed[result_name] = node.name
                removed.add(result_name)
        special[node.name] = {
            "node": node,
            "replacement": replacement,
            "activation": activation,
            "result_name": result_name,
            "arguments": arguments,
            **attributes,
        }
        quantized_values.add(node.name)
        quantized_values.add(result_name)

    for plan in plans.values():
        consumers = [
            candidate
            for candidate in plans.values()
            if candidate["activation_name"] == plan["result_name"]
        ]
        plan["output_scale"] = (
            max(candidate["required_input_scale"] for candidate in consumers)
            if consumers
            else plan["calibrated_output_scale"]
        )

    for item in reversed(list(special.values())):
        node = item["node"]
        consumers = [
            candidate
            for candidate in plans.values()
            if renamed.get(candidate["activation_name"], candidate["activation_name"])
            == node.name
        ]
        downstream_scales = [
            candidate["output_scale"]
            for candidate in special.values()
            if "output_scale" in candidate
            and any(
                uses(candidate["node"], source)
                for source in (node.name, item["result_name"])
            )
        ]
        if not consumers and isinstance(node, (MeanOp, MaxPool2dOp)):
            users = [
                candidate
                for candidate in original_body
                if candidate.name not in removed and uses(candidate, node.name)
            ]
            if len(users) != 1 or not isinstance(users[0], (ViewOp, ReshapeOp)):
                raise ValueError(
                    f"quantized graph mean has no unique consumer: {node.name}"
                )
            consumers = [
                candidate
                for candidate in plans.values()
                if renamed.get(
                    candidate["activation_name"], candidate["activation_name"]
                )
                == users[0].name
            ]
        if not consumers and not downstream_scales:
            users = [
                candidate.name
                for candidate in original_body
                if uses(candidate, node.name)
            ]
            raise ValueError(
                f"quantized graph stage has no compute consumer: {node.name}, "
                f"args={node.args}, users={users}"
            )
        item["output_scale"] = max(
            [consumer["required_input_scale"] for consumer in consumers]
            + downstream_scales
        )

    for plan in plans.values():
        max_pool_consumers = [
            item
            for item in special.values()
            if isinstance(item["replacement"], QuantizedMaxPool2dOp)
            and renamed.get(str(item["node"].args[0]), str(item["node"].args[0]))
            == plan["node"].name
        ]
        if len(max_pool_consumers) > 1:
            raise ValueError(f"multiple MaxPool2D consumers for {plan['node'].name}")
        if max_pool_consumers:
            plan["output_scale"] = max_pool_consumers[0]["output_scale"]

    changed = True
    while changed:
        changed = False
        for name, item in special.items():
            if not isinstance(
                item["replacement"],
                (
                    QuantizedChannelSliceOp,
                    QuantizedChannelConcatOp,
                    QuantizedResizeNearestOp,
                    QuantizedMaxPool2dOp,
                ),
            ):
                continue
            sources = item["arguments"]
            if sources is None:
                sources = [
                    renamed.get(str(item["node"].args[0]), str(item["node"].args[0]))
                ]
            scales = [item["output_scale"]]
            for source in sources:
                if source in plans:
                    scales.append(plans[source]["output_scale"])
                elif source in special:
                    scales.append(special[source]["output_scale"])
                else:
                    raise ValueError(
                        f"Quantized view source has no scale at {name}: {source}"
                    )
            scale = max(scales)
            if item["output_scale"] != scale:
                item["output_scale"] = scale
                changed = True
            for source in sources:
                producer = plans[source] if source in plans else special[source]
                if producer["output_scale"] != scale:
                    producer["output_scale"] = scale
                    changed = True

    value_scale = {}
    for plan in plans.values():
        value_scale[plan["node"].name] = plan["output_scale"]
        value_scale[plan["result_name"]] = plan["output_scale"]
    for name, item in special.items():
        value_scale[name] = item["output_scale"]
        value_scale[item["result_name"]] = item["output_scale"]

    for plan in plans.values():
        activation_name = renamed.get(plan["activation_name"], plan["activation_name"])
        input_scale = value_scale.get(activation_name, plan["required_input_scale"])
        dw = plan["dw"]
        input_channel_scales = plan.get("input_channel_scales")
        if input_channel_scales is not None:
            input_channel_scales = np.asarray(input_channel_scales, dtype=np.float32)
            if not isinstance(
                plan["replacement"], QuantizedDepthwiseConv2dOp
            ) or input_channel_scales.shape != (dw.size,):
                raise ValueError(
                    f"per-channel input scale requires matching depthwise Conv: "
                    f"{plan['node'].name}"
                )
            minimum_weight_scale = np.abs(plan["bias"].astype(np.float64)) / (
                (np.iinfo(np.int32).max - plan["product_bound"] - 1)
                * input_channel_scales.astype(np.float64)
            )
            dw = np.maximum(
                dw,
                np.nextafter(
                    minimum_weight_scale.astype(np.float32), np.float32(np.inf)
                ),
            )
            plan["dw"] = dw
            graph.node_table[plan["weight_arg"]]._quantized_weight_scales = dw.copy()
        accumulator_input_scale = (
            input_channel_scales
            if input_channel_scales is not None
            else np.float32(input_scale)
        )
        bias_i64 = np.rint(
            plan["bias"].astype(np.float64)
            / (
                np.asarray(accumulator_input_scale, dtype=np.float64)
                * dw.astype(np.float64)
            )
        ).astype(np.int64)
        overflow = np.flatnonzero(
            np.abs(bias_i64) + plan["product_bound"] > np.iinfo(np.int32).max
        )
        if overflow.size:
            channel = int(overflow[0])
            scale_index = channel if input_channel_scales is not None else 0
            raise ValueError(
                f"INT32 accumulator overflow for {plan['node'].name} channel {channel}: "
                f"input_scale={float(np.asarray(accumulator_input_scale).reshape(-1)[scale_index])}, "
                f"weight_scale={float(dw[channel])}, "
                f"bias_i32={int(bias_i64[channel])}, "
                f"product_bound={plan['product_bound']}"
            )

        node = plan["node"]
        replacement = plan["replacement"]
        activation = plan["activation"]
        replacement._name = node.name
        replacement._arguments = [activation_name, plan["weight_arg"]]
        replacement._parents = list(replacement._arguments)
        replacement._children = [
            renamed.get(user.name, user.name) for user in plan["users"]
        ]
        replacement._tensor_meta = node._tensor_meta.copy()
        direct_compute_consumers = [
            candidate
            for candidate in plans.values()
            if candidate["activation_name"] == plan["result_name"]
        ]
        replacement._final_output = (
            not isinstance(node, Conv2dOp) and not direct_compute_consumers
        )
        replacement._tensor_meta["dtype"] = (
            TensorDType.Float32 if replacement._final_output else TensorDType.Int8
        )
        replacement._input_scale = input_scale
        replacement._output_scale = plan["output_scale"]
        replacement._bias_i32 = bias_i64.astype(np.int32).tolist()
        requant_output_scale = np.asarray(
            plan.get("lane_input_scales", plan["output_scale"]),
            dtype=np.float32,
        )
        replacement._requant_scale = (
            np.asarray(accumulator_input_scale, dtype=np.float32)
            * dw
            / requant_output_scale
        ).tolist()
        replacement._dequant_scale = (
            np.asarray(accumulator_input_scale, dtype=np.float32) * dw
        ).tolist()
        replacement._activation = plan["activation_kind"]
        if replacement._activation == 2:
            raw = np.arange(256, dtype=np.int16)
            signed = np.where(raw < 128, raw, raw - 256).astype(np.float32)
            if "lane_input_scales" in plan:
                x = plan["lane_input_scales"][:, None] * signed[None, :]
            else:
                x = signed * np.float32(plan["output_scale"])
            if plan["lut_kind"] == "hardswish":
                y = x * np.clip(x + np.float32(3.0), 0.0, 6.0) / np.float32(6.0)
            elif plan["lut_kind"] == "hardsigmoid":
                y = np.clip(x + np.float32(3.0), 0.0, 6.0) / np.float32(6.0)
            elif plan["lut_kind"] == "silu":
                y = x / (
                    np.float32(1.0)
                    + np.exp(-np.clip(x, np.float32(-80.0), np.float32(80.0)))
                )
            else:
                raise ValueError(f"missing LUT function for {node.name}")
            lut_output_scale = np.asarray(
                plan.get("lane_output_scales", plan["output_scale"]),
                dtype=np.float32,
            )
            if lut_output_scale.ndim == 1:
                lut_output_scale = lut_output_scale[:, None]
            replacement._lut_i8 = (
                np.clip(np.rint(y / lut_output_scale), -128, 127)
                .astype(np.int8)
                .reshape(-1)
                .tolist()
            )
            replacement._lane_output_scales = plan.get(
                "lane_output_scales", np.empty(0, dtype=np.float32)
            ).tolist()
        else:
            replacement._lut_i8 = [0]
            replacement._lane_output_scales = []
        replacement.trace_meta = node.trace_meta
        if isinstance(replacement, (QuantizedConv2dOp, QuantizedDepthwiseConv2dOp)):
            replacement._input_shape = list(
                plan["activation_node"].tensor_meta["shape"]
            )
            replacement._weight_shape = plan["weight_shape"]
            replacement._output_shape = list(node.tensor_meta["shape"])
            replacement._stride = int(node.args[3][0])
            replacement._padding = int(plan["padding"][0])
    for name, item in special.items():
        node = item["node"]
        replacement = item["replacement"]
        args = item["arguments"]
        if args is None:
            args = [
                renamed.get(str(arg), str(arg))
                for arg in node.args
                if isinstance(arg, str)
            ]
        if isinstance(replacement, QuantizedMaxPool2dOp):
            if len(args) != 1 or args[0] not in value_scale:
                raise ValueError(f"invalid Quantized MaxPool2D input for {name}")
            replacement._input_scale = value_scale[args[0]]
            replacement._output_scale = replacement._input_scale
            if "input_shape" in item:
                replacement._input_shape = item["input_shape"]
                replacement._output_shape = item["output_shape"]
                replacement._kernel = item["kernel"]
                replacement._stride = item["stride"]
                replacement._padding = item["padding"]
            else:
                replacement._input_shape = list(
                    graph.node_table[str(node.args[0])].tensor_meta["shape"]
                )
                replacement._output_shape = list(node.tensor_meta["shape"])
                replacement._kernel = int(node.args[1][0])
                replacement._stride = int(node.args[2][0])
                padding = node.args[3] if len(node.args) == 4 else [0, 0]
                replacement._padding = int(padding[0])
            replacement._final_output = False
        elif isinstance(replacement, QuantizedChannelSliceOp):
            if len(args) != 1 or args[0] not in value_scale:
                raise ValueError(f"invalid Quantized channel-slice input for {name}")
            replacement._offset = item["offset"]
            replacement._output_shape = item["output_shape"]
        elif isinstance(replacement, QuantizedChannelConcatOp):
            if any(arg not in value_scale for arg in args):
                raise ValueError(f"invalid Quantized channel-concat input for {name}")
            replacement._output_shape = item["output_shape"]
        elif isinstance(replacement, QuantizedResizeNearestOp):
            if len(args) != 1 or args[0] not in value_scale:
                raise ValueError(f"invalid Quantized resize input for {name}")
            replacement._output_shape = item["output_shape"]
            replacement._scale_h = item["scale_h"]
            replacement._scale_w = item["scale_w"]
        elif isinstance(replacement, QuantizedGlobalAvgPoolOp):
            if len(args) != 1 or args[0] not in value_scale:
                raise ValueError(f"invalid Quantized global-average input for {name}")
            replacement._input_scale = value_scale[args[0]]
        else:
            if len(args) != 2 or any(arg not in value_scale for arg in args):
                raise ValueError(f"invalid Quantized INT8 elementwise input for {name}")
            replacement._lhs_scale = value_scale[args[0]]
            replacement._rhs_scale = value_scale[args[1]]
            replacement._activation = 1 if item["activation"] is not None else 0
        replacement._name = name
        replacement._arguments = args
        replacement._parents = list(args)
        replacement._children = [
            renamed.get(child, child)
            for child in (
                item["activation"]._children
                if item["activation"] is not None
                else node._children
            )
        ]
        replacement._tensor_meta = node._tensor_meta.copy()
        replacement._tensor_meta["dtype"] = TensorDType.Int8
        replacement._output_scale = item["output_scale"]
        replacement.trace_meta = node.trace_meta

    replacements = {name: plan["replacement"] for name, plan in plans.items()} | {
        name: item["replacement"] for name, item in special.items()
    }

    def canonical(argument):
        if isinstance(argument, list):
            return [canonical(value) for value in argument]
        if isinstance(argument, tuple):
            return tuple(canonical(value) for value in argument)
        while isinstance(argument, str) and argument in renamed:
            argument = renamed[argument]
        return argument

    graph._body = [
        replacements.get(node.name, node)
        for node in original_body
        if node.name not in removed
    ]
    graph.node_table = {node.name: node for node in graph.body}
    for node in graph.body:
        node._arguments = canonical(node.args)
        node._parents = []
        node._children = []
    for node in graph.body:
        pending = list(node.args)
        while pending:
            argument = pending.pop(0)
            if isinstance(argument, (list, tuple)):
                pending[:0] = argument
            elif isinstance(argument, str) and argument in graph.node_table:
                if argument not in node._parents:
                    node._parents.append(argument)
                    graph.node_table[argument]._children.append(node.name)
    for node in replacements.values():
        if isinstance(
            node, (QuantizedConv2dOp, QuantizedDepthwiseConv2dOp, QuantizedMatmulOp)
        ):
            node._final_output = any(name not in replacements for name in node.children)
            node._tensor_meta["dtype"] = (
                TensorDType.Float32 if node._final_output else TensorDType.Int8
            )
        elif isinstance(node, QuantizedMaxPool2dOp):
            node._final_output = any(name not in replacements for name in node.children)
    graph._fake_params = [graph.body.index(node) for node in param_nodes]
    graph._inputs = [graph.body.index(node) for node in input_node_list]
    from buddy.compiler.graph.transform.quantization.quantize import sort_graph

    sort_graph(graph)


def quantize_model_graph(
    graph,
    params,
    names: list[str],
    output_dir: Path,
    model_name: str,
    calibration: dict,
    *,
    weight_scales=None,
    assets: dict[str, bytes] | None = None,
) -> None:
    if len(graph.params) != len(params) or len(names) != len(params):
        raise ValueError("parameter metadata does not match imported graph")
    param_nodes = list(graph.params)
    parameter_names, arrays = {}, {}
    for node, param, name in zip(param_nodes, params, names):
        array = param.detach().cpu().numpy().astype(np.float32)
        parameter_names[node.name] = name
        arrays[name] = array

    if weight_scales is None:
        from stack.compiler.frontend.quant.transforms.symmetric import (
            quantize_symmetric,
        )

        weight_scales = {
            name: quantize_symmetric(array, [0])[1]
            for name, array in arrays.items()
            if name.endswith(".weight") and array.ndim >= 2
        }

    _quantize_operators(
        graph,
        parameter_names,
        arrays,
        weight_scales,
        calibration,
    )

    tensors = []
    weights, fp_params, scales = [], [], []
    weight_off = param_off = scale_off = 0
    for index, (node, name) in enumerate(zip(param_nodes, names)):
        array = arrays[name]
        if getattr(node, "_quantized_weight", False):
            axes = [0]
            dw = np.asarray(node._quantized_weight_scales, dtype=np.float32).reshape(-1)
            if (
                dw.size != array.shape[0]
                or not np.all(np.isfinite(dw))
                or np.any(dw <= 0)
            ):
                raise ValueError(f"invalid adjusted weight scales for {name}")
            q = np.clip(
                np.rint(array / dw.reshape((-1,) + (1,) * (array.ndim - 1))),
                -128,
                127,
            ).astype(np.int8)
            node.tensor_meta["dtype"] = TensorDType.Int8
            node.tensor_meta["shape"] = list(q.shape)
            params[index] = torch.from_numpy(q.copy())
            scale_bytes = dw.tobytes()
            tensors.append(
                QuantTensor(
                    name,
                    list(array.shape),
                    list(q.shape),
                    "i8",
                    axes,
                    weight_off,
                    q.nbytes,
                    scale_off,
                    len(scale_bytes),
                )
            )
            weights.append(q.tobytes())
            scales.append(scale_bytes)
            weight_off += q.nbytes
            scale_off += len(scale_bytes)
        else:
            raw = array.tobytes()
            tensors.append(
                QuantTensor(
                    name,
                    list(array.shape),
                    list(array.shape),
                    "f32",
                    [],
                    param_off,
                    len(raw),
                    0,
                    0,
                )
            )
            fp_params.append(raw)
            param_off += len(raw)
    package = RaxQuantPackage(
        tensors,
        b"".join(weights),
        b"".join(fp_params),
        b"".join(scales),
        assets if assets is not None else {},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_payload(package, output_dir / f"{model_name}.payload")
    (output_dir / "weights.bin").write_bytes(package.weights)
    (output_dir / "params.f32").write_bytes(package.params_f32)
    (output_dir / "scales.bin").write_bytes(package.scales_f32)
