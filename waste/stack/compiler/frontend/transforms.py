# ===- graph.py ----------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the graph level of the Buddy Compiler frontend.
#
# ===---------------------------------------------------------------------------

import torch
from pathlib import Path

from buddy.compiler.graph.operation import *
from buddy.compiler.graph.graph import NodeType, _replace_node_name
from buddy.compiler.graph.type import TensorDType
from stack.compiler.frontend.operation import *


def quantize(
    graph,
    config,
    *,
    parameter_names=None,
    weights=None,
    activation_scales=None,
    calibration_model=None,
    calibration_dataset=None,
):
    """Apply the weight and activation policies from a quantization TOML.

    Offline artifacts supply data; the configuration selects transformations
    and whether each activation scale is calibrated or computed online.
    """
    from stack.compiler.frontend.quant.transforms.activation_quantize import (
        quantize_activations,
    )

    if "weights" in config:
        if parameter_names is None:
            raise ValueError("Weight quantization requires parameter names")
        if weights is None:
            from stack.compiler.frontend.quant.offline.weights import quantize_weights

            parameters = (
                dict(calibration_model.state_dict())
                if calibration_model is not None
                else {}
            )
            for node, value in zip(graph.params, graph._params_ref, strict=True):
                if "weight_permutation" in node.tensor_meta:
                    permutation = node.tensor_meta["weight_permutation"]
                    inverse = [permutation.index(axis) for axis in range(value.ndim)]
                    value = value.permute(inverse).contiguous()
                parameters[parameter_names[node.name]] = value
            weights = quantize_weights(parameters, config["weights"])
        if set(config["weights"]) != set(weights["axes"]):
            raise ValueError("Offline weights must match the configured weight names")
        if "calibration" in config and activation_scales is None:
            from stack.compiler.frontend.quant.offline.activations import (
                calibrate_activations,
            )

            calibration = dict(config["calibration"])
            if config.get("execution") == "int8":
                calibration["layers"] = {
                    name: policy["module"]
                    for name, policy in config["activations"].items()
                }
                calibration["granularities"] = {
                    name: policy["granularity"]
                    for name, policy in config["activations"].items()
                }
            activation_scales = calibrate_activations(
                calibration_model,
                weights,
                calibration_dataset,
                **calibration,
            )
        bind_quantized_weights(graph, weights, parameter_names)
    if "activations" in config:
        quantize_activations(
            graph,
            config["activations"],
            activation_scales,
            config.get("activation_target", "native"),
        )
    if "execution" in config:
        if config["execution"] != "int8":
            raise ValueError("Unsupported quantized execution policy")
        from stack.compiler.frontend.ops.reductions import (
            bmm,
            gelu,
            layer_norm,
            mean,
            softmax,
            tanh,
        )
        from stack.compiler.frontend.quant.transforms.integerize import integerize

        integerize(graph, config["activations"])
        graph._ops_registry = {
            **graph._ops_registry,
            "MeanOp": mean,
            "BatchMatmulOp": bmm,
            "SoftmaxOp": softmax,
            "NativeLayerNormOp": layer_norm,
            "GeluOp": gelu,
            "TanhOp": tanh,
        }
    return graph._params_ref


def bind_quantized_weights(graph, quantized, parameter_names):
    """Bind offline INT8 weights and insert runtime weight dequantization.

    parameter_names maps graph parameter names to checkpoint names.
    Update the imported parameter list in place to match graph.params.
    """
    from buddy.compiler.graph.transform.quantization.quantize import sort_graph

    parameters = list(graph.params)
    values = {}
    for node in parameters:
        name = parameter_names[node.name]
        value = quantized["tensors"][name]
        if "weight_permutation" in node.tensor_meta:
            value = value.permute(node.tensor_meta["weight_permutation"]).contiguous()
        values[node.name] = value
        if tuple(value.shape) != tuple(node.tensor_meta["shape"]):
            raise ValueError(f"{name}: offline weight shape differs from graph")
        if name not in quantized["axes"]:
            continue
        if value.dtype != torch.int8:
            raise ValueError(f"{name}: expected offline INT8 weights")
        node.tensor_meta["dtype"] = TensorDType.Int8
        shape = [1] * value.ndim
        axis = quantized["axes"][name]
        if axis is not None:
            if "weight_permutation" in node.tensor_meta:
                axis = node.tensor_meta["weight_permutation"].index(axis)
            shape[axis] = value.shape[axis]

        scale = PlaceholderOp()
        scale.name = f"scaler_{node.name}"
        scale.tensor_meta = {"shape": shape, "dtype": TensorDType.Float32}
        graph.add_node(scale, NodeType.FakeNode)
        values[scale.name] = quantized["scales"][name].reshape(shape)

        cast = ConvertElementTypeOp()
        cast.name = f"cast_{node.name}"
        cast.add_argument(node.name)
        cast.add_parent(node.name)
        cast.tensor_meta = {
            "shape": value.shape,
            "dtype": TensorDType.Float32,
        }
        graph.add_node(cast)

        dequant = MulOp()
        dequant.name = f"dequantized_{node.name}"
        dequant._arguments = [cast.name, scale.name]
        dequant._parents = [cast.name, scale.name]
        dequant.tensor_meta = dict(cast.tensor_meta)
        dequant._children = list(node._children)
        graph.add_node(dequant)
        for child_name in dequant._children:
            child = graph.node_table[child_name]
            child._arguments = _replace_node_name(
                child.args, node.name, dequant.name, graph.node_table
            )
            child._keyword_arguments = _replace_node_name(
                child.kwargs, node.name, dequant.name, graph.node_table
            )
            child._parents = [
                dequant.name if p == node.name else p for p in child.parents
            ]
        node._children = [cast.name]
        cast._children = [dequant.name]
        scale._children = [dequant.name]

    sort_graph(graph)
    graph._params_ref[:] = [values[node.name] for node in graph.params]
    return graph._params_ref
