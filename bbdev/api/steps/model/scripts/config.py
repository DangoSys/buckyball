"""Compile model-owned TOML parameters to protobuf and ordinary build arguments."""
import ast
import json
import math
import operator
import re
import tomllib
from pathlib import Path

from steps.config.scripts.model_pb2 import ModelConfig


_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


def compile_config(source: Path, destination: Path, name: str) -> None:
    config = tomllib.loads(source.read_text())
    unknown = config.keys() - {"parameters", "derived"}
    if unknown:
        raise ValueError(f"Unknown model config sections: {sorted(unknown)}")
    values = dict(config["parameters"])
    derived = config.get("derived", {})
    if values.keys() & derived.keys():
        raise ValueError("A model parameter cannot be both supplied and derived")
    visiting = set()

    def resolve(key):
        if key in values:
            return values[key]
        if key in visiting:
            raise ValueError(f"Cyclic model parameter: {key}")
        visiting.add(key)
        value = evaluate(ast.parse(derived[key], mode="eval").body)
        visiting.remove(key)
        values[key] = value
        return value

    def evaluate(node):
        if isinstance(node, ast.Name):
            return resolve(node.id)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if type(left) not in (int, float) or type(right) not in (int, float):
                raise ValueError("Derived arithmetic requires numeric parameters")
            return _OPERATORS[type(node.op)](left, right)
        raise ValueError("Model expressions support names, numbers, +, -, *, // and % only")

    for key in derived:
        resolve(key)
    message = ModelConfig(version=1, name=name)
    fields = {int: "integer", float: "real", str: "text", bool: "boolean"}
    for key, value in values.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError(f"Invalid model parameter name: {key}")
        if type(value) not in fields or isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Invalid model parameter value: {key}")
        setattr(message.parameters[key], fields[type(value)], value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(message.SerializeToString(deterministic=True))
    destination.with_suffix(".json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


def cmake_arguments(path: Path, name: str) -> list[str]:
    message = ModelConfig.FromString(path.read_bytes())
    if message.version != 1 or message.name != name:
        raise ValueError(f"Invalid model config version or model name: {path}")
    arguments = []
    for key, parameter in sorted(message.parameters.items()):
        field = parameter.WhichOneof("value")
        if field is None:
            raise ValueError(f"Model parameter has no value: {key}")
        value = getattr(parameter, field)
        if field == "boolean":
            arguments.append(f"-D{key.upper()}:BOOL={'ON' if value else 'OFF'}")
        else:
            arguments.append(f"-D{key.upper()}:STRING={value}")
    return arguments
