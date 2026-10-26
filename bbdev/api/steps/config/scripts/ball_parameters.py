"""Typed parameter messages owned by individual Ball components."""

from importlib import import_module
from google.protobuf.json_format import ParseDict


def pack_parameters(instance, values):
    message = import_module(f"{instance.ball_dir}_params_pb2").Params()
    ParseDict(values, message)
    missing = [
        field.name
        for field in message.DESCRIPTOR.fields
        if not message.HasField(field.name)
    ]
    if missing:
        raise ValueError(f"{instance.ball_name}: missing parameters {missing}")
    instance.parameters.Pack(message, deterministic=True)


def unpack_parameters(instance):
    message = import_module(f"{instance.ball_dir}_params_pb2").Params()
    if not instance.parameters.Unpack(message):
        raise ValueError(f"{instance.ball_name}: wrong parameter message type")
    missing = [
        field.name
        for field in message.DESCRIPTOR.fields
        if not message.HasField(field.name)
    ]
    if missing:
        raise ValueError(f"{instance.ball_name}: missing parameters {missing}")
    return message
