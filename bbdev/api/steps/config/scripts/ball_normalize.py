"""Normalize external Ball mappings and the explicitly enabled RVV kernel."""

from copy import deepcopy
from typing import Any


def normalize_ball_domain(core: dict[str, Any]) -> dict[str, Any] | None:
    rvv = core.get("rvv")
    enabled = False
    if rvv is not None:
        if not isinstance(rvv, dict) or ("enable" in rvv and type(rvv["enable"]) is not bool):
            raise ValueError("rvv.enable must be an explicit boolean")
        enabled = rvv.get("enable", False)
    raw = core.get("balldomain")
    if raw is None and not enabled:
        return None
    if not isinstance(raw, dict):
        raise ValueError("RVV kernel requires a Ball domain")
    domain = deepcopy(raw)
    mappings = domain.get("ballIdMappings")
    isa = domain.get("ballISA")
    if not isinstance(mappings, list) or not isinstance(isa, list):
        raise ValueError("Ball domain requires ballIdMappings and ballISA lists")
    if any(not isinstance(entry, dict) for entry in isa):
        raise ValueError("Ball ISA entries must be tables")
    if type(domain.get("ballNum")) is not int or domain["ballNum"] != len(mappings):
        raise ValueError("ballNum must equal the number of mappings")
    kernel = None
    for index, mapping in enumerate(mappings):
        if not isinstance(mapping, dict) or type(mapping.get("ballId")) is not int or mapping["ballId"] != index:
            raise ValueError("Ball IDs must enumerate mappings without gaps or reordering")
        if "builtin" in mapping:
            if mapping["builtin"] != "kernel":
                raise ValueError("unknown builtin Ball")
            if kernel is not None or index != len(mappings) - 1:
                raise ValueError("builtin kernel must occur once at the final Ball ID")
            kernel = mapping
            continue
        klass = mapping.get("ballClass")
        config = mapping.get("config")
        if not isinstance(klass, str) or not klass.startswith("examples.balls."):
            raise ValueError("external Ball class must be under examples.balls")
        if not isinstance(config, dict) or not isinstance(config.get("_file"), str) or not isinstance(config.get("ball"), dict):
            raise ValueError("external Ball requires a resolved config with [ball] parameters")
        if any(type(mapping.get(key)) is not int or mapping[key] <= 0 for key in ("inBW", "outBW")):
            raise ValueError("external Ball inBW/outBW must be positive integers")
    if not enabled:
        if kernel is not None:
            raise ValueError("builtin kernel requires rvv.enable=true")
        return domain
    params = {}
    for key in ("laneNumber", "vLen", "eLen", "iBufWords", "memoryPorts"):
        value = rvv.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"rvv.{key} must be a positive integer")
        params[key] = str(value)
    bid = len(mappings) if kernel is None else len(mappings) - 1
    expected = {"ballId": bid, "ballName": "kernel", "ballClass": "framework.balldomain.kernel.KernelBall",
                "builtin": "kernel", "inBW": 0, "outBW": 0, "ball_params": params}
    instruction = {"mnemonic": "RUN_KERNEL", "funct7": 15, "bid": bid}
    conflicts = [entry for entry in isa if entry.get("funct7") == 15 or entry.get("mnemonic") == "RUN_KERNEL"]
    if kernel is not None:
        if kernel != expected or conflicts != [instruction]:
            raise ValueError("builtin kernel mapping or RUN_KERNEL ISA differs from RVV configuration")
    else:
        if conflicts:
            raise ValueError("RUN_KERNEL/funct7=15 conflicts with an existing ISA entry")
        mappings.append(expected)
        isa.append(instruction)
        domain["ballNum"] += 1
    return domain
