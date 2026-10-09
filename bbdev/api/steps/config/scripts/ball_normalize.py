"""Validate and copy external Ball mappings."""

from copy import deepcopy
from typing import Any


def normalize_ball_domain(core: dict[str, Any]) -> dict[str, Any] | None:
    raw = core.get("balldomain")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("Ball domain must be a table")
    domain = deepcopy(raw)
    mappings = domain.get("ballIdMappings")
    isa = domain.get("ballISA")
    if not isinstance(mappings, list) or not isinstance(isa, list):
        raise ValueError("Ball domain requires ballIdMappings and ballISA lists")
    if any(not isinstance(entry, dict) for entry in isa):
        raise ValueError("Ball ISA entries must be tables")
    if type(domain.get("ballNum")) is not int or domain["ballNum"] != len(mappings):
        raise ValueError("ballNum must equal the number of mappings")
    for index, mapping in enumerate(mappings):
        if not isinstance(mapping, dict) or type(mapping.get("ballId")) is not int or mapping["ballId"] != index:
            raise ValueError("Ball IDs must enumerate mappings without gaps or reordering")
        if "builtin" in mapping:
            raise ValueError("Ball mappings must select external Balls")
        klass = mapping.get("ballClass")
        config = mapping.get("config")
        if not isinstance(klass, str) or not klass.startswith("examples.balls."):
            raise ValueError("external Ball class must be under examples.balls")
        if not isinstance(config, dict) or not isinstance(config.get("_file"), str) or not isinstance(config.get("ball"), dict):
            raise ValueError("external Ball requires a resolved config with [ball] parameters")
        if any(type(mapping.get(key)) is not int or mapping[key] <= 0 for key in ("inBW", "outBW")):
            raise ValueError("external Ball inBW/outBW must be positive integers")
    if any(entry.get("funct7") == 15 or entry.get("mnemonic") == "RUN_KERNEL" for entry in isa):
        raise ValueError("RUN_KERNEL is a global control instruction, not a Ball instruction")
    return domain
