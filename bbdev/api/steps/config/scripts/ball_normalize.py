"""Validate Ball instances and their directly owned instructions."""

from copy import deepcopy
from typing import Any


def normalize_ball_domain(thread: dict[str, Any]) -> dict[str, Any] | None:
    raw = thread.get("balldomain")
    if raw is None:
        return None
    domain = deepcopy(raw)
    balls = domain["balls"]
    ids, names, codes = set(), set(), set()
    for ball in balls:
        identity, name = ball["id"], ball["name"]
        if type(identity) is not int or not 0 <= identity < 32 or identity in ids:
            raise ValueError(f"Invalid or duplicate Ball ID: {identity}")
        if name in names:
            raise ValueError(f"Duplicate Ball instance name: {name}")
        ids.add(identity)
        names.add(name)
        if not ball["component"].isidentifier():
            raise ValueError(f"Invalid Ball component: {ball['component']}")
        port = ball["port"]
        if any(
            type(port[key]) is not int or port[key] <= 0 for key in ("inBW", "outBW")
        ):
            raise ValueError(
                "Ball input and output bandwidths must be positive integers"
            )
        if any(
            type(port[key]) is not int or port[key] < 0
            for key in ("mmioReadBW", "mmioWriteBW")
        ):
            raise ValueError("Ball MMIO bandwidths must be non-negative integers")
        mnemonics = set()
        for instruction in ball["instructions"]:
            code, mnemonic = instruction["funct7"], instruction["mnemonic"]
            if type(code) is not int or not 0 <= code < 128 or code in codes:
                raise ValueError(f"Invalid or duplicate instruction encoding: {code}")
            if mnemonic in mnemonics or code == 15 or mnemonic == "RUN_KERNEL":
                raise ValueError(f"Invalid or duplicate Ball instruction: {mnemonic}")
            codes.add(code)
            mnemonics.add(mnemonic)
    return domain
