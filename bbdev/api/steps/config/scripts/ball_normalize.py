"""Validate Ball instances and their directly owned instructions."""

from copy import deepcopy
from typing import Any

# funct7 codes the framework decodes before the Ball domain (arch DISA.scala):
# barrier, the memory domain's move/set instructions, RUN_KERNEL and RUN_OPERATOR.
framework_funct7 = {
    1: "BARRIER",
    13: "MVOVER",
    15: "RUN_KERNEL",
    16: "MVOUT",
    18: "MVOUT_STRIDED",
    19: "MVOUT_BLOCKS",
    32: "MSET",
    33: "MVIN",
    34: "MVIN_2D",
    35: "MVIN_MMIO",
    36: "MVIN_STRIDED",
    37: "MVIN_BLOCKS",
    44: "MVIN_OPERATOR",
    79: "RUN_OPERATOR",
}


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
            if type(code) is not int or not 0 < code < 128 or code in codes:
                raise ValueError(f"Invalid or duplicate instruction encoding: {code}")
            if code in framework_funct7:
                raise ValueError(
                    f"Ball {name} instruction {mnemonic} uses funct7 {code}, "
                    f"reserved for {framework_funct7[code]}"
                )
            if mnemonic in mnemonics or mnemonic in framework_funct7.values():
                raise ValueError(f"Invalid or duplicate Ball instruction: {mnemonic}")
            codes.add(code)
            mnemonics.add(mnemonic)
    return domain
