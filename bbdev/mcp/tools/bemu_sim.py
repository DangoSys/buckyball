"""MCP tool: bbdev_bemu_sim."""

from __future__ import annotations
from typing import Any, Dict
from common import submit, err, fmt


def register(mcp):

    @mcp.tool()
    def bbdev_bemu_sim(
        model: str | None = None,
        bench: str | None = None,
        os: str | None = None,
        test: str | None = None,
        dtb: str | None = None,
        initrd: str | None = None,
        memory_mib: int | None = None,
        disasm: bool = False,
        tool_profile: bool = False,
        itrace: bool = False,
        mtrace: bool = False,
        ctrace: bool = False,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Run on BEMU a model or a benchmark (model or bench, with os
        "baremetal" or "linux": its <design>-<name>-<os>) or a test by its whole name (test, e.g. the Linux
        regression image <design>-test-linux). POST /bebop/bemu/sim."""
        if sum(x is not None for x in (model, bench, test)) != 1:
            return err("give exactly one of model (with os), bench (with os) or test")
        params: Dict[str, Any] = {
            "disasm": disasm,
            "tool-profile": tool_profile,
            "itrace": itrace,
            "mtrace": mtrace,
            "ctrace": ctrace,
        }
        for key, value in (
            ("model", model),
            ("bench", bench),
            ("os", os),
            ("test", test),
            ("dtb", dtb),
            ("initrd", initrd),
            ("memory-mib", memory_mib),
        ):
            if value is not None:
                params[key] = value
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if hart is not None:
            params["hart"] = hart
        return fmt(submit("/bebop/bemu/sim", params))
