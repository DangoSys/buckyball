"""MCP tool: bbdev_bemu_sim."""

from __future__ import annotations
from typing import Any, Dict
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bemu_sim(
        design: str,
        binary: str | None = None,
        dtb: str | None = None,
        initrd: str | None = None,
        memory_mib: int | None = None,
        disasm: bool = False,
        tool_profile: bool = False,
        itrace: bool = False,
        mtrace: bool = False,
        device: bool = False,
        core: str | None = None,
    ) -> str:
        """Run a workload or kernel ELF on BEMU. POST /bebop/bemu/sim."""
        for n, v in (("design", design), ("binary", binary)):
            if e := need(n, v):
                return err(e)
        params: Dict[str, Any] = {
            "design": design,
            "binary": binary,
            "disasm": disasm,
            "tool-profile": tool_profile,
            "itrace": itrace,
            "mtrace": mtrace,
        }
        for key, value in (
            ("dtb", dtb),
            ("initrd", initrd),
            ("memory-mib", memory_mib),
        ):
            if value is not None:
                params[key] = value
        if device:
            params["device"] = True
        if core is not None:
            params["core"] = core
        return fmt(submit("/bebop/bemu/sim", params))
