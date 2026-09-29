"""MCP tool: bbdev_bemu_sim."""

from __future__ import annotations
from typing import Any, Dict
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bemu_sim(
        chip: str,
        binary: str,
        pk: bool = False,
        disasm: bool = False,
        tool_profile: bool = False,
        itrace: bool = False,
        mtrace: bool = False,
        core_index: int | None = None,
        arguments: list[str] | None = None,
    ) -> str:
        """Run one workload on bebop-bemu. POST /bebop/bemu/sim."""
        for n, v in (("chip", chip), ("binary", binary)):
            if e := need(n, v):
                return err(e)
        params: Dict[str, Any] = {
            "chip": chip,
            "binary": binary,
            "pk": pk,
            "disasm": disasm,
            "tool-profile": tool_profile,
            "itrace": itrace,
            "mtrace": mtrace,
            "core_index": core_index,
            "arguments": arguments if arguments is not None else [],
        }
        return fmt(submit("/bebop/bemu/sim", params))
