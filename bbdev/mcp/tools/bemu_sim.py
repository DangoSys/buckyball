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
        rushB: bool = False,
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
            "rushB": rushB,
        }
        return fmt(submit("/bebop/bemu/sim", params))
