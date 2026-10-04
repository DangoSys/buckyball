"""MCP tool: bbdev_bemu_sim."""

from __future__ import annotations
from typing import Any, Dict
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bemu_sim(
        chip: str,
        binary: str | None = None,
        pk: bool = False,
        disasm: bool = False,
        tool_profile: bool = False,
        itrace: bool = False,
        mtrace: bool = False,
        core_index: int | None = None,
        arguments: list[str] | None = None,
        model: str | None = None,
        reuse_simulator: bool = False,
    ) -> str:
        """Run a logical ELF file name from the selected chip's workload/kernel outputs, or a built model recipe, on BEMU. Model runs use running-param.toml. POST /bebop/bemu/sim."""
        if model is not None:
            for n, v in (("chip", chip), ("model", model)):
                if e := need(n, v):
                    return err(e)
            if binary is not None or pk or disasm or tool_profile or itrace or mtrace or core_index is not None or arguments:
                return err("model runs use running-param.toml; ELF options cannot be supplied")
            return fmt(submit("/bebop/bemu/sim", {
                "chip": chip, "model": model, "reuse-simulator": reuse_simulator}))
        if reuse_simulator:
            return err("reuse-simulator is supported for model runs only")
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
