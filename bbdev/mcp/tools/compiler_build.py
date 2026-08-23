"""MCP tool: bbdev_compiler_build."""

from __future__ import annotations


from typing import Any, Dict, Optional

from common import submit, err, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_compiler_build(
        core: Optional[str] = None, chip: Optional[str] = None, stable: bool = False
    ) -> str:
        """Build buddy-mlir compiler for exactly one Core or chip. POST /compiler/build."""
        if bool(core) == bool(chip):
            return err("specify exactly one compiler target: core or chip")
        params: Dict[str, Any] = {"stable": stable}
        if core:
            params["core"] = core
        else:
            params["chip"] = chip
        return fmt(submit("/compiler/build", params))
