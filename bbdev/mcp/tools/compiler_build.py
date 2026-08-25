"""MCP tool: bbdev_compiler_build."""

from __future__ import annotations


from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_compiler_build(chip: str) -> str:
        """Build buddy-mlir compiler for one chip. POST /compiler/build."""
        return fmt(submit("/compiler/build", {"chip": chip}))
