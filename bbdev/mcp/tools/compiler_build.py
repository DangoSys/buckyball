"""MCP tool: bbdev_compiler_build."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_compiler_build(chip: str, stable: bool = False) -> str:
        """Build buddy-mlir compiler for a chip. POST /compiler/build."""
        if e := need("chip", chip):
            return err(e)
        return fmt(
            submit("/compiler/build", {"chip": chip, "stable": stable})
        )

