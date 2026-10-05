"""MCP tool: bbdev_bebop_verilator_batch."""

from __future__ import annotations
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bebop_verilator_batch(
        chip: str, test: str, clean_before: bool = False, diff: bool = False
    ) -> str:
        """Batch bebop-verilator regression. POST /bebop/verilator/batch."""
        if e := need("chip", chip):
            return err(e)
        if test not in ("bare-tests", "linux-tests"):
            return err("test must be bare-tests or linux-tests")
        return fmt(
            submit(
                "/bebop/verilator/batch",
                {
                    "chip": chip,
                    "test": test,
                    "clean-before": clean_before,
                    "diff": diff,
                },
            )
        )
