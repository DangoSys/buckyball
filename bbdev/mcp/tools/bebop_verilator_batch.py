"""MCP tool: bbdev_bebop_verilator_batch."""

from __future__ import annotations
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bebop_verilator_batch(
        design: str,
        test: str,
        diff: bool = False,
        device: str | None = None,
    ) -> str:
        """Batch bebop-verilator regression. POST /bebop/verilator/batch."""
        if e := need("design", design):
            return err(e)
        if test not in ("bare-tests", "linux-tests"):
            return err("test must be bare-tests or linux-tests")
        request = {
            "design": design,
            "test": test,
            "diff": diff,
        }
        if device is not None:
            request["device"] = device
        return fmt(
            submit(
                "/bebop/verilator/batch",
                request,
            )
        )
