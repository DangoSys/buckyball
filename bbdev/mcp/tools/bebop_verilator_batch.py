"""MCP tool: bbdev_bebop_verilator_batch."""

from __future__ import annotations
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bebop_verilator_batch(
        design: str,
        test: str,
        clean_before: bool = False,
        diff: bool = False,
        junit: str | None = None,
        device: bool = False,
    ) -> str:
        """Batch bebop-verilator regression. POST /bebop/verilator/batch."""
        if e := need("design", design):
            return err(e)
        if test not in ("bare-tests", "linux-tests"):
            return err("test must be bare-tests or linux-tests")
        request = {
            "design": design,
            "test": test,
            "clean-before": clean_before,
            "diff": diff,
            "junit": junit,
        }
        if device:
            request["device"] = True
        return fmt(
            submit(
                "/bebop/verilator/batch",
                request,
            )
        )
