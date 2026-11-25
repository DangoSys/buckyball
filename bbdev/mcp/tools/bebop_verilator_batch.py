"""MCP tool: bbdev_bebop_verilator_batch."""

from __future__ import annotations
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bebop_verilator_batch(
        test: str,
        diff: bool = False,
        config: str | None = None,
        device: str | None = None,
    ) -> str:
        """Batch bebop-verilator regression. POST /bebop/verilator/batch."""
        if test not in ("bare-tests", "linux-tests"):
            return err("test must be bare-tests or linux-tests")
        request = {
            "test": test,
            "diff": diff,
        }
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        return fmt(
            submit(
                "/bebop/verilator/batch",
                request,
            )
        )
