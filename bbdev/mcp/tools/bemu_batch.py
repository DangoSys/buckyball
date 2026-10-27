"""MCP tool: bbdev_bemu_batch."""

from __future__ import annotations
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bemu_batch(
        design: str,
        test: str,
        jobs: int = 1,
        device: str | None = None,
    ) -> str:
        """Batch bemu regression. test: bare-tests|linux-tests. POST /bebop/bemu/batch."""
        if e := need("design", design):
            return err(e)
        if test not in ("bare-tests", "linux-tests"):
            return err("test must be bare-tests or linux-tests")
        request = {
            "design": design,
            "test": test,
            "jobs": jobs,
        }
        if device is not None:
            request["device"] = device
        return fmt(
            submit(
                "/bebop/bemu/batch",
                request,
            )
        )
