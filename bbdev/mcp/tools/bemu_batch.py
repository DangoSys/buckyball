"""MCP tool: bbdev_bemu_batch."""

from __future__ import annotations
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bemu_batch(
        test: str,
        jobs: int = 1,
        config: str | None = None,
        device: str | None = None,
    ) -> str:
        """Batch bemu regression. test: bare-tests|linux-tests. POST /bebop/bemu/batch."""
        if test not in ("bare-tests", "linux-tests"):
            return err("test must be bare-tests or linux-tests")
        request = {
            "test": test,
            "jobs": jobs,
        }
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        return fmt(
            submit(
                "/bebop/bemu/batch",
                request,
            )
        )
