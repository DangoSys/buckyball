"""MCP tool: bbdev_bemu_batch."""

from __future__ import annotations
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bemu_batch(
        design: str,
        test: str,
        clean_before: bool = False,
        jobs: int = 1,
        junit: str | None = None,
        device: bool = False,
    ) -> str:
        """Batch bemu regression. test: bare-tests|linux-tests. POST /bebop/bemu/batch."""
        if e := need("design", design):
            return err(e)
        if test not in ("bare-tests", "linux-tests"):
            return err("test must be bare-tests or linux-tests")
        request = {
            "design": design,
            "test": test,
            "clean-before": clean_before,
            "jobs": jobs,
            "junit": junit,
        }
        if device:
            request["device"] = True
        return fmt(
            submit(
                "/bebop/bemu/batch",
                request,
            )
        )
