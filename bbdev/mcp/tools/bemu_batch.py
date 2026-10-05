"""MCP tool: bbdev_bemu_batch."""

from __future__ import annotations
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_bemu_batch(
        chip: str,
        test: str,
        clean_before: bool = False,
        jobs: int = 1,
    ) -> str:
        """Batch bemu regression. test: bare-tests|linux-tests. POST /bebop/bemu/batch."""
        if e := need("chip", chip):
            return err(e)
        if test not in ("bare-tests", "linux-tests"):
            return err("test must be bare-tests or linux-tests")
        return fmt(
            submit(
                "/bebop/bemu/batch",
                {
                    "chip": chip,
                    "test": test,
                    "clean-before": clean_before,
                    "jobs": jobs,
                },
            )
        )
