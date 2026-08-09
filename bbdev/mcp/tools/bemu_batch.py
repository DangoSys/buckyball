"""MCP tool: bbdev_bemu_batch."""

from __future__ import annotations


from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_bemu_batch(
        chip: str, test: str, clean_before: bool = False, rushB: bool = False
    ) -> str:
        """Batch bemu regression. test: elf-tests|pk-tests. POST /bebop/bemu/batch."""
        if e := need("chip", chip):
            return err(e)
        if test not in ("elf-tests", "pk-tests"):
            return err("test must be elf-tests or pk-tests")
        return fmt(
            submit(
                "/bebop/bemu/batch",
                {
                    "chip": chip,
                    "test": test,
                    "clean-before": clean_before,
                    "rushB": rushB,
                },
            )
        )
