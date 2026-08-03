"""MCP tool: bbdev_bebop_verilator_batch."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_verilator_batch(
        chip: str,
        config: str,
        test: str,
        clean_before: bool = False,
    ) -> str:
        """Batch bebop-verilator regression. POST /bebop/verilator/batch."""
        for n, v in (("chip", chip), ("config", config)):
            if e := need(n, v):
                return err(e)
        if test not in ("elf-tests", "pk-tests"):
            return err("test must be elf-tests or pk-tests")
        return fmt(
            submit(
                "/bebop/verilator/batch",
                {
                    "chip": chip,
                    "config": config,
                    "test": test,
                    "clean-before": clean_before,
                },
            )
        )

