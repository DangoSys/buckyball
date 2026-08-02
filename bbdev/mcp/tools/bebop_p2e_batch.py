"""MCP tool: bbdev_bebop_p2e_batch."""

from __future__ import annotations


from common import call, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_batch(
        chip: str,
        bitstream: str,
        build_dir: str,
        test: str,
    ) -> str:
        """Batch bebop-p2e regression. test: elf-tests|pk-tests. POST /bebop/p2e/batch."""
        for n, v in (
            ("chip", chip),
            ("bitstream", bitstream),
            ("build_dir", build_dir),
        ):
            if e := need(n, v):
                return err(e)
        if test not in ("elf-tests", "pk-tests"):
            return err("test must be elf-tests or pk-tests")
        return fmt(
            call(
                "/bebop/p2e/batch",
                {
                    "chip": chip,
                    "bitstream": bitstream,
                    "build-dir": build_dir,
                    "test": test,
                },
                timeout=14400,
            )
        )

