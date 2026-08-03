"""MCP tool: bbdev_bebop_p2e_runworkload."""

from __future__ import annotations


from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_runworkload(image: str, bitstream: str) -> str:
        """Run one image on bebop-p2e FPGA. POST /bebop/p2e/runworkload."""
        for n, v in (("image", image), ("bitstream", bitstream)):
            if e := need(n, v):
                return err(e)
        return fmt(
            submit(
                "/bebop/p2e/runworkload",
                {"image": image, "bitstream": bitstream},
            )
        )

