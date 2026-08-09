"""MCP tool: bbdev_bebop_p2e_runworkload."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_runworkload(
        image: str,
        bitstream: str,
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
    ) -> str:
        """Run one image on bebop-p2e FPGA. POST /bebop/p2e/runworkload."""
        for n, v in (("image", image), ("bitstream", bitstream)):
            if e := need(n, v):
                return err(e)
        params: Dict[str, Any] = {"image": image, "bitstream": bitstream}
        if itrace:
            params["itrace"] = True
        if mtrace:
            params["mtrace"] = True
        if pmctrace:
            params["pmctrace"] = True
        return fmt(submit("/bebop/p2e/runworkload", params))

