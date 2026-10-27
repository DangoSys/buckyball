"""MCP tool: bbdev_bebop_p2e_runworkload."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_runworkload(
        design: str,
        image: str = "",
        bitstream: str = "",
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        reuse_runtime: bool = False,
        diff: bool = False,
        device: str | None = None,
        core: str | None = None,
    ) -> str:
        """Cold-load one hex image on P2E."""
        for n, v in (("design", design), ("image", image), ("bitstream", bitstream)):
            if e := need(n, v):
                return err(e)
        params: Dict[str, Any] = {
            "design": design,
            "image": image,
            "bitstream": bitstream,
        }
        if itrace:
            params["itrace"] = True
        if mtrace:
            params["mtrace"] = True
        if reuse_runtime:
            params["reuse-runtime"] = True
        if pmctrace:
            params["pmctrace"] = True
        if ctrace:
            params["ctrace"] = True
        if diff:
            params["diff"] = True
        if device is not None:
            params["device"] = device
        if core is not None:
            params["core"] = core
        return fmt(submit("/bebop/p2e/runworkload", params))
