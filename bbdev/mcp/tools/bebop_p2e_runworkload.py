"""MCP tool: bbdev_bebop_p2e_runworkload."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_runworkload(
        chip: str,
        image: str = "",
        bitstream: str = "",
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        reuse_runtime: bool = False,
        diff: bool = False,
        load_manifest: str = "",
    ) -> str:
        """Cold-load one hex image or a validated binary load manifest on P2E."""
        if bool(image) == bool(load_manifest):
            return err("exactly one of image and load_manifest is required")
        if load_manifest and diff:
            return err("load_manifest cannot use single-ELF diff")
        for n, v in (("chip", chip), ("bitstream", bitstream)):
            if e := need(n, v):
                return err(e)
        params: Dict[str, Any] = {"chip": chip, "image": image, "bitstream": bitstream}
        if load_manifest:
            params.pop("image")
            params["load-manifest"] = load_manifest
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
        return fmt(submit("/bebop/p2e/runworkload", params))
