"""MCP tool: bbdev_bebop_p2e_runworkload."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_bebop_p2e_runworkload(
        model: str | None = None,
        bench: str | None = None,
        os: str | None = None,
        test: str | None = None,
        bitstream: str = "",
        itrace: bool = False,
        mtrace: bool = False,
        pmctrace: bool = False,
        ctrace: bool = False,
        banktrace: bool = False,
        reuse_runtime: bool = False,
        diff: bool = False,
        ddr_image: str | None = None,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Cold-load on P2E a model or a benchmark (model or bench, with os "baremetal" or "linux") or a
        test by its whole name (test), as the hex of what goes into the DDR."""
        if sum(x is not None for x in (model, bench, test)) != 1:
            return err("give exactly one of model (with os), bench (with os) or test")
        if e := need("bitstream", bitstream):
            return err(e)
        params: Dict[str, Any] = {"bitstream": bitstream}
        for key, value in (
            ("model", model),
            ("bench", bench),
            ("os", os),
            ("test", test),
        ):
            if value is not None:
                params[key] = value
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
        if banktrace:
            params["banktrace"] = True
        if diff:
            params["diff"] = True
        if ddr_image is not None:
            if not diff:
                return err("ddr_image requires diff")
            params["ddr-image"] = ddr_image
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if hart is not None:
            params["hart"] = hart
        return fmt(submit("/bebop/p2e/runworkload", params))
