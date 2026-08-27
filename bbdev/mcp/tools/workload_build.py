"""MCP tool: bbdev_workload_build."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import submit, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_workload_build(
        chip: str,
        model: Optional[str] = None,
        rushB: Optional[str] = None,
        ctest: bool = False,
        mlirtest: bool = False,
    ) -> str:
        """Build workloads for a chip. POST /workload/build."""
        if e := need("chip", chip):
            return err(e)
        params: Dict[str, Any] = {"chip": chip}
        if model:
            params["model"] = model
        if rushB:
            if rushB not in {"bemu", "verilator"}:
                return err("rushB must be bemu or verilator")
            params["rushB"] = rushB
        if ctest and mlirtest:
            return err("ctest and mlirtest cannot be used together")
        if (ctest or mlirtest) and (model or rushB):
            return err("ctest and mlirtest cannot be used with model or rushB")
        if ctest:
            params["ctest"] = True
        if mlirtest:
            params["mlirtest"] = True
        return fmt(submit("/workload/build", params))
