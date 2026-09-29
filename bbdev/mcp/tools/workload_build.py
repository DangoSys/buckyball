"""MCP tool: bbdev_workload_build."""

from __future__ import annotations
from typing import Any, Dict
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_workload_build(
        chip: str,
        ctest: bool = False,
        mlirtest: bool = False,
    ) -> str:
        """Build workloads for a chip. POST /workload/build."""
        if e := need("chip", chip):
            return err(e)
        params: Dict[str, Any] = {"chip": chip}
        if ctest and mlirtest:
            return err("ctest and mlirtest cannot be used together")
        if ctest:
            params["ctest"] = True
        if mlirtest:
            params["mlirtest"] = True
        return fmt(submit("/workload/build", params))
