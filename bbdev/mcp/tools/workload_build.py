"""MCP tool: bbdev_workload_build."""

from __future__ import annotations
from typing import Any, Dict
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_workload_build(
        design: str,
        ctest: bool = False,
        mlirtest: bool = False,
        device: bool = False,
        core: str | None = None,
    ) -> str:
        """Build workloads for a design. POST /workload/build."""
        if e := need("design", design):
            return err(e)
        params: Dict[str, Any] = {"design": design}
        if ctest and mlirtest:
            return err("ctest and mlirtest cannot be used together")
        if ctest:
            params["ctest"] = True
        if mlirtest:
            params["mlirtest"] = True
        if device:
            params["device"] = True
        if core is not None:
            params["core"] = core
        return fmt(submit("/workload/build", params))
