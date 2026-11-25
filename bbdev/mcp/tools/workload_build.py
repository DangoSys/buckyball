"""MCP tool: bbdev_workload_build."""

from __future__ import annotations
from typing import Any, Dict
from common import submit, err, fmt, need


def register(mcp):

    @mcp.tool()
    def bbdev_workload_build(
        ctest: bool = False,
        mlirtest: bool = False,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Build workloads for a design. POST /workload/build."""
        params: Dict[str, Any] = {}
        if ctest and mlirtest:
            return err("ctest and mlirtest cannot be used together")
        if ctest:
            params["ctest"] = True
        if mlirtest:
            params["mlirtest"] = True
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if hart is not None:
            params["hart"] = hart
        return fmt(submit("/workload/build", params))
