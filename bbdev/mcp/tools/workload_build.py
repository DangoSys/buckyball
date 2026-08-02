"""MCP tool: bbdev_workload_build."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import call, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_workload_build(chip: str, model: Optional[str] = None) -> str:
        """Build workloads for a chip. POST /workload/build."""
        if e := need("chip", chip):
            return err(e)
        params: Dict[str, Any] = {"chip": chip}
        if model:
            params["model"] = model
        return fmt(call("/workload/build", params, timeout=1800))

