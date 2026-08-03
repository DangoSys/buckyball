"""MCP tool: bbdev_kernel_build."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import submit, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_kernel_build(
        model: Optional[str] = None,
        visible_hart_count: Optional[int] = None,
        total_hart_count: Optional[int] = None,
    ) -> str:
        """Build RISC-V kernel + rootfs. POST /kernel/build."""
        params: Dict[str, Any] = {}
        opt(params, model=model)
        if visible_hart_count is not None:
            params["visible-hart-count"] = visible_hart_count
        if total_hart_count is not None:
            params["total-hart-count"] = total_hart_count
        return fmt(submit("/kernel/build", params))

