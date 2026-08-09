"""MCP tool: bbdev_kernel_build."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import submit, fmt, opt


def register(mcp):
    @mcp.tool()
    def bbdev_kernel_build(
        chip: Optional[str] = None,
        model: Optional[str] = None,
        interactive: bool = False,
        visible_hart_count: Optional[int] = None,
        total_hart_count: Optional[int] = None,
    ) -> str:
        """Build RISC-V kernel + rootfs. POST /kernel/build.

        With --chip, uses examples/chips/<chip>/kernel OS overlay and packs
        that chip's bemu workloads-pk.toml into fw_payload-<chip>-pk.
        With --interactive, keep shared /init shell and do not auto-run.
        """
        params: Dict[str, Any] = {}
        opt(params, chip=chip, model=model)
        if interactive:
            params["interactive"] = True
        if visible_hart_count is not None:
            params["visible-hart-count"] = visible_hart_count
        if total_hart_count is not None:
            params["total-hart-count"] = total_hart_count
        return fmt(submit("/kernel/build", params))

