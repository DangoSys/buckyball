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
        guest_memory_mib: int = 512,
        model_storage: str = "initramfs",
        visible_hart_count: Optional[int] = None,
        total_hart_count: Optional[int] = None,
    ) -> str:
        """Build RISC-V kernel + rootfs. POST /kernel/build.

        With --chip only: packs programs listed in examples/chips/<chip>/kernel/workloads.toml
        and runs them under Linux using the shared workload /init.
        With --model: requires --chip; packs a native model artifact from
        stack/models/build/<chip>/<model>/artifact into fw_payload-<model>.
        With --interactive: keep shared /init shell and do not auto-run.
        """
        params: Dict[str, Any] = {"guest-memory-mib": guest_memory_mib, "model-storage": model_storage}
        opt(params, chip=chip, model=model)
        if interactive:
            params["interactive"] = True
        if visible_hart_count is not None:
            params["visible-hart-count"] = visible_hart_count
        if total_hart_count is not None:
            params["total-hart-count"] = total_hart_count
        return fmt(submit("/kernel/build", params))

