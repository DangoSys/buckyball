"""MCP tool: bbdev_kernel_build."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, fmt, need, err


def register(mcp):
    @mcp.tool()
    def bbdev_kernel_build(
        design: str,
        interactive: bool = False,
        device: str | None = None,
        model: str | None = None,
    ) -> str:
        """Build RISC-V kernel + rootfs. POST /kernel/build.

        With --design only: packs the programs listed in
        examples/designs/<design>/sims/device/workloads/kernel/CMakeLists.txt
        and runs them under Linux using the shared workload /init.
        With --interactive: keep shared /init shell and do not auto-run.
        With model: include the model image in the DDR image.
        """
        if e := need("design", design):
            return err(e)
        params: Dict[str, Any] = {
            "design": design,
        }
        if interactive:
            params["interactive"] = True
        if device is not None:
            params["device"] = device
        if model is not None:
            params["model"] = model
        return fmt(submit("/kernel/build", params))
