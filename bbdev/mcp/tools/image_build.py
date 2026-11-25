"""MCP tool: bbdev_image_build."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, fmt, need, err


def register(mcp):
    @mcp.tool()
    def bbdev_image_build(
        interactive: bool = False,
        config: str | None = None,
        device: str | None = None,
        model: str | None = None,
        bench: str | None = None,
    ) -> str:
        """Build RISC-V kernel + rootfs. POST /image/build.

        With --config and --device only: packs the programs listed in
        examples/designs/<design>/devices/<device>/sims/device/workloads/image/CMakeLists.txt
        and runs them under Linux using the shared workload /init.
        With --interactive: keep shared /init shell and do not auto-run.
        With model: include the model image in the DDR image.
        With bench: run the benchmark's Linux program (bench --build first).
        """
        params: Dict[str, Any] = {}
        if interactive:
            params["interactive"] = True
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if model is not None:
            params["model"] = model
        if bench is not None:
            params["bench"] = bench
        return fmt(submit("/image/build", params))
