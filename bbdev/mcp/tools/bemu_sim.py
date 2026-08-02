"""MCP tool: bbdev_bemu_sim."""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import call, err, fmt, need, opt


def register(mcp):
    @mcp.tool()
    def bbdev_bemu_sim(
        chip: str,
        binary: str,
        pk: bool = False,
        log_dir: Optional[str] = None,
    ) -> str:
        """Run one workload on bebop-bemu. POST /bebop/bemu/sim."""
        for n, v in (("chip", chip), ("binary", binary)):
            if e := need(n, v):
                return err(e)
        params: Dict[str, Any] = {"chip": chip, "binary": binary, "pk": pk}
        if log_dir:
            params["log_dir"] = log_dir
        return fmt(call("/bebop/bemu/sim", params, timeout=1800))

