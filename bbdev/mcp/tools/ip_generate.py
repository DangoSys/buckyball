"""MCP tool: bbdev_ip_generate."""

from __future__ import annotations

from typing import Optional

from common import err, fmt, need, opt, submit


def register(mcp):
    @mcp.tool()
    def bbdev_ip_generate(
        design: str,
        consumer: Optional[str] = None,
        top: Optional[str] = None,
        output_dir: Optional[str] = None,
        next_topic: Optional[str] = None,
        device: bool = False,
        core: str | None = None,
    ) -> str:
        """Generate SRAM macros from mems.conf. POST /ip/generate."""
        if e := need("design", design):
            return err(e)
        request = opt(
            {"design": design},
            consumer=consumer,
            top=top,
            output_dir=output_dir,
            next_topic=next_topic,
        )
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(
            submit(
                "/ip/generate",
                request,
            )
        )
