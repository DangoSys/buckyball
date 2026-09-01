"""MCP tool: bbdev_ip_generate."""

from __future__ import annotations

from typing import Optional

from common import err, fmt, need, opt, submit


def register(mcp):
    @mcp.tool()
    def bbdev_ip_generate(
        chip: str,
        consumer: Optional[str] = None,
        top: Optional[str] = None,
        output_dir: Optional[str] = None,
        next_topic: Optional[str] = None,
    ) -> str:
        """Generate SRAM macros from mems.conf. POST /ip/generate."""
        if e := need("chip", chip):
            return err(e)
        return fmt(
            submit(
                "/ip/generate",
                opt(
                    {"chip": chip},
                    consumer=consumer,
                    top=top,
                    output_dir=output_dir,
                    next_topic=next_topic,
                ),
            )
        )
