"""MCP tool: bbdev_ip_replace."""

from __future__ import annotations

from typing import Optional

from common import err, fmt, need, opt, submit


def register(mcp):
    @mcp.tool()
    def bbdev_ip_replace(
        chip: str,
        consumer: Optional[str] = None,
        top: Optional[str] = None,
        output_dir: Optional[str] = None,
        next_topic: Optional[str] = None,
    ) -> str:
        """Assemble synthesis source list plus SRAM macros. POST /ip/replace."""
        if e := need("chip", chip):
            return err(e)
        return fmt(
            submit(
                "/ip/replace",
                opt(
                    {"chip": chip},
                    consumer=consumer,
                    top=top,
                    output_dir=output_dir,
                    next_topic=next_topic,
                ),
            )
        )
