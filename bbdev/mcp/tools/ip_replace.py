"""MCP tool: bbdev_ip_replace."""

from __future__ import annotations

from typing import Optional

from common import err, fmt, need, opt, submit


def register(mcp):
    @mcp.tool()
    def bbdev_ip_replace(
        consumer: Optional[str] = None,
        top: Optional[str] = None,
        next_topic: Optional[str] = None,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Assemble synthesis source list plus SRAM macros. POST /ip/replace."""
        request = opt(
            {},
            consumer=consumer,
            top=top,
            next_topic=next_topic,
        )
        if config is not None:
            request["config"] = config
        if device is not None:
            request["device"] = device
        if hart is not None:
            request["hart"] = hart
        return fmt(
            submit(
                "/ip/replace",
                request,
            )
        )
