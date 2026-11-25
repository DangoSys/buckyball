"""MCP tool: bbdev_stack_build."""

from __future__ import annotations


from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_stack_build(
        hart: str | None = None, config: str | None = None, device: str | None = None
    ) -> str:
        """Build a registered device or hart simulation stack. POST /stack/build."""
        body = {}
        if hart is not None:
            body["hart"] = hart
        if config is not None:
            body["config"] = config
        if device is not None:
            body["device"] = device
        return fmt(submit("/stack/build", body))
