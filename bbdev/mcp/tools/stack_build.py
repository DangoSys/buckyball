"""MCP tool: bbdev_stack_build."""

from __future__ import annotations


from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_stack_build(
        design: str, core: str | None = None, device: str | None = None
    ) -> str:
        """Build a registered device or core simulation stack. POST /stack/build."""
        body = {"design": design}
        if core is not None:
            body["core"] = core
        if device is not None:
            body["device"] = device
        return fmt(submit("/stack/build", body))
