"""MCP tool: bbdev_stack_build."""

from __future__ import annotations


from common import submit, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_stack_build(
        design: str, core: str | None = None, device: bool = False
    ) -> str:
        """Build a registered device or core simulation stack. POST /stack/build."""
        body = {"design": design}
        if core is not None:
            body["core"] = core
        if device:
            body["device"] = True
        return fmt(submit("/stack/build", body))
