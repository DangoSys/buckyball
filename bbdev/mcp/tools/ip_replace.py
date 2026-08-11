"""MCP tool: bbdev_ip_replace_run."""

from __future__ import annotations

from typing import Optional

from common import err, fmt, opt, submit


def register(mcp):
    @mcp.tool()
    def bbdev_ip_replace_run(
        source_list: str,
        output_dir: str,
        top: Optional[str] = None,
        consumer: Optional[str] = None,
    ) -> str:
        """Replace behavioral IP RTL. POST /ip/replace/run."""
        if not source_list or not output_dir:
            return err("source_list and output_dir are required")
        return fmt(
            submit(
                "/ip/replace/run",
                opt({}, source_list=source_list, output_dir=output_dir, top=top, consumer=consumer),
            )
        )
