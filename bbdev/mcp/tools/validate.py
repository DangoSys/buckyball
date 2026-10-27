"""MCP tool: validate."""

from __future__ import annotations

from typing import Optional

from common import err, fmt, need, validate_balldomains


def register(mcp):
    @mcp.tool()
    def validate(design: str, device: str, balldomain: Optional[str] = None) -> str:
        """Validate the compiler Thread's BallDomain registration for a device config of a Design."""
        if e := need("design", design) or need("device", device):
            return err(e)
        try:
            return fmt(validate_balldomains(design, device, balldomain))
        except Exception as ex:
            return fmt(
                {"passed": False, "success": False, "failure": True, "error": str(ex)}
            )
