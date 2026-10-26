"""MCP tool: validate."""

from __future__ import annotations

from typing import Optional

from common import err, fmt, need, validate_balldomains


def register(mcp):
    @mcp.tool()
    def validate(design: str = "toy", balldomain: Optional[str] = None) -> str:
        """Validate the compiler Thread's BallDomain registration for a Design."""
        if e := need("design", design):
            return err(e)
        try:
            return fmt(validate_balldomains(design, balldomain))
        except Exception as ex:
            return fmt(
                {"passed": False, "success": False, "failure": True, "error": str(ex)}
            )
