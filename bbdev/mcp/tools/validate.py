"""MCP tool: validate."""

from __future__ import annotations

from typing import Optional

from common import balldomain_path, err, fmt, need, validate_toml


def register(mcp):
    @mcp.tool()
    def validate(chip: str = "toy", balldomain: Optional[str] = None) -> str:
        """Validate chip balldomain TOML registration."""
        if e := need("chip", chip):
            return err(e)
        try:
            return fmt(validate_toml(balldomain_path(chip, balldomain)))
        except Exception as ex:
            return fmt(
                {"passed": False, "success": False, "failure": True, "error": str(ex)}
            )

