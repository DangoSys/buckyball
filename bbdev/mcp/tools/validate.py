"""MCP tool: validate."""

from __future__ import annotations

from typing import Optional

from common import err, fmt, need, repo_path, validate_balldomains


def register(mcp):
    @mcp.tool()
    def validate(
        device: str, config: Optional[str] = None, balldomain: Optional[str] = None
    ) -> str:
        """Validate the compiler Thread's BallDomain registration for a device
        config of a Design: the device's default config unless config names another."""
        if e := need("device", device):
            return err(e)
        if config is None:
            (default,) = (repo_path() / "examples/designs").glob(
                f"*/devices/{device}/configs/*DefaultConfig.toml"
            )
            config = str(default.relative_to(repo_path() / "examples/designs"))
        try:
            return fmt(validate_balldomains(config, device, balldomain))
        except Exception as ex:
            return fmt(
                {"passed": False, "success": False, "failure": True, "error": str(ex)}
            )
