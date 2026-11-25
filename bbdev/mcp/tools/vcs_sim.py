"""MCP tool: bbdev_vcs_sim."""

from __future__ import annotations

from typing import Any, Dict

from common import submit, err, fmt


def register(mcp):
    @mcp.tool()
    def bbdev_vcs_sim(
        model: str | None = None,
        bench: str | None = None,
        os: str | None = None,
        test: str | None = None,
        batch: bool = False,
        wave: bool = False,
        config: str | None = None,
        device: str | None = None,
        hart: str | None = None,
    ) -> str:
        """Run a built simv on a model or benchmark (model or bench, with os "baremetal" or "linux") or a
        test by its whole name (test). POST /vcs/sim.
        """
        if sum(x is not None for x in (model, bench, test)) != 1:
            return err("give exactly one of model (with os), bench (with os) or test")
        params: Dict[str, Any] = {}
        if batch:
            params["batch"] = True
        if wave:
            params["wave"] = True
        for key, value in (
            ("model", model),
            ("bench", bench),
            ("os", os),
            ("test", test),
        ):
            if value is not None:
                params[key] = value
        if config is not None:
            params["config"] = config
        if device is not None:
            params["device"] = device
        if hart is not None:
            params["hart"] = hart
        return fmt(submit("/vcs/sim", params))
