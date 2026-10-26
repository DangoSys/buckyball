"""MCP tool: bbdev_dc_power."""

from __future__ import annotations

from typing import Optional

from common import err, fmt, opt, submit


def register(mcp):
    @mcp.tool()
    def bbdev_dc_power(
        design: Optional[str] = None,
        top: Optional[str] = None,
        activity: Optional[str] = None,
        activity_format: Optional[str] = None,
        strip_path: Optional[str] = None,
        workload: Optional[str] = None,
        start_ns: Optional[str] = None,
        end_ns: Optional[str] = None,
        device: bool = False,
        core: str | None = None,
    ) -> str:
        """Run DC synthesis, design-owned activity simulation, and PrimeTime PX.

        Omit activity/activity_format for the normal rerun-and-measure flow.
        Supply both only as a debugging override.
        """
        if not design:
            return err("design is required")
        if bool(activity) != bool(activity_format):
            return err("activity and activity_format must be provided together")
        request = opt(
            {},
            design=design,
            top=top,
            activity=activity,
            format=activity_format,
            strip_path=strip_path,
            workload=workload,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        if device:
            request["device"] = True
        if core is not None:
            request["core"] = core
        return fmt(
            submit(
                "/dc/power",
                request,
            )
        )
