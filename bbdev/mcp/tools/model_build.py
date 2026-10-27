"""MCP tool: bbdev_model_build."""

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_model_build(design: str, device: str, model: str | None = None) -> str:
        """Build one model recipe on a device config, or all recipes of the design
        when model is omitted. POST /model/build."""
        if error := need("design", design) or need("device", device):
            return err(error)
        params = {"design": design, "device": device}
        if model is not None:
            params["model"] = model
        return fmt(submit("/model/build", params))
