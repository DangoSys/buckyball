"""MCP tool: bbdev_model_build."""

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_model_build(device: str, model: str, config: str | None = None) -> str:
        """Build one model recipe on a device (its default config unless config
        names another), or all recipes of the design with model "all".
        POST /model/build."""
        if error := need("device", device) or need("model", model):
            return err(error)
        params = {"device": device, "model": model}
        if config is not None:
            params["config"] = config
        return fmt(submit("/model/build", params))
