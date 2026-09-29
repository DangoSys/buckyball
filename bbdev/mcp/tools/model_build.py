"""MCP tool: bbdev_model_build."""

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_model_build(chip: str, model: str | None = None, ctrace: bool = False, dtrace: bool = False) -> str:
        """Build one model recipe, or all recipes for the chip when model is omitted. POST /model/build."""
        if error := need("chip", chip):
            return err(error)
        params = {"chip": chip, "ctrace": ctrace, "dtrace": dtrace}
        if model is not None:
            params["model"] = model
        return fmt(submit("/model/build", params))
