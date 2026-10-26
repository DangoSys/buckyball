"""MCP tool: bbdev_model_build."""

from common import submit, err, fmt, need


def register(mcp):
    @mcp.tool()
    def bbdev_model_build(design: str, model: str | None = None) -> str:
        """Build one model recipe, or all recipes for the design when model is omitted. POST /model/build."""
        if error := need("design", design):
            return err(error)
        params = {"design": design}
        if model is not None:
            params["model"] = model
        return fmt(submit("/model/build", params))
