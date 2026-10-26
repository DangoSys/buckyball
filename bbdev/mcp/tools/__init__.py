"""Discover MCP capabilities from their owning tool modules."""

import importlib
import pkgutil


def modules():
    return [
        importlib.import_module(f"{__name__}.{entry.name}")
        for entry in sorted(
            pkgutil.iter_modules(__path__), key=lambda entry: entry.name
        )
        if not entry.name.startswith("_")
    ]


def register_all(mcp):
    for module in modules():
        module.register(mcp)
