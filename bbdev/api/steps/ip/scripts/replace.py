from __future__ import annotations

import json
import re
from pathlib import Path

MODULE_NAME_RE = re.compile(
    r"^\s*(?:\(\*.*?\*\)\s*)*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b",
    re.M,
)
INSTANCE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s*(?:#\s*\(.*?\)\s*)?([A-Za-z_][A-Za-z0-9_$]*)\s*\(",
    re.M | re.S,
)
COMMENT_RE = re.compile(r"/\*.*?\*/|//.*?$", re.M | re.S)
MODULE_TOKEN_RE = re.compile(r"\bmodule\b")


def replace_sources(
    *,
    source_paths: list[str],
    top_module: str,
    sram_macros_v: str | Path,
    output_dir: str | Path,
    consumer: str,
    generate_manifest: str | Path | None = None,
) -> dict:
    paths = [Path(path).resolve() for path in source_paths]
    macros = Path(sram_macros_v).resolve()

    modules: dict[str, Path] = {}
    texts: dict[str, str] = {}
    for path in paths:
        text = path.read_text()
        match = MODULE_NAME_RE.search(text)
        if match is None:
            stripped = COMMENT_RE.sub("", text)
            if MODULE_TOKEN_RE.search(stripped):
                raise RuntimeError(f"no module declaration: {path}")
            continue
        name = match.group(1)
        if name in modules:
            raise RuntimeError(f"duplicate module {name}: {modules[name]} vs {path}")
        modules[name] = path
        texts[name] = text

    if top_module not in modules:
        raise ValueError(f"synthesis top is not defined in source list: {top_module}")
    if not macros.is_file():
        raise FileNotFoundError(f"missing sram_macros.v: {macros}")

    reachable = {top_module}
    pending = [top_module]
    while pending:
        cur = pending.pop()
        for child, _inst in INSTANCE_RE.findall(texts[cur]):
            if child in modules and child not in reachable:
                reachable.add(child)
                pending.append(child)

    selected_paths = {modules[name] for name in reachable}
    selected = [path for path in paths if path in selected_paths]
    if macros not in selected:
        selected.append(macros)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_list_path = out / f"{consumer}_sources.list"
    source_list_path.write_text("\n".join(str(path) for path in selected) + "\n")

    manifest = {
        "top_module": top_module,
        "consumer": consumer,
        "source_count": len(selected),
        "sram_macros_v": str(macros),
    }
    if generate_manifest is not None:
        gman = Path(generate_manifest).resolve()
        if not gman.is_file():
            raise FileNotFoundError(f"missing generate_manifest: {gman}")
        manifest["generate_manifest"] = str(gman)
    manifest_path = out / "replace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "source_list": str(source_list_path),
        "replace_manifest": str(manifest_path),
        "source_count": len(selected),
    }
