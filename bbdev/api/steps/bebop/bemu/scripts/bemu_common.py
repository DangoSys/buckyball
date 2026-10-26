from pathlib import Path

from utils.event_common import require_design
from utils.path import runner_manifest


def _repo(bbdir: str | None) -> Path:
    from utils.path import get_buckyball_path

    return Path(bbdir or get_buckyball_path())


def bemu_manifest(design: str, bbdir: str | None = None, *, selection: str) -> Path:
    design = require_design({"design": design})
    root = _repo(bbdir)
    path = runner_manifest(root, design, selection)
    if not path.is_file():
        raise FileNotFoundError(f"missing design manifest: {path}")
    return path


def bemu_binary() -> str:
    return "bebop-bemu"
