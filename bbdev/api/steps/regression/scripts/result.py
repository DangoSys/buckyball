import json
import os
from pathlib import Path


def result_path(bbdir: str) -> Path:
    rel = os.environ.get("EVAL_RESULT_PATH", "chipcrowd-eval-result.json")
    root = Path(bbdir).resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError as e:
        raise ValueError(f"EVAL_RESULT_PATH escapes repo: {rel}") from e
    return target


def merge_metrics(bbdir: str, **fields) -> dict:
    path = result_path(bbdir)
    data = {}
    if path.is_file():
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"eval result is not an object: {path}")
    data.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data
