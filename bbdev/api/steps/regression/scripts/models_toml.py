import tomllib
from pathlib import Path


def load_eval_models(chip: str, bbdir: str) -> list[str]:
    path = Path(bbdir) / "examples" / "chips" / chip / "regression" / "eval" / "models.toml"
    if not path.is_file():
        raise ValueError(f"eval models toml does not exist: {path}")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    eval_sec = data.get("eval")
    if not isinstance(eval_sec, dict):
        raise ValueError(f"missing [eval] section: {path}")
    models = eval_sec.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"[eval].models must be a non-empty list: {path}")
    for name in models:
        if not isinstance(name, str) or not name:
            raise ValueError(f"[eval].models entries must be non-empty strings: {path}")
    return models
