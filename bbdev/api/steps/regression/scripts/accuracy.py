"""Per-model accuracy parsing for regression eval-performance.

UART contracts (fail loud if missing / malformed):
  top1  -> line "top1=<correct>/<total>"
  map   -> line "map=<float in [0,1]>"
"""
from __future__ import annotations

import re
from typing import Any

_TOP1 = re.compile(r"^top1=(\d+)/(\d+)\s*$", re.M)
_MAP = re.compile(r"^map=([0-9]*\.?[0-9]+)\s*$", re.M)


def accuracy_from_uart(metric: str, uart: str) -> float:
    if not isinstance(metric, str) or not metric:
        raise ValueError("accuracy metric must be a non-empty string")
    if not isinstance(uart, str) or not uart:
        raise ValueError("uart log is empty; cannot parse accuracy")
    m = metric.lower()
    if m == "top1":
        return _top1(uart)
    if m == "map":
        return _map(uart)
    raise ValueError(f"unknown accuracy metric: {metric}")


def _top1(uart: str) -> float:
    hits = _TOP1.findall(uart)
    if not hits:
        raise ValueError("uart missing top1=<correct>/<total> line")
    correct_s, total_s = hits[-1]
    correct = int(correct_s)
    total = int(total_s)
    if total <= 0:
        raise ValueError(f"top1 total must be > 0, got {total}")
    if correct < 0 or correct > total:
        raise ValueError(f"top1 correct out of range: {correct}/{total}")
    return correct / total


def _map(uart: str) -> float:
    hits = _MAP.findall(uart)
    if not hits:
        raise ValueError("uart missing map=<float> line")
    val = float(hits[-1])
    if val < 0.0 or val > 1.0:
        raise ValueError(f"map out of range [0,1]: {val}")
    return val


def mean_accuracy(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot mean empty accuracy list")
    return sum(values) / len(values)


def load_eval_accuracy(chip: str, bbdir: str, models: list[str]) -> dict[str, dict[str, Any]]:
    """Load [eval.accuracy.<model>] for each model. Missing section -> error."""
    import tomllib
    from pathlib import Path

    path = Path(bbdir) / "examples" / "chips" / chip / "regression" / "eval" / "models.toml"
    if not path.is_file():
        raise ValueError(f"eval models toml does not exist: {path}")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    eval_sec = data.get("eval")
    if not isinstance(eval_sec, dict):
        raise ValueError(f"missing [eval] section: {path}")
    acc_root = eval_sec.get("accuracy")
    if not isinstance(acc_root, dict):
        raise ValueError(f"missing [eval.accuracy] tables: {path}")
    out: dict[str, dict[str, Any]] = {}
    for name in models:
        spec = acc_root.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"missing [eval.accuracy.{name}]: {path}")
        metric = spec.get("metric")
        dataset = spec.get("dataset")
        if not isinstance(metric, str) or not metric:
            raise ValueError(f"[eval.accuracy.{name}].metric must be non-empty string")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"[eval.accuracy.{name}].dataset must be non-empty string")
        if dataset.startswith("/") or ".." in dataset.split("/"):
            raise ValueError(
                f"[eval.accuracy.{name}].dataset must be a relative path without '..': {dataset}"
            )
        out[name] = {"metric": metric, "dataset": dataset}
    return out
