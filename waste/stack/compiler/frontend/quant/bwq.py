from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TensorSpec:
    name: str
    shape: list[int]
    storage: str
    axes: list[int]
    weight_off: int
    weight_len: int
    scale_off: int
    scale_len: int


@dataclass
class BwqPackage:
    version: int
    tensors: list[TensorSpec]
    weights: bytes
    scales: bytes


_FORBIDDEN = ("groupSize", "group_size", "zeroPoint", "zero_point")
_STORAGE = {"i8"}


def _numel(shape: list[int], name: str) -> int:
    n = 1
    for d in shape:
        if d <= 0:
            raise ValueError(f"bad shape for {name}")
        n *= d
    return n


def validate_bwq(pkg: BwqPackage) -> None:
    if pkg.version != 1:
        raise ValueError(f"unsupported version: {pkg.version}")
    if not pkg.tensors:
        raise ValueError("empty tensor list")
    seen: set[str] = set()
    for t in pkg.tensors:
        if t.name in seen:
            raise ValueError(f"duplicate tensor name: {t.name}")
        seen.add(t.name)
        if t.storage not in _STORAGE:
            raise ValueError(f"unsupported storage {t.storage!r} for {t.name}")
        if t.weight_off < 0 or t.weight_len < 0:
            raise ValueError(f"bad weight range for {t.name}")
        if t.weight_off + t.weight_len > len(pkg.weights):
            raise ValueError(f"weight OOB for {t.name}")
        if t.scale_off < 0 or t.scale_len < 0:
            raise ValueError(f"bad scale range for {t.name}")
        if t.scale_off + t.scale_len > len(pkg.scales):
            raise ValueError(f"scale OOB for {t.name}")
        n = _numel(t.shape, t.name)
        for ax in t.axes:
            if ax < 0 or ax >= len(t.shape):
                raise ValueError(f"axes OOB for {t.name}")
        if t.weight_len != n:
            raise ValueError(f"weight_len {t.weight_len} != numel {n} for {t.name}")
        if t.axes:
            scale_n = 1
            for ax in t.axes:
                scale_n *= t.shape[ax]
            if t.scale_len != scale_n * 4:
                raise ValueError(f"scale_len mismatch for {t.name}")
        elif t.scale_len != 4:
            raise ValueError(f"per-tensor scale_len must be 4 for {t.name}")


def write_bwq(pkg: BwqPackage, path: Path) -> None:
    validate_bwq(pkg)
    path = Path(path)
    if path.exists() and not path.is_dir():
        raise ValueError(f"not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    lines = [f"version = {pkg.version}", ""]
    for t in pkg.tensors:
        lines += [
            "[[tensor]]",
            f'name = "{t.name}"',
            f"shape = [{', '.join(str(x) for x in t.shape)}]",
            f'storage = "{t.storage}"',
            f"axes = [{', '.join(str(x) for x in t.axes)}]",
            f"weight_off = {t.weight_off}",
            f"weight_len = {t.weight_len}",
            f"scale_off = {t.scale_off}",
            f"scale_len = {t.scale_len}",
            "",
        ]
    (path / "manifest.toml").write_text("\n".join(lines), encoding="utf-8")
    (path / "weights.bin").write_bytes(pkg.weights)
    (path / "scales.bin").write_bytes(pkg.scales)


def read_bwq(path: Path) -> BwqPackage:
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"bwq path is not a directory: {path}")
    manifest = path / "manifest.toml"
    if not manifest.is_file():
        raise ValueError(f"missing manifest.toml in {path}")
    try:
        meta = tomllib.loads(manifest.read_bytes().decode("utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"invalid manifest.toml: {e}") from e
    except UnicodeDecodeError as e:
        raise ValueError(f"invalid manifest.toml encoding: {e}") from e
    for key in _FORBIDDEN:
        if key in meta:
            raise ValueError(f"forbidden key at top level: {key}")
    if "version" not in meta:
        raise ValueError("missing field version")
    tensors: list[TensorSpec] = []
    for entry in meta.get("tensor", []):
        for key in _FORBIDDEN:
            if key in entry and entry[key] is not None:
                raise ValueError(f"forbidden key {key} on tensor {entry.get('name')}")
        need = (
            "name",
            "shape",
            "storage",
            "axes",
            "weight_off",
            "weight_len",
            "scale_off",
            "scale_len",
        )
        for k in need:
            if k not in entry:
                raise ValueError(f"missing field {k}")
        tensors.append(
            TensorSpec(
                name=entry["name"],
                shape=list(entry["shape"]),
                storage=entry["storage"],
                axes=list(entry["axes"]),
                weight_off=int(entry["weight_off"]),
                weight_len=int(entry["weight_len"]),
                scale_off=int(entry["scale_off"]),
                scale_len=int(entry["scale_len"]),
            )
        )
    weights_path = path / "weights.bin"
    scales_path = path / "scales.bin"
    if not weights_path.is_file():
        raise ValueError(f"missing weights.bin in {path}")
    if not scales_path.is_file():
        raise ValueError(f"missing scales.bin in {path}")
    pkg = BwqPackage(
        version=int(meta["version"]),
        tensors=tensors,
        weights=weights_path.read_bytes(),
        scales=scales_path.read_bytes(),
    )
    validate_bwq(pkg)
    return pkg
