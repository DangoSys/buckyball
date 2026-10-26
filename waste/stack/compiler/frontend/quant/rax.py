from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

from stack.compiler.frontend.quant.bwq import BwqPackage, validate_bwq


def scale_count(shape: list[int], axes: list[int]) -> int:
    return math.prod(shape[axis] for axis in axes)


@dataclass
class QuantTensor:
    name: str
    shape: list[int]
    payload_shape: list[int]
    storage: str
    axes: list[int]
    payload_off: int
    payload_len: int
    scale_off: int
    scale_len: int
    layout: dict | None = None


@dataclass
class RaxQuantPackage:
    tensors: list[QuantTensor]
    weights: bytes
    params_f32: bytes
    scales_f32: bytes
    assets: dict[str, bytes]


def rax_from_bwq(pkg: BwqPackage) -> RaxQuantPackage:
    validate_bwq(pkg)
    tensors: list[QuantTensor] = []
    weights: list[bytes] = []
    scales: list[bytes] = []
    weight_off = scale_off = 0
    for source in pkg.tensors:
        if source.storage != "i8" or source.axes not in ([], [0]):
            raise ValueError(
                f"RAX W8A8 requires i8 tensor/channel weight: {source.name}"
            )
        weight = pkg.weights[source.weight_off : source.weight_off + source.weight_len]
        raw_scales = pkg.scales[source.scale_off : source.scale_off + source.scale_len]
        raw_count = 1 if not source.axes else source.shape[0]
        values = struct.unpack(f"<{raw_count}f", raw_scales)
        padded_count = scale_count(source.shape, source.axes)
        padded = values + (1.0,) * (padded_count - raw_count)
        scale = struct.pack(f"<{padded_count}f", *padded)
        tensors.append(
            QuantTensor(
                source.name,
                source.shape,
                source.shape,
                source.storage,
                source.axes,
                weight_off,
                len(weight),
                scale_off,
                len(scale),
            )
        )
        weights.append(weight)
        scales.append(scale)
        weight_off += len(weight)
        scale_off += len(scale)
    return RaxQuantPackage(tensors, b"".join(weights), b"", b"".join(scales), {})


def _numel(shape: list[int], name: str) -> int:
    n = 1
    for d in shape:
        if d <= 0:
            raise ValueError(f"bad shape for {name}")
        n *= d
    return n


def validate_rax_quant(pkg: RaxQuantPackage) -> None:
    if not pkg.tensors:
        raise ValueError("empty tensor list")
    names: set[str] = set()
    weight_end = param_end = scale_end = 0
    for tensor in pkg.tensors:
        if tensor.name in names:
            raise ValueError(f"duplicate tensor name: {tensor.name}")
        names.add(tensor.name)
        if tensor.storage not in ("i8", "mxfp8", "f32"):
            raise ValueError(f"unsupported storage for {tensor.name}: {tensor.storage}")
        if tensor.payload_off < 0 or tensor.payload_len < 0:
            raise ValueError(f"bad payload range for {tensor.name}")
        if tensor.scale_off < 0 or tensor.scale_len < 0:
            raise ValueError(f"bad scale range for {tensor.name}")
        numel = _numel(tensor.shape, tensor.name)
        payload_numel = _numel(tensor.payload_shape, tensor.name)
        if tensor.storage == "i8":
            if len(tensor.axes) > 1 or any(
                axis < 0 or axis >= len(tensor.shape) for axis in tensor.axes
            ):
                raise ValueError(
                    f"unsupported scale axes for {tensor.name}: {tensor.axes}"
                )
            if tensor.payload_off + tensor.payload_len > len(pkg.weights):
                raise ValueError(f"weight OOB for {tensor.name}")
            if tensor.payload_off != weight_end:
                raise ValueError(f"non-contiguous weight payload for {tensor.name}")
            if tensor.payload_len != payload_numel:
                raise ValueError(f"weight_len mismatch for {tensor.name}")
            if tensor.scale_off + tensor.scale_len > len(pkg.scales_f32):
                raise ValueError(f"scale OOB for {tensor.name}")
            scale_n = scale_count(tensor.shape, tensor.axes)
            if tensor.scale_len != scale_n * 4:
                raise ValueError(f"scale_len mismatch for {tensor.name}")
            if tensor.scale_off % 4:
                raise ValueError(f"unaligned scale offset for {tensor.name}")
            if tensor.scale_off != scale_end:
                raise ValueError(f"non-contiguous scale payload for {tensor.name}")
            scales = struct.unpack_from(
                f"<{scale_n}f", pkg.scales_f32, tensor.scale_off
            )
            if any(not math.isfinite(scale) or scale <= 0.0 for scale in scales):
                raise ValueError(f"invalid scale value for {tensor.name}")
            channel_count = tensor.shape[tensor.axes[0]] if tensor.axes else 1
            if tensor.axes and any(scale != 1.0 for scale in scales[channel_count:]):
                raise ValueError(f"bad channel-scale padding for {tensor.name}")
            weight_end += tensor.payload_len
            scale_end += tensor.scale_len
        elif tensor.storage == "mxfp8":
            if len(tensor.shape) != 2 or tensor.shape[1] % 32 or tensor.axes != [1]:
                raise ValueError(f"invalid MXFP8 shape or block axis: {tensor.name}")
            if len(tensor.payload_shape) == 2:
                stride = tensor.shape[1] + tensor.shape[1] // 32
                expected = tensor.shape[0] * stride
                if (
                    tensor.payload_shape != [tensor.shape[0], stride]
                    or tensor.layout != {"row_stride": stride}
                    or tensor.payload_len != expected
                ):
                    raise ValueError(
                        f"invalid MXFP8 embedding row layout: {tensor.name}"
                    )
            elif len(tensor.payload_shape) == 1:
                rows, k = tensor.layout["tile_n"], tensor.layout["tile_k"]
                stride = tensor.layout["panel_stride"]
                expected = (
                    ((tensor.shape[0] + rows - 1) // rows)
                    * ((tensor.shape[1] + k - 1) // k)
                    * stride
                )
                if (
                    rows % 16
                    or k % 32
                    or stride != tensor.layout["bank_bytes"]
                    or rows * k * 33 // 32 > stride
                    or tensor.payload_len != expected
                    or payload_numel != expected
                ):
                    raise ValueError(f"invalid MXFP8 packed layout: {tensor.name}")
            else:
                raise ValueError(f"invalid MXFP8 payload rank: {tensor.name}")
            if tensor.payload_off != weight_end or weight_end + expected > len(
                pkg.weights
            ):
                raise ValueError(f"invalid MXFP8 weight range: {tensor.name}")
            if tensor.scale_off or tensor.scale_len:
                raise ValueError("MXFP8 scales are embedded E8M0 bytes")
            weight_end += tensor.payload_len
        else:
            if tensor.axes or tensor.scale_off or tensor.scale_len:
                raise ValueError(f"f32 parameter has quant metadata: {tensor.name}")
            if tensor.payload_off + tensor.payload_len > len(pkg.params_f32):
                raise ValueError(f"f32 parameter OOB for {tensor.name}")
            if tensor.payload_off != param_end:
                raise ValueError(f"non-contiguous f32 payload for {tensor.name}")
            if tensor.payload_len != numel * 4:
                raise ValueError(f"f32 parameter size mismatch for {tensor.name}")
            param_end += tensor.payload_len
    if weight_end != len(pkg.weights):
        raise ValueError("unused weight payload")
    if param_end != len(pkg.params_f32):
        raise ValueError("unused FP32 parameter payload")
    if scale_end != len(pkg.scales_f32):
        raise ValueError("unused scale payload")


def _payload_dir(rax: Path) -> Path:
    return rax.parent / f"{rax.stem}.payload"


def _quant_index(pkg: RaxQuantPackage) -> dict:
    tensors = []
    for tensor in pkg.tensors:
        tensors.append(
            {
                "name": tensor.name,
                "shape": tensor.shape,
                "payload_shape": tensor.payload_shape,
                "storage": tensor.storage,
                "payload": "params_f32" if tensor.storage == "f32" else "weights",
                "payload_offset": tensor.payload_off,
                "payload_bytes": tensor.payload_len,
                "scale_offset": tensor.scale_off if tensor.storage == "i8" else 0,
                "scale_bytes": tensor.scale_len if tensor.storage == "i8" else 0,
                "scale_axes": tensor.axes,
                **(
                    {
                        "layout": tensor.layout,
                        "block_size": 32,
                        "scale_encoding": "e8m0",
                    }
                    if tensor.storage == "mxfp8"
                    else {}
                ),
            }
        )
    return {"version": 2, "tensors": tensors}


def _manifest(
    model_name: str,
    payload_name: str,
    weights_bytes: int,
    params_bytes: int,
    scales_bytes: int,
    index_bytes: int,
    assets: dict[str, bytes],
) -> str:
    asset_constants = "".join(
        f'  rhal.constant @{name.replace(".", "_")} {{id = {5 + i} : i32, storage = "external",\n'
        f"                             type = tensor<{len(data)}xi8>,\n"
        f'                             uri = "file:{payload_name}/{name}"}}\n'
        for i, (name, data) in enumerate(assets.items())
    )
    return f"""rhal.module @quant_{model_name} attributes {{
    version = "0.1.0",
    model_name = "{model_name}"}} {{
  rhal.constant @weights {{id = 1 : i32, storage = "external",
                             type = tensor<{weights_bytes}xi8>,
                             uri = "file:{payload_name}/weights.bin"}}
  rhal.constant @params_f32 {{id = 2 : i32, storage = "external",
                             type = tensor<{params_bytes}xi8>,
                             uri = "file:{payload_name}/params.f32"}}
  rhal.constant @scales_f32 {{id = 3 : i32, storage = "external",
                              type = tensor<{scales_bytes}xi8>,
                              uri = "file:{payload_name}/scales.bin"}}
  rhal.constant @quant_index {{id = 4 : i32, storage = "external",
                               type = tensor<{index_bytes}xi8>,
                               uri = "file:{payload_name}/quant-index.json"}}
{asset_constants}}}
"""


def write_payload(pkg: RaxQuantPackage, payload_dir: Path | str) -> None:
    validate_rax_quant(pkg)
    payload_dir = Path(payload_dir)
    if payload_dir.exists():
        shutil.rmtree(payload_dir)
    payload_dir.mkdir(parents=True)
    index = json.dumps(_quant_index(pkg), sort_keys=True, indent=2).encode("ascii")
    (payload_dir / "weights.bin").write_bytes(pkg.weights)
    (payload_dir / "params.f32").write_bytes(pkg.params_f32)
    (payload_dir / "scales.bin").write_bytes(pkg.scales_f32)
    (payload_dir / "quant-index.json").write_bytes(index)
    for name, data in pkg.assets.items():
        (payload_dir / name).write_bytes(data)


def write_rax(
    pkg: RaxQuantPackage, rax: Path | str, rax_pack: Path | str, model_name: str
) -> None:
    rax = Path(rax)
    rax_pack = Path(rax_pack)
    if not rax_pack.is_file():
        raise ValueError(f"rax-pack not found: {rax_pack}")
    payload_dir = _payload_dir(rax)
    write_payload(pkg, payload_dir)
    index = (payload_dir / "quant-index.json").read_bytes()
    manifest = payload_dir / "quant.rhal.mlir"
    manifest.write_text(
        _manifest(
            model_name,
            payload_dir.name,
            len(pkg.weights),
            len(pkg.params_f32),
            len(pkg.scales_f32),
            len(index),
            pkg.assets,
        ),
        encoding="ascii",
    )
    subprocess.run(
        [str(rax_pack), str(manifest), "-o", str(rax), "--embed-payload"],
        check=True,
    )
