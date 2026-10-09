import math

import torch


def plan(rows, columns, reduction, bank_bytes, *, mxfp8):
    alignment = 32 if mxfp8 else 4
    if min(rows, columns, reduction, bank_bytes) <= 0 or reduction % alignment:
        raise ValueError("matmul shape violates the format alignment")
    row_limit = 1 if rows == 1 else (rows + 15) // 16 * 16
    column_limit = (columns + 15) // 16 * 16
    candidates = []
    for m in ([1] if rows == 1 else range(16, row_limit + 1, 16)):
        for n in range(16, column_limit + 1, 16):
            if m * n * 4 > bank_bytes:
                continue
            factor = 33 / 32 if mxfp8 else 4
            k = min(reduction, int(bank_bytes / (max(m, n) * factor)))
            k = k // alignment * alignment
            if not k:
                continue
            panels = (
                math.ceil(rows / m) * math.ceil(columns / n) * math.ceil(reduction / k)
            )
            quant_panels = (
                math.ceil(rows / m) * math.ceil(reduction / k) if mxfp8 else 0
            )
            candidates.append(((panels + quant_panels, panels, -k, -m, -n), (m, n, k)))
    if not candidates:
        raise ValueError("matmul cannot fit an aligned block in its banks")
    m, n, k = min(candidates)[1]
    return {
        "tile_m": m,
        "tile_n": n,
        "tile_k": k,
        "bank_bytes": bank_bytes,
        "panel_stride": bank_bytes,
    }


def pack(codes, scales, layout):
    if (
        not isinstance(codes, torch.Tensor)
        or not isinstance(scales, torch.Tensor)
        or codes.dtype != torch.int8
        or scales.dtype != torch.int8
    ):
        raise ValueError("MXFP8 packing requires INT8 Tensor codes and scales")
    if codes.ndim != 2 or scales.device != codes.device:
        raise ValueError("MXFP8 packing requires a matrix and matching devices")
    rows, width = codes.shape
    n, k = layout["tile_n"], layout["tile_k"]
    stride = layout["panel_stride"]
    if width % 32 or scales.shape != (rows, width // 32):
        raise ValueError("MXFP8 codes and scales disagree")
    panels = []
    for begin_row in range(0, rows, n):
        for begin_k in range(0, width, k):
            count = min(k, width - begin_k)
            plane = torch.zeros((n, count), dtype=torch.int8, device=codes.device)
            scale = torch.full(
                (n, count // 32), 127, dtype=torch.int8, device=codes.device
            )
            valid = min(n, rows - begin_row)
            plane[:valid] = codes[
                begin_row : begin_row + valid, begin_k : begin_k + count
            ]
            scale[:valid] = scales[
                begin_row : begin_row + valid, begin_k // 32 : (begin_k + count) // 32
            ]
            payload = torch.cat((plane.flatten(), scale.flatten()))
            if payload.numel() > stride:
                raise ValueError("MXFP8 panel exceeds its bank window")
            panels.append(
                torch.nn.functional.pad(payload, (0, stride - payload.numel()))
            )
    return torch.cat(panels)


def bank_bytes(compiler_build, target):
    import re
    from pathlib import Path

    text = (
        Path(compiler_build) / "external_dialects/BuckyballTargetRegistry.inc"
    ).read_text()
    configs = {
        name: int(width) * int(depth) // 8
        for name, width, depth in re.findall(
            r'\{"([^"]+)", "[^"]+", \d+, (\d+), (\d+), \d+, (?:true|false), llvm::ArrayRef',
            text,
        )
    }
    if target not in configs:
        raise ValueError(
            f"Target {target} has no registered bank geometry in {compiler_build}"
        )
    return configs[target]
