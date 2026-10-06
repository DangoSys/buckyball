import math

import numpy as np


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
    rows, width = codes.shape
    n, k = layout["tile_n"], layout["tile_k"]
    stride = layout["panel_stride"]
    if width % 32 or scales.shape != (rows, width // 32):
        raise ValueError("MXFP8 codes and scales disagree")
    panels = []
    for begin_row in range(0, rows, n):
        for begin_k in range(0, width, k):
            count = min(k, width - begin_k)
            plane = np.zeros((n, count), dtype=np.uint8)
            scale = np.full((n, count // 32), 127, dtype=np.uint8)
            valid = min(n, rows - begin_row)
            plane[:valid] = codes[
                begin_row : begin_row + valid, begin_k : begin_k + count
            ]
            scale[:valid] = scales[
                begin_row : begin_row + valid, begin_k // 32 : (begin_k + count) // 32
            ]
            payload = np.concatenate((plane.ravel(), scale.ravel()))
            if payload.size > stride:
                raise ValueError("MXFP8 panel exceeds its bank window")
            panels.append(np.pad(payload, (0, stride - payload.size)))
    return np.concatenate(panels)


def bank_bytes(compiler_build, target):
    import re
    from pathlib import Path

    text = (
        Path(compiler_build) / "external_dialects/BuckyballTargetRegistry.inc"
    ).read_text()
    configs = {
        name: int(width) * int(depth) // 8
        for name, width, depth in re.findall(
            r'\{"([^"]+)", "[^"]+", \d+, (\d+), (\d+), llvm::ArrayRef', text
        )
    }
    return configs[target]
