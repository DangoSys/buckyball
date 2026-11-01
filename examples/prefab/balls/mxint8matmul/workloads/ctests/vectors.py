#!/usr/bin/env python3
"""Writes vectors.h: the bank images of the mxint8matmul C tests. A and B are
packed by quantize_mxint8 of stack/compiler/frontend/quant.py; the expected C
follows the Ball's accumulation order: per element, block by block,
t = fp32(p * 2^(ea + eb)) and acc = fp32(acc + t) from +0.0 or the old C.

    nix develop -c python3 examples/prefab/balls/mxint8matmul/workloads/ctests/vectors.py

A test moves <case>_a, <case>_b (and with accumulation <case>_c, the old C)
into three banks, runs one MXINT8MM and moves the C bank back; its window
(from bank row <CASE>_C_BASE) must equal <case>_expected bit for bit. The
output is a new version of its bank: rows outside the window are undefined.
"""

import sys
from pathlib import Path

import numpy as np
import torch

here = Path(__file__).resolve().parent
repo = here.parents[5]
sys.path[:0] = [
    str(repo / "stack/thirdparty/buddy-mlir/build/python_packages"),
    str(repo / "stack"),
]
from compiler.frontend.quant import quantize_mxint8  # noqa: E402


def packed_rows(k):
    return k // 16 + (k + 511) // 512


def image(packed, k, base):
    """Bank image of packed MXINT8 rows from bank row `base`; the rows
    before `base` hold 0x80 codes the Ball must not read."""
    rows, length = packed.shape
    stride = packed_rows(k) * 16
    out = np.full((base * 16 + rows * stride,), 0x80, np.uint8)
    for row in range(rows):
        start = base * 16 + row * stride
        out[start : start + stride] = 0
        out[start : start + length] = packed[row]
    return out


def product(a, b, k, old):
    codes_a = a[:, :k].view(np.int8).astype(np.int64)
    codes_b = b[:, :k].view(np.int8).astype(np.int64)
    exp_a, exp_b = a[:, k:].astype(np.int64), b[:, k:].astype(np.int64)
    c = old.copy()
    for m in range(a.shape[0]):
        for n in range(b.shape[0]):
            acc = c[m, n]
            for block in range(k // 32):
                lanes = slice(block * 32, block * 32 + 32)
                p = int((codes_a[m, lanes] * codes_b[n, lanes]).sum())
                scale = int(exp_a[m, block] + exp_b[n, block] - 254)
                term = np.float32(np.float64(p) * np.ldexp(1.0, scale))
                acc = np.float32(acc + term)
                assert np.isfinite(term) and np.isfinite(acc)
            c[m, n] = acc
    return c


def array(ctype, name, data):
    words = ", ".join(f"0x{int(v):x}" for v in data)
    return f"static const {ctype} {name}[{len(data)}] __attribute__((aligned(64))) = {{{words}}};\n"


def pack(values):
    return (
        quantize_mxint8(torch.from_numpy(values), -1).view(torch.uint8).numpy().copy()
    )


def case(name, pa, pb, k, bases, old=None):
    m, n = pa.shape[0], pb.shape[0]
    initial = np.zeros((m, n), np.float32) if old is None else old
    expected = product(pa, pb, k, initial)
    # NaN before the old C: the Ball must not read it.
    c_image = np.full(bases[2] * 4 + m * n, np.nan, np.float32)
    c_image[bases[2] * 4 :] = initial.reshape(-1)
    upper = name.upper()
    return (
        f"#define {upper}_M {m}\n#define {upper}_N {n}\n#define {upper}_K {k}\n"
        f"#define {upper}_A_BASE {bases[0]}\n#define {upper}_B_BASE {bases[1]}\n"
        f"#define {upper}_C_BASE {bases[2]}\n#define {upper}_ACCUMULATE {int(old is not None)}\n"
        f"#define {upper}_A_ROWS {bases[0] + m * packed_rows(k)}\n"
        f"#define {upper}_B_ROWS {bases[1] + n * packed_rows(k)}\n"
        f"#define {upper}_C_ROWS {bases[2] + m * n // 4}\n"
        + array("uint8_t", f"{name}_a", image(pa, k, bases[0]))
        + array("uint8_t", f"{name}_b", image(pb, k, bases[1]))
        + array("uint32_t", f"{name}_c", c_image.view(np.uint32))
        + array("uint32_t", f"{name}_expected", expected.reshape(-1).view(np.uint32))
    )


def normal(rng, rows, k, spread):
    scale = np.exp2(rng.integers(-spread, spread, (rows, k // 32, 1))).astype(
        np.float32
    )
    return (
        (rng.standard_normal((rows, k // 32, 32)) * scale)
        .astype(np.float32)
        .reshape(rows, k)
    )


def extremes():
    rng = np.random.default_rng(5)
    a = normal(rng, 3, 64, 2)
    b = normal(rng, 4, 64, 2)
    # Row 0: subnormal elements, block exponent -127: subnormal and zero terms.
    a[0] = (rng.standard_normal(64) * 1e-39).astype(np.float32)
    b[0] = (rng.standard_normal(64) * 1e-39).astype(np.float32)
    b[1] = (rng.standard_normal(64) * 1e-30).astype(np.float32)
    # Row 1: two blocks with opposite products cancel to +0.
    a[1, 32:] = -a[1, :32]
    b[2, 32:] = b[2, :32]
    # Row 2 x B row 3: products near the top of the FP32 range.
    a[2] = (rng.standard_normal(64) * 1e18).astype(np.float32)
    b[3] = (rng.standard_normal(64) * 1e18).astype(np.float32)
    pa, pb = pack(a), pack(b)
    # Block 1 of A row 1 and B row 0: every code -128, which quant.py never
    # writes; the block sum is 32 * 128 * 128 = 2^19.
    pa[1, 32:64] = 0x80
    pb[0, 32:64] = 0x80
    return pa, pb


def main():
    rng = np.random.default_rng(3)
    text = (
        "/* Generated by vectors.py. Do not edit. */\n#pragma once\n#include <stdint.h>\n"
        + case(
            "mm_small",
            pack(normal(rng, 3, 64, 8)),
            pack(normal(rng, 8, 64, 8)),
            64,
            (1, 2, 3),
        )
        + case(
            "mm_accumulate",
            pack(normal(rng, 5, 96, 20)),
            pack(normal(rng, 12, 96, 20)),
            96,
            (0, 0, 0),
            (rng.standard_normal((5, 12)) * 1e3).astype(np.float32),
        )
        + case("mm_extremes", *extremes(), 64, (0, 0, 0))
        + case(
            "mm_long",
            pack(normal(rng, 6, 544, 6)),
            pack(normal(rng, 12, 544, 6)),
            544,
            (3, 1, 2),
            (rng.standard_normal((6, 12)) * 10).astype(np.float32),
        )
    )
    (here / "vectors.h").write_text(text)


if __name__ == "__main__":
    main()
