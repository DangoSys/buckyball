#!/usr/bin/env python3
"""Show last-Da matadd dequant is wrong; shared-Da matches per-cin FP dequant."""

import numpy as np


def quant(x, da=None):
    if da is None:
        m = float(np.max(np.abs(x)))
        da = m / 127.0 if m > 0 else 1.0
    q = np.clip(np.rint(x / da), -128, 127).astype(np.int8)
    return q, da


def main():
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, 0.1, (8, 8)).astype(np.float32)
    x1 = rng.normal(0, 8.0, (8, 8)).astype(np.float32)
    w0 = rng.integers(-20, 21, (9, 16), dtype=np.int8)
    w1 = rng.integers(-20, 21, (9, 16), dtype=np.int8)
    dw = 0.05

    q0, da0 = quant(x0)
    q1, da1 = quant(x1)
    das = max(float(np.max(np.abs(x0))), float(np.max(np.abs(x1)))) / 127.0
    qs0, _ = quant(x0, das)
    qs1, _ = quant(x1, das)

    g0 = q0.reshape(-1)[:9].astype(np.int32) @ w0.astype(np.int32)
    g1 = q1.reshape(-1)[:9].astype(np.int32) @ w1.astype(np.int32)
    gs0 = qs0.reshape(-1)[:9].astype(np.int32) @ w0.astype(np.int32)
    gs1 = qs1.reshape(-1)[:9].astype(np.int32) @ w1.astype(np.int32)

    correct = g0.astype(np.float32) * da0 * dw + g1.astype(np.float32) * da1 * dw
    buggy = (g0 + g1).astype(np.float32) * da1 * dw
    shared = (gs0 + gs1).astype(np.float32) * das * dw

    bug_err = float(np.max(np.abs(buggy - correct)))
    shared_err = float(np.max(np.abs(shared - correct)))
    print(f"da0={da0:.6g} da1={da1:.6g} shared_da={das:.6g}")
    print(f"last_da_err={bug_err:.6g} shared_da_err={shared_err:.6g}")
    if bug_err <= shared_err * 2:
        raise SystemExit("expected last-Da path to be clearly worse")
    if shared_err > 0.5:
        raise SystemExit("shared-Da path too far from per-cin FP dequant")
    print("pytorch_shared_da_correlation PASSED")


if __name__ == "__main__":
    main()
