#!/usr/bin/env python3
"""Correlate ResNet-stem-like 2-cin INT32 accumulate with torch.nn.functional.conv2d."""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    raise SystemExit(f"pytorch required: {e}")


def cdiv(a, b):
    return (a + b - 1) // b


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # Match matadd_after_smatmul_os_test geometry loosely via conv:
    # nCin=2, k=7, tile=14 -> wins=196, but gemm pads M to 208, K to 64.
    nCin, ksize, tile, stride = 2, 7, 14, 2
    in_size = (tile - 1) * stride + ksize
    oc = 16
    x = torch.randint(-30, 31, (1, nCin, in_size, in_size), dtype=torch.int8)
    w = torch.randint(-23, 24, (oc, nCin, ksize, ksize), dtype=torch.int8)

    y = F.conv2d(x.float(), w.float(), stride=stride)
    assert tuple(y.shape) == (1, oc, tile, tile)

    # Per-cin im2col + INT32 gemm + matadd (pad K to 64)
    padded_k = cdiv(ksize * ksize, 16) * 16
    acc = np.zeros((tile * tile, oc), dtype=np.int32)
    for ci in range(nCin):
        cols = []
        for oh in range(tile):
            for ow in range(tile):
                ih = oh * stride
                iw = ow * stride
                patch = x[0, ci, ih : ih + ksize, iw : iw + ksize].reshape(-1).numpy()
                pad = np.zeros(padded_k, dtype=np.int8)
                pad[: ksize * ksize] = patch
                cols.append(pad)
        A = np.stack(cols, 0)
        B = np.zeros((padded_k, oc), dtype=np.int8)
        B[: ksize * ksize] = w[:, ci].reshape(oc, -1).T.numpy()
        acc += A.astype(np.int32) @ B.astype(np.int32)

    ref = y[0].permute(1, 2, 0).reshape(-1, oc).numpy().astype(np.int32)
    md = int(np.max(np.abs(acc - ref)))
    print(f"torch_vs_matadd_equiv maxdiff={md}")
    if md != 0:
        raise SystemExit("matadd-equivalent path diverges from torch.conv2d")
    print("pytorch_correlation PASSED")


if __name__ == "__main__":
    main()
