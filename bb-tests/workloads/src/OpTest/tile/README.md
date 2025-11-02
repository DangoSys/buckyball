# Tile Dialect Test Cases

## Test Files

### Stage-by-Stage Tests
- `tile-matmul.mlir` - Test Linalg → Tile conversion
- `tile-to-buckyball.mlir` - Test Tile → Buckyball conversion
- `buckyball-to-llvm.mlir` - Test Buckyball MatMulOp → LLVM Intrinsics conversion

### End-to-End Tests
- `end-to-end.mlir` - Complete conversion pipeline test (Linalg → Tile → Buckyball → LLVM)

## Running Tests

```bash
# Stage 1: Linalg → Tile
./compiler/build/bin/buddy-opt bb-tests/workloads/src/OpTest/tile/tile-matmul.mlir -convert-linalg-to-tile

# Stage 2: Tile → Buckyball
./compiler/build/bin/buddy-opt bb-tests/workloads/src/OpTest/tile/tile-to-buckyball.mlir -convert-tile-to-buckyball

# Stage 3: Buckyball → LLVM Intrinsics
./compiler/build/bin/buddy-opt bb-tests/workloads/src/OpTest/tile/buckyball-to-llvm.mlir \
  -lower-buckyball="dim=16 sp_addr_len=14 spad_rows=1024 acc_rows=1024 warp=16 lane=16"

# End-to-end test: Complete pipeline
./compiler/build/bin/buddy-opt bb-tests/workloads/src/OpTest/tile/end-to-end.mlir \
  -convert-linalg-to-tile \
  -convert-tile-to-buckyball \
  -lower-buckyball="dim=16 sp_addr_len=14 spad_rows=1024 acc_rows=1024 warp=16 lane=16"
```

## New Architecture Description

```
Linalg MatmulOp
    ↓ (convert-linalg-to-tile)
Tile TileMatMulOp
    ↓ (convert-tile-to-buckyball)
Buckyball MatMulOp
    ↓ (lower-buckyball)
LLVM Intrinsics (mvin/mvout/mul_warp16)
```
