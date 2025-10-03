# Tile Dialect 测试用例

## 测试文件

### 分阶段测试
- `tile-matmul.mlir` - 测试 Linalg → Tile 转换
- `tile-to-buckyball.mlir` - 测试 Tile → Buckyball 转换
- `buckyball-to-llvm.mlir` - 测试 Buckyball MatMulOp → LLVM Intrinsics 转换

### 端到端测试
- `end-to-end.mlir` - 完整的转换流程测试（Linalg → Tile → Buckyball → LLVM）

## 运行测试

```bash
# 阶段1: Linalg → Tile
./compiler/build/bin/buddy-opt bb-tests/workloads/src/OpTest/tile/tile-matmul.mlir -convert-linalg-to-tile

# 阶段2: Tile → Buckyball
./compiler/build/bin/buddy-opt bb-tests/workloads/src/OpTest/tile/tile-to-buckyball.mlir -convert-tile-to-buckyball

# 阶段3: Buckyball → LLVM Intrinsics
./compiler/build/bin/buddy-opt bb-tests/workloads/src/OpTest/tile/buckyball-to-llvm.mlir \
  -lower-buckyball="dim=16 sp_addr_len=14 spad_rows=1024 acc_rows=1024 warp=16 lane=16"

# 端到端测试：完整流程
./compiler/build/bin/buddy-opt bb-tests/workloads/src/OpTest/tile/end-to-end.mlir \
  -convert-linalg-to-tile \
  -convert-tile-to-buckyball \
  -lower-buckyball="dim=16 sp_addr_len=14 spad_rows=1024 acc_rows=1024 warp=16 lane=16"
```

## 新架构说明

```
Linalg MatmulOp
    ↓ (convert-linalg-to-tile)
Tile TileMatMulOp
    ↓ (convert-tile-to-buckyball)
Buckyball MatMulOp
    ↓ (lower-buckyball)
LLVM Intrinsics (mvin/mvout/mul_warp16)
```
