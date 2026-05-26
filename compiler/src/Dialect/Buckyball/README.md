# Buckyball Dialect

Buckyball Dialect 是 RISC-V Buckyball NPU 扩展的 MLIR 表示，采用三层抽象：High-level Ops → Bank SSA → Intrinsics。

## 三层抽象

### 1. High-level Ops（算子层）
完整的硬件算子，输入为 bank 宽度（16）× 任意深度的 memref。

- **`buckyball.matmul`**：矩阵乘法（A[M][K] × B[K][N] → C[M][N]，K/N 必须是 16 的倍数）
- **`buckyball.transpose`**：矩阵转置
- **`buckyball.im2col`**：Im2col 变换（用于卷积）

**Lowering**（`-lower-buckyball-to-bank-ssa`）：展开为 Bank SSA ops，包含完整的数据流：
```
mvin(A_fp32) → quant(A_i8) → mul_warp16 → dequant(C_fp32) → mvout(C)
```

### 2. Bank SSA Ops（虚拟 bank 层）
使用虚拟 bank handle 的 SSA 形式操作，类似 LLVM 虚拟寄存器。

- **`buckyball.bank_alloc(row, col) -> i64`**：分配虚拟 bank，返回 handle
  - `row × col` 表示需要的连续 bank 数量（如 `col=4` 表示 fp32 需要 4 个 bank）
- **`buckyball.bank_mvin(memref, bank, depth, stride) -> i64`**：DRAM → bank
- **`buckyball.bank_mvout(memref, bank, depth, stride) -> i64`**：bank → DRAM
- **`buckyball.bank_mul_warp16(op1, op2, wr, iter, mode) -> i64`**：16×16 systolic array 矩阵乘
- **`buckyball.bank_quant / bank_dequant`**：量化/反量化
- **`buckyball.bank_transpose / bank_im2col`**：转置/Im2col
- **`buckyball.bank_release(bank)`**：释放虚拟 bank

**设计意图**：
- 前端不关心物理 bank 数量限制（默认 16 个）
- 编译器在 `-assign-physical-banks` pass 中统一做寄存器分配
- 支持 bank 复用优化（release 后可重新分配）

**Lowering**（`-assign-physical-banks`）：
1. 遍历所有 `bank_alloc`，贪心分配物理 bank ID（first-fit）
2. 如果超出 16 个 bank，报错 `out of physical banks`
3. 将虚拟 handle 替换为物理 bank ID 常量
4. 插入 `mset(bankId, alloc=true/false)` 标记 bank 分配/释放

### 3. Intrinsic Ops（物理 bank 层）
直接对应 RISC-V Buckyball 指令的 MLIR wrapper。

- **`buckyball.mset(bankId, alloc, row, col)`**：分配/释放物理 bank（funct7=32）
- **`buckyball.mvin(memref, bankId, depth, stride)`**：DRAM → bank（funct7=33）
- **`buckyball.mvout(memref, bankId, depth, stride)`**：bank → DRAM（funct7=16）
- **`buckyball.mul_warp16(op1, op2, wr, iter, mode)`**：16×16 矩阵乘（funct7=64）
- **`buckyball.transpose_intr / quant_intr / dequant_intr / im2col_intr`**：其他 Ball 指令
- **`buckyball.fence()`**：内存屏障（funct7=0）

**Lowering**（`-lower-buckyball`）：转换为 LLVM intrinsics（`@llvm.riscv.bb_mset` 等）

## Ball 架构

Buckyball 是可组合的 NPU 框架，每个 Ball 是独立的计算单元：

- **VecBall (bid=0)**：`mul_warp16` — 16×16 systolic array
- **ReluBall (bid=1)**：`relu` — ReLU 激活
- **TransposeBall (bid=2)**：`transpose` — 矩阵转置
- **Im2colBall (bid=3)**：`im2col` — Im2col 变换
- **SystolicArrayBall (bid=4)**：`systolic` — BBFP 乘法
- **Fp2IntBall (bid=5)**：`quant` — 量化
- **Int2FpBall (bid=6)**：`dequant` — 反量化

每个 Ball 独立开发，通过 bank 接口组合。

## 硬件约束

- **Bank 数量**：16 个物理 bank（可通过 `-assign-physical-banks -bank_num=N` 调整）
- **Bank 宽度**：16 字节（128 bit）
- **Bank 深度**：1024 行（i8 模式）/ 256 行（i32 accumulator 模式）
- **Bank 大小**：16KB per bank（16 × 1024）
- **Systolic array**：16×16 PE array

## 常见错误

### `assign-physical-banks: out of physical banks`
**原因**：同时存活的虚拟 bank 数量超过 16
**解决**：
1. 减小输入 tile 尺寸（在 Tile Dialect 层调整）
2. 优化 bank 生命周期（提前 `bank_release`）
3. 增加物理 bank 数量（修改硬件参数）

### `K and N must be multiples of 16`
**原因**：`buckyball.matmul` 要求 K、N 对齐到 bank 宽度
**解决**：在 Tile Dialect 层自动 padding（`tile.tile_matmul` 已处理）

## Pass 顺序

完整的 lowering pipeline：
```
linalg.matmul
  ↓ -convert-linalg-to-tile
tile.tile_matmul
  ↓ -convert-tile-to-buckyball
buckyball.matmul
  ↓ -lower-buckyball-to-bank-ssa
buckyball.bank_alloc + bank_mvin + bank_mul_warp16 + ...
  ↓ -assign-physical-banks
buckyball.mset + mvin + mul_warp16 + ...
  ↓ -expand-strided-metadata -convert-scf-to-cf -llvm-request-c-wrappers
  ↓ -lower-buckyball -convert-arith-to-llvm -finalize-memref-to-llvm
  ↓ -convert-func-to-llvm -reconcile-unrealized-casts
LLVM IR (with @llvm.riscv.bb_* intrinsics)
  ↓ buddy-translate --buddy-to-llvmir
  ↓ buddy-llc -mattr=+buddyext
RISC-V assembly (bb_mset / bb_mvin / bb_mul_warp16 / ...)
```
