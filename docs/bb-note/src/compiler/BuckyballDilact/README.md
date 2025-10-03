# Tile Dialect 重构文档

## 重构背景与目标

本次重构的核心目标是在 Linalg Dialect 和 Buckyball Dialect 之间引入一个新的中间层 —— Tile Dialect，以实现更清晰的职责分离和更好的代码组织。在原有架构中，从 `linalg.matmul` 到硬件指令的转换是通过 `convert-linalg-to-buckyball` 一步完成的，这导致 Buckyball Dialect 既要处理任意尺寸矩阵的切片逻辑，又要处理硬件级别的内存管理和计算调度，职责过于混杂。新架构将转换过程拆分为两个阶段：`convert-linalg-to-tile` 和 `convert-tile-to-buckyball`，使得每一层都有明确且单一的职责。

## 新架构设计

整个编译流程现在分为三个清晰的层次。首先是 Linalg 层，它表示高层次的线性代数操作，比如 `linalg.matmul` 表示任意尺寸的矩阵乘法，这一层不关心硬件约束。接下来是新引入的 Tile 层，它的核心职责是将任意尺寸的矩阵操作切分成符合硬件约束的固定大小块。Tile 层通过 `tile.tile_matmul` 操作来表示这种高层次的切片意图，具体的切片策略、循环生成和边界处理都在 `convert-tile-to-buckyball` 这个 pass 中实现。最后是 Buckyball 层，它专注于硬件级别的操作，`buckyball.bb_matmul` 接收已经切分好的固定大小矩阵块，负责生成精确的硬件指令序列，包括数据搬移（mvin/mvout）、计算调度（mul_warp16）和内存地址计算。

## Tile Dialect 设计细节

Tile Dialect 定义了 `TileMatMulOp` 操作，它接受三个 memref 参数分别代表矩阵 A、B 和 C。这个操作的语义是：对输入的任意尺寸矩阵执行乘法，自动处理切片、填充和循环。在实现上，`TileMatMulOp` 会被 `convert-tile-to-buckyball` pass 转换成多个 `buckyball.bb_matmul` 操作以及相应的 `memref.subview` 操作。这个转换过程会考虑硬件的 scratchpad 大小限制、warp 和 lane 的并行度约束，生成最优的切片策略。Tile 层的设计理念是提供一个平台无关的中间表示，使得上层优化可以在不了解具体硬件细节的情况下进行矩阵操作的变换。

## Buckyball Dialect 简化

在新架构中，Buckyball Dialect 被大幅简化。原来的 `VecTileMatMulOp`、`MergeTileMatMulOp`、`MetaTileMatMulOp` 和 `VecMulWarp16Op` 这四个操作被统一为单一的 `MatMulOp`。这个简化是合理的，因为切片逻辑已经上移到 Tile 层，Buckyball 层只需要表达"对一个已经符合硬件约束的矩阵块执行硬件级乘法"这一个概念。`buckyball.bb_matmul` 的 lowering 过程会直接生成 LLVM intrinsics：首先通过 `Mvin_IntrOp` 将 A 和 B 矩阵加载到 scratchpad，然后根据 warp 和 lane 参数生成多个 `Mul_Warp16_IntrOp` 进行计算，最后通过 `Mvout_IntrOp` 将结果写回主内存。所有的地址计算、编码都在这个 lowering 过程中完成。

## 关键实现细节

在实现 `convert-linalg-to-tile` pass 时，核心逻辑非常简单：匹配 `linalg.matmul` 操作，直接替换为 `tile.tile_matmul`，传递相同的三个 memref 操作数。这个 pass 的作用主要是类型和语义的转换，表明我们从通用的线性代数操作域进入了面向硬件的 tile 操作域。

`convert-tile-to-buckyball` pass 是整个重构中最复杂的部分。它需要从 `tile.tile_matmul` 的操作数中提取矩阵的维度信息（M、K、N），然后根据硬件参数（dim、warp、lane）计算最优的切片策略。对于 K 维度，会按照 warp 大小进行切片；对于 M 和 N 维度，会考虑 scratchpad 的容量限制。每个切片对应一个 `buckyball.bb_matmul` 操作，切片之间通过 `memref.subview` 来创建矩阵的视图。特别需要注意的是边界情况的处理：当矩阵维度不能被切片大小整除时，需要计算最后一个切片的实际大小，避免越界访问。

在实现 `BuckyballMatMulLowering` 时，我们遇到了 MLIR 类型转换系统的一个重要概念：OpAdaptor。在 conversion pattern 中，原始操作的类型（比如 `memref<32x16xi8>`）在 lowering 过程中会被 TypeConverter 转换为 LLVM 类型（比如 LLVM 的 struct 类型）。OpAdaptor 提供的是转换后的值，而我们需要从原始操作中获取类型信息（比如 shape），因为这些静态信息在转换后可能不再以相同形式存在。因此，正确的做法是：从 `matMulOp.getOperandTypes()` 获取原始的 `MemRefType` 来提取 shape 信息，用于地址计算和循环生成；而对于实际的值操作（比如 `ExtractAlignedPointerAsIndexOp`），则使用原始的 memref value，因为 MLIR 的 memref 操作仍然需要 MemRefType。

另一个关键的设计决策是：`MatMulOp` 的 lowering 应该直接生成 intrinsic operations（`Mvin_IntrOp`、`Mul_Warp16_IntrOp`、`Mvout_IntrOp`），而不是生成 `MvinOp`、`MvoutOp` 然后再等它们被 lower。这样做的原因是在 LLVM lowering 阶段，类型系统已经发生了转换，再创建高层次的 Buckyball 操作会导致类型不匹配的问题。直接生成 intrinsics 避免了多次类型转换，也使得代码更加清晰高效。参考 Gemmini dialect 的实现，我们采用了相同的策略。

## 测试体系

为了验证新架构的正确性，我们在 `bb-tests/workloads/src/OpTest/tile/` 目录下创建了完整的测试用例。测试分为两类：分阶段测试和端到端测试。

`tile-matmul.mlir` 测试 Linalg 到 Tile 的转换，验证 `linalg.matmul` 是否正确转换为 `tile.tile_matmul`，这是最基础的类型转换测试。`tile-to-buckyball.mlir` 测试 Tile 到 Buckyball 的转换，验证切片逻辑是否正确，是否生成了正确数量的 `buckyball.bb_matmul` 操作和 `memref.subview` 操作。`buckyball-to-llvm.mlir` 测试 Buckyball MatMulOp 到 LLVM intrinsics 的转换，验证是否生成了正确的 `buckyball.intr.bb_mvin`、`buckyball.intr.bb_mul_warp16` 和 `buckyball.intr.bb_mvout` 指令序列。

`end-to-end.mlir` 是最重要的测试，它测试完整的转换流程：从 `linalg.matmul` 开始，依次经过 `-convert-linalg-to-tile`、`-convert-tile-to-buckyball`、`-lower-buckyball` 三个 pass，最终生成 LLVM intrinsics。这个测试确保整个 pipeline 的每个环节都能正常工作，并且环节之间的衔接没有问题。

## Pass 注册与工具链集成

新增的两个 pass 需要在多个地方进行注册。首先在 `InitAll.cpp` 中注册 pass 的创建函数 `registerLowerLinalgToTilePass()` 和 `registerLowerTileToBuckyballPass()`，同时注册 `buddy::tile::TileDialect`。在 `buddy-opt` 工具中，需要在 dialect registry 中添加 `buddy::tile::TileDialect`，使得工具能够识别和解析 tile dialect 的操作。CMake 构建系统中，需要将新的库 `BuddyTile`、`LowerLinalgToTilePass`、`LowerTileToBuckyballPass` 添加到链接依赖中，并确保依赖关系正确。

特别值得注意的是，在 `LegalizeForLLVMExport.cpp` 的 `configureBuckyballLegalizeForExportTarget` 函数中，我们需要添加 `target.addLegalDialect<memref::MemRefDialect>()` 和 `target.addLegalDialect<arith::ArithDialect>()`，因为在 `MatMulOp` 的 lowering 过程中会使用 memref 和 arith 操作。如果不将这些 dialect 标记为 legal，conversion framework 会尝试 lower 这些操作，导致类型转换冲突。
