# SystolicArrayUnit 独立功能测试指南

本文档说明如何只对 `SystolicArrayUnit` 编写功能测试。测试不需要编译器、MLIR、workload 或完整 SoC：测试程序直接准备 bank 数据、发送 `SYSTOLIC` 指令、模拟 bank 读写，并把写回的 C 矩阵与 CPU golden 比较。

本文档描述的是当前 RTL 的接口和存储合同：

- 单个 bank 地址宽度为 128 bit。
- A/B 元素为有符号 `int8`；一个 bank 地址放 16 个元素。
- C 元素为有符号 `int32`；一个 128-bit beat 放 4 个元素。
- A/B 的尾 tile 在 bank 中补零。
- C 的尾 beat 只写有效元素，测试应检查写 mask，不应要求无效 lane 被覆盖为零。

## 1. 测试对象和前置配置

测试对象是：

```scala
test(new SystolicArrayUnit(config)) { dut =>
  // 驱动命令、bank read response、bank write response。
}
```

`config` 必须包含一个名为 `SystolicArrayBall` 的 Ball 映射，且当前 RTL 要求：

| 配置项 | 要求 |
| --- | --- |
| `ballName` | `SystolicArrayBall` |
| `inBW` | 至少 2；端口 0 读 A，端口 1 读 B |
| `outBW` | 至少 4；当前 RTL 的接口参数要求，但不表示 4 个并发写请求 |
| `bankWidth` | 128 bit |
| `bankEntries` | 每个 group 的 bank 地址数；当前常用值为 128 |
| ISA | `SYSTOLIC` 的 `funct7` 为 65 |

测试文件通常放在：

```text
arch/test/src/framework/balldomain/prototype/systolicarray/
```

## 2. 指令格式

每个矩阵乘法任务发送一条 `funct7 = 65` 指令。逻辑语义为：

```text
C[M,N] = A[M,K] × B[K,N]
```

`rs1` 规定 bank 和起始地址：

| 字段 | 位段 | 含义 |
| --- | --- | --- |
| `op1_bank` | `rs1[9:0]` | A 所在 bank |
| `op2_bank` | `rs1[19:10]` | B 所在 bank |
| `wr_bank` | `rs1[29:20]` | C 写回 bank |
| `op1_base` | `rs1[36:30]` | A 的起始 128-bit 行地址 |
| `op2_base` | `rs1[43:37]` | B 的起始 128-bit 行地址 |
| `wr_base` | `rs1[50:44]` | C 的起始 128-bit beat 地址 |

`rs2` 规定矩阵尺寸和数据流：

| 字段 | 位段 | 含义 |
| --- | --- | --- |
| `M` | `rs2[11:0]` | A/C 的行数 |
| `N` | `rs2[23:12]` | B/C 的列数 |
| `K` | `rs2[35:24]` | A 的列数、B 的行数 |
| `mode` | `rs2[36]` | `0` 为 OS，`1` 为 WS |

测试中应至少设置：

```scala
dut.io.cmdReq.bits.cmd.funct7.poke(65.U)
dut.io.cmdReq.bits.cmd.rs1.poke(rs1.U)
dut.io.cmdReq.bits.cmd.rs2.poke(rs2.U)
dut.io.cmdReq.bits.rob_id.poke(robId.U)
dut.io.cmdReq.valid.poke(true.B)
```

保持 `cmdReq.valid` 直到 `cmdReq.ready` 为高并完成握手。命令完成后，检查 `cmdResp.valid && cmdResp.ready`，并验证返回的 `rob_id`。

## 3. 通用 bank 表示

推荐在测试中用三元组保存 bank 内容：

```scala
case class BankKey(group: Int, bank: Int, addr: Int)
val memory = mutable.Map.empty[BankKey, BigInt]
```

一个逻辑 bank 地址先加上对应的 `base`。每累计 `bankEntries` 个 128-bit 地址，`group` 加一，`addr` 回到 0。

例如 `bankEntries = 128`、起始地址为 120 时，连续 16 行会使用：

```text
group 0, addr 120..127
group 1, addr 0..7
```

## 4. A 在 `op1_bank` 中的布局

A 是逻辑 `M×K` 矩阵。A 按 M tile、再按 K tile 存放；每个 tile 的尺寸为 `16×16`。

例如当前配置 `bankEntries=128`、`M=129, K=19` 时，A 有 9 个 M tile 和 2 个 K tile，共 18 个 tile。若 `op1_base=0`，实际存放顺序为：

```text
group 0, addr   0..15 : A[行   0..15,  列  0..15]；完整 tile
group 0, addr  16..31 : A[行   0..15,  列 16..18]；每行 lane 0..2 有效
group 0, addr  32..47 : A[行  16..31,  列  0..15]；完整 tile
group 0, addr  48..63 : A[行  16..31,  列 16..18]；每行 lane 0..2 有效
group 0, addr  64..79 : A[行  32..47,  列  0..15]；完整 tile
group 0, addr  80..95 : A[行  32..47,  列 16..18]；每行 lane 0..2 有效
group 0, addr  96..111: A[行  48..63,  列  0..15]；完整 tile
group 0, addr 112..127: A[行  48..63,  列 16..18]；每行 lane 0..2 有效
group 1, addr   0..15 : A[行  64..79,  列  0..15]；跨 group 后 addr 回到 0
group 1, addr  16..31 : A[行  64..79,  列 16..18]；每行 lane 0..2 有效
group 1, addr  32..47 : A[行  80..95,  列  0..15]；完整 tile
group 1, addr  48..63 : A[行  80..95,  列 16..18]；每行 lane 0..2 有效
group 1, addr  64..79 : A[行  96..111, 列  0..15]；完整 tile
group 1, addr  80..95 : A[行  96..111, 列 16..18]；每行 lane 0..2 有效
group 1, addr  96..111: A[行 112..127, 列  0..15]；完整 tile
group 1, addr 112..127: A[行 112..127, 列 16..18]；每行 lane 0..2 有效
group 2, addr   0..15 : A[行 128,      列  0..15]；仅 tile row 0 有效
group 2, addr  16..31 : A[行 128,      列 16..18]；仅 row 0 的 lane 0..2 有效
```

每个 A tile 占 16 个连续 bank 地址。一个地址对应 tile 的一行；从低到高的 16 个 int8 lane 对应该行的 16 个 K 元素。

写入 A 时可按下面的伪代码组织：

```text
for each M tile mt:
  for each K tile kt:
    for row = 0..15:
      logicalRow = op1_base + (mt * KTileCount + kt) * 16 + row
      bank[logicalRow] = A[mt*16 + row, kt*16 + lane], lane = 0..15
```

超出真实矩阵范围的行或 lane 写 0。上例中，K tile 1 的 lane 3..15 写 0；M tile 8 的 row 1..15 写 0。

## 5. B 在 `op2_bank` 中的布局

B 是逻辑 `K×N` 矩阵。B 按 N tile、再按 K tile 存放；每个 tile 同样为 `16×16`。

沿用 `bankEntries=128`、`K=19, N=19` 的例子，B 有 2 个 N tile 和 2 个 K tile，共 4 个 tile。B 本身只占 64 个地址；为了明确展示跨 group，下面使用 `op2_base=96`：

```text
group 0, addr  96..111: B[行  0..15, 列  0..15]；完整 tile
group 0, addr 112..127: B[行 16..18, 列  0..15]；仅 tile row 0..2 有效
group 1, addr   0..15 : B[行  0..15, 列 16..18]；跨 group，每行 lane 0..2 有效
group 1, addr  16..31 : B[行 16..18, 列 16..18]；仅 row 0..2 的 lane 0..2 有效
```

一个 B bank 地址对应 K 方向的一行；从低到高的 16 个 int8 lane 对应 N 方向的列。

```text
for each N tile nt:
  for each K tile kt:
    for row = 0..15:
      logicalRow = op2_base + (nt * KTileCount + kt) * 16 + row
      bank[logicalRow] = B[kt*16 + row, nt*16 + lane], lane = 0..15
```

超出 N 或 K 的位置写 0。上例中，K tile 1 的 row 3..15 写 0；N tile 1 的 lane 3..15 写 0。

## 6. C 在 `wr_bank` 中的布局

当前 RTL 的 C 存储合同是：先 M tile、再 N tile、然后 tile 内逐行存放。这里的存储顺序与 WS 的计算到达顺序无关；Ctrl 会把 WS 的结果重映射到下面的地址顺序。

沿用 `bankEntries=128`、`M=129, N=19`、`wr_base=0` 的例子，C 有 9 个 M tile 和 2 个 N tile。实际写入顺序为：

```text
group 0, addr   0..63 : C[行   0..15,  列  0..15]；每行 4 beats
group 0, addr  64..79 : C[行   0..15,  列 16..18]；每行 1 beat，lane 0..2 有效
group 0, addr  80..127: C[行  16..27,  列  0..15]；前 12 行，每行 4 beats
group 1, addr   0..15 : C[行  28..31,  列  0..15]；同一 C tile 跨 group
group 1, addr  16..31 : C[行  16..31,  列 16..18]；每行 1 beat，lane 0..2 有效
group 1, addr  32..95 : C[行  32..47,  列  0..15]；每行 4 beats
group 1, addr  96..111: C[行  32..47,  列 16..18]；每行 1 beat，lane 0..2 有效
group 1, addr 112..127: C[行  48..51,  列  0..15]；前 4 行，每行 4 beats
group 2, addr   0..47 : C[行  52..63,  列  0..15]；同一 C tile 跨 group
group 2, addr  48..63 : C[行  48..63,  列 16..18]；每行 1 beat，lane 0..2 有效
group 2, addr  64..127: C[行  64..79,  列  0..15]；每行 4 beats
group 3, addr   0..15 : C[行  64..79,  列 16..18]；每行 1 beat，lane 0..2 有效
group 3, addr  16..79 : C[行  80..95,  列  0..15]；每行 4 beats
group 3, addr  80..95 : C[行  80..95,  列 16..18]；每行 1 beat，lane 0..2 有效
group 3, addr  96..127: C[行  96..103, 列  0..15]；前 8 行，每行 4 beats
group 4, addr   0..31 : C[行 104..111, 列  0..15]；同一 C tile 跨 group
group 4, addr  32..47 : C[行  96..111, 列 16..18]；每行 1 beat，lane 0..2 有效
group 4, addr  48..111: C[行 112..127, 列  0..15]；每行 4 beats
group 4, addr 112..127: C[行 112..127, 列 16..18]；每行 1 beat，lane 0..2 有效
group 5, addr   0..3  : C[行 128,      列  0..15]；最后一行，共 4 beats
group 5, addr   4     : C[行 128,      列 16..18]；最后一行，lane 0..2 有效
```

在一个 C tile 内，每行每 4 个 int32 使用一个 128-bit beat：

```text
beat 0: 列 0..3
beat 1: 列 4..7
beat 2: 列 8..11
beat 3: 列 12..15
```

最后一个 beat 只启用有效元素对应的 byte mask。例如 N tile 只有 1 列有效时，该行只产生一个 beat，其中 lane 0 是有效结果；lane 1..3 不应被测试当作有效 C 元素。

该 C 一共写入 `129×(ceil(16/4)+ceil(3/4))=645` 个 128-bit beats，即 5 个完整 group 加 5 个地址。当前配置 `bank.num=32`、`bank.entries=128`，从 group 0 开始时单个 `wr_bank` 可表示 4096 个 beats，因此不会超过容量。

这是一种 tile-major 的紧密布局；它不是“把全矩阵第 0 行的所有 N 列连续放完，再放第 1 行”的全局 row-major 布局。新测试必须按照本节的 tile 顺序检查当前 RTL。

## 7. Bank response 和 write response 模型

测试循环每个周期应做四件事：

1. 检查 bank read request 是否握手；用 `(group, bank, addr)` 从 `memory` 取出 128-bit 数据，排入对应 read response 队列。
2. 在 response 队列非空时拉高对应 `bankRead(port).io.resp.valid`，保持数据直到 `ready`。
3. 检查 bank write request 是否握手；按请求的 byte mask 合并到 `memory`。不要直接整字覆盖，因为 C 尾 beat 可能只有部分 lane 有效。
4. 对已接受的 write request 返回成功 response，并持续推进 `clock.step()`。

当前 `SystolicArrayUnit` 只使用 `bankWrite(0)`：一行 C 被拆成最多 4 个 128-bit beat，但每个 beat 发出后都必须先等待 port 0 的 response，之后才会发送下一个 beat。`bankWrite(1..3)` 当前保持无效。因此测试模型一次只需处理一个在途 bank write request；`outBW=4` 是 Unit 的接口配置要求，不是当前实现的并发写吞吐。

读写端口都应加入确定性的 backpressure，例如每隔若干周期拉低一次 `ready`。这样可以验证 Load/EX/Store 的 Decoupled 握手，而不是只验证无阻塞路径。

## 8. CPU golden 和检查方法

CPU golden 应使用标准矩阵乘法：每个 C 元素由 A 的对应行和 B 的对应列逐项相乘累加得到。

建议使用固定 seed 的随机有符号 int8 输入，例如从 `[-63, 63]` 中取非零值。这样能覆盖正负数、避免单位矩阵或全 1 矩阵掩盖错位问题，同时在常见 K 值下不会使 int32 累加溢出。

检查时：

1. 等待命令完成，确认每条 `rob_id` 都返回。
2. 根据第 6 节的 C 地址顺序读取 `wr_bank`。
3. 只比较有效 C lane；同时检查每个 write request 的 byte mask 和有效 beat 数。
4. mismatch 至少打印 M/N/K、mode、随机 seed、C 坐标、bank/group/addr/lane、expected 和 actual。
5. 对 WS 任务额外记录 B 读取行数和地址，确认每个 B tile 最多复用 8 个 M tile 后才被重新加载。

## 9. 建议测试案例

| 目标 | M×N×K | mode | 应覆盖的行为 |
| --- | --- | --- | --- |
| 小尺寸与补零 | 5×7×3 | OS | M/N/K 都不满 16 |
| 小型权重驻留 | 9×6×11 | WS | 单个 B tile 的 PE 权重加载 |
| 多 tile OS | 17×33×19 | OS | 多 M/N/K tile、尾行/尾列/尾 K tile |
| B 复用边界 | 256×17×24 | WS | 两个 8-M-tile batch、N 尾列、K 尾 tile、跨 group |
| 多任务 | 多条连续命令 | OS+WS | 命令队列、`rob_id` 顺序、bank backpressure |

多任务同时入队时，应为每个仍可能并发执行的任务分配不重叠的 A、B、C bank，或在前一任务完成并完成结果检查后再复用 bank。
