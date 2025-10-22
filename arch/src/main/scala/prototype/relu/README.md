# ReLU 激活函数加速器

## 概述

该目录实现了 BuckyBall 的 ReLU（Rectified Linear Unit）激活加速器，位于 `arch/src/main/scala/prototype/relu` 下。模块以矢量化方式对 Scratchpad 中的数据按 tile（`veclane × veclane`）进行逐元素 ReLU 处理，并将结果写回。

实现的核心组件：
- **Relu.scala**: ReLU 加速器主体实现

## 代码结构

```
relu/
└── Relu.scala  - ReLU 加速器实现
```

### 模块职责

**Relu.scala**（加速器实现层）
- 从 Scratchpad 读取一个 `veclane × veclane` tile 数据
- 对每个元素执行带符号比较的 ReLU 运算（负数置 0）
- 以掩码全写方式写回同尺寸 tile
- 提供 Ball 域命令接口并回传完成响应/状态

## 模块说明

### Relu.scala

**主要功能**:

逐 tile（`veclane × veclane`）读取输入 → 执行逐元素 ReLU → 逐行写回输出；支持 `iter` 驱动的批量处理与流水工作流。

**状态机定义**:

```scala
val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
val state = RegInit(idle)
```

**关键寄存器**:

```scala
// 数据缓存：veclane × veclane，每个元素宽度为 inputType.getWidth
val regArray = RegInit(
  VecInit(Seq.fill(b.veclane)(
    VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W)))
  ))
)

// 计数器
val readCounter  = RegInit(0.U(log2Ceil(b.veclane + 1).W)) // 已发起的读请求“行”计数
val respCounter  = RegInit(0.U(log2Ceil(b.veclane + 1).W)) // 已接收的读响应“行”计数
val writeCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W)) // 已写回的“行”计数

// 指令字段寄存器
val robid_reg = RegInit(0.U(10.W)) // 记录命令的 ROB ID
val waddr_reg = RegInit(0.U(10.W)) // 写回的起始行地址
val wbank_reg = RegInit(0.U(log2Up(b.sp_banks).W)) // 写回目标的 Scratchpad bank 选择
val raddr_reg = RegInit(0.U(10.W)) // 读取的起始行地址
val rbank_reg = RegInit(0.U(log2Up(b.sp_banks).W)) // 读取来源的 Scratchpad bank 选择
val iter_reg  = RegInit(0.U(10.W)) // 命令中指定的处理行数/长度
val cycle_reg = RegInit(0.U(6.W))      // tile 轮数（由 iter 推导）
val iterCnt   = RegInit(0.U(32.W))     // 完成的批次数

// 写回数据与掩码
val spad_w       = b.veclane * b.inputType.getWidth // 一行打包的数据位宽
val writeDataReg = Reg(UInt(spad_w.W)) // 待写回的一行打包数据
val writeMaskReg = Reg(Vec(b.spad_mask_len, UInt(1.W))) // 写回掩码向量
```

**命令解析**:

```scala
when(io.cmdReq.fire) {
  // 进入读取阶段并初始化本轮计数
  state        := sRead
  readCounter  := 0.U      // 已发起的读请求行计数清零
  writeCounter := 0.U      // 已写回的行计数清零

  // 记录命令标识
  robid_reg := io.cmdReq.bits.rob_id            // ROB ID（用于完成响应匹配）

  // 输出（写回）目标地址：使用 wr_* 字段
  waddr_reg := io.cmdReq.bits.cmd.wr_bank_addr  // 写回起始行地址
  wbank_reg := io.cmdReq.bits.cmd.wr_bank       // 写回目标 bank

  // 输入（读取）来源地址：使用 op1_* 字段
  raddr_reg := io.cmdReq.bits.cmd.op1_bank_addr // 读取起始行地址
  rbank_reg := io.cmdReq.bits.cmd.op1_bank      // 读取来源 bank

  // 迭代与轮次
  iter_reg  := io.cmdReq.bits.cmd.iter          // 需要处理的总行数（迭代数）
  // 计算本批需要的 tile 轮数：每轮处理 veclane 行
  // cycle_reg = ceil(iter / veclane) - 1，写/读完成一轮后递减
  cycle_reg := (io.cmdReq.bits.cmd.iter +& (b.veclane.U - 1.U)) / b.veclane.U - 1.U
}
```

**数据转换逻辑（ReLU）**:

- 读取返回宽度为 `spad_w = veclane × inputWidth` 的一行打包数据；
- 将其拆分为 `veclane` 个元素，并进行带符号判断：`x < 0 ? 0 : x`；

```scala
// 拆分 + ReLU（按列）
val dataWord = io.sramRead(rbank_reg).resp.bits.data
for (col <- 0 until b.veclane) {
  val hi = (col + 1) * b.inputType.getWidth - 1
  val lo = col * b.inputType.getWidth
  val raw    = dataWord(hi, lo)
  val signed = raw.asSInt
  val relu   = Mux(signed < 0.S, 0.S(b.inputType.getWidth.W), signed)
  regArray(respCounter)(col) := relu.asUInt
}
```

写回时，将一整行的 `veclane` 个元素重新打包：

```scala
// 将 regArray(rowIdx) 打包成一行写回
writeDataReg := Cat((0 until b.veclane).reverse.map(j => regArray(rowIdx)(j)))
// 全掩码写
for (i <- 0 until b.spad_mask_len) { writeMaskReg(i) := 1.U }
```

**SRAM 接口**:

```scala
val io = IO(new Bundle {
  val cmdReq    = Flipped(Decoupled(new BallRsIssue))
  val cmdResp   = Decoupled(new BallRsComplete)
  val sramRead  = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len)))
  val status    = new Status
})
```

**处理流程**:

1. **idle**：等待命令；解析输入/输出地址 bank/addr、迭代次数 `iter`，并据此计算 `cycle_reg`。
2. **sRead**：按行对输入 bank/addr 连续发起读请求；接收数据后逐元素执行 ReLU，填充到 `regArray`；当累计 `veclane` 行后进入写阶段。
3. **sWrite**：逐行打包并写回到 `wbank` 的连续地址，掩码全写；写满 `veclane` 行后进入完成阶段。
4. **complete**：当所有轮次完成（`cycle_reg == 0`）时，发出 `cmdResp` 完成响应；随后回到 `idle`。

**输入输出**:

- 输入：Ball 域命令（`wr_bank/wr_bank_addr`、`op1_bank/op1_bank_addr`、`iter` 等）
- 输出：写回后的 ReLU 结果 tile，`cmdResp` 完成通知
- 边界与约束：
  - 每轮处理 `veclane` 行，迭代轮数由 `iter` 推导；
  - 数据元素按 `b.inputType` 进行带符号比较；
  - 写回使用全掩码（可根据需求扩展部分写）。

## 使用方法

- 将源数据布置在 Scratchpad 指定的 `op1_bank/op1_bank_addr` 起始位置，保证每行宽度为 `veclane × inputWidth`；
- 配置输出 `wr_bank/wr_bank_addr`，以及待处理的元素行数 `iter`；
- 发送 Ball 命令后，等待 `cmdResp` 完成；
- 可轮询 `status`：`ready/valid/idle/init/running/complete/iter` 获取运行时信息。

### 注意事项

1. **有符号比较**：ReLU 使用 `asSInt` 进行负值判断，负数置 0；请确保 `b.inputType` 与上游数据约定一致（定点/补码）。
2. **带宽与对齐**：每次读写为一行打包（`spad_w` 位），地址需按行对齐并连续递增。
3. **掩码策略**：当前实现为全掩码写；若需稀疏/部分写，可扩展 `writeMaskReg` 的生成逻辑。
4. **迭代与分块**：`iter` 非整倍数 `veclane` 时，`cycle_reg` 会按向上取整处理剩余行；必要时在边界补 0 或裁剪。
5. **子模块交互**：`sramRead/Write` 的 ready/valid 握手需与 Scratchpad 控制器保持一致；如存在返序/多拍延迟，需在 `respCounter` 逻辑中保护乱序情况。
6. **复位行为**：复位将清空 `regArray`、`writeDataReg`、`writeMaskReg`，便于仿真可重复。
