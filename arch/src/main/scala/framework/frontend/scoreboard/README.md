# Buckyball 指令级并行方案：Bank Scoreboard

## 一、背景与问题

### 1.1 现状

Buckyball NPU 的 Global ROB 支持乱序发射和乱序完成，但软件中每两条指令之间都插入了 `bb_fence()`。fence 要求 ROB **完全排空**后才接受新指令，等价于 ILP=1——指令级并行**形同虚设**。

典型模式：
```
MVIN(bank0) → fence → RELU(bank0→bank1) → fence → MVOUT(bank1)
```

fence 解决的本质问题是 **bank 级 RAW/WAR/WAW 数据依赖**，但它使用了最粗粒度的全局屏障——即使操作不同 bank 的指令之间完全无依赖，fence 也会强制它们串行执行。

### 1.2 目标

- 在 `framework/frontend/scoreboard/` 中实现一个**指令无感的 Bank Scoreboard**
- Scoreboard 只关心每条指令的**读 bank 集合**和**写 bank 集合**，不关心具体指令类型
- 保留 fence 指令语义，但 fence 不再要求排空整个 ROB；fence 作为 ROB 中的**屏障点**，ROB 扫描发射时只扫描 fence 之前的指令
- 统一所有指令的 bank 信息编码到 rs1 的特定位
- 实现乱序跳跃发射：不同 bank 的指令可以越过有冲突的指令先发射

---

## 二、统一 rs1 Bank 编码

### 2.1 设计原则

所有指令的 bank 操作数统一编码到 rs1：

```
rs1[7:0]   = bank_0  （第一操作数 bank / MVIN 写 bank / MVOUT 读 bank）
rs1[15:8]  = bank_1  （第二操作数 bank，仅双操作数指令使用）
rs1[23:16] = bank_2  （写结果 bank）
```

### 2.2 各指令新编码

#### Mem Domain 指令

| 指令 | rs1 | rs2 | 变更说明 |
|------|-----|-----|---------|
| **MVIN** (24) | `[7:0]=wr_bank`, `[63:8]=mem_addr` | `[9:0]=depth, [28:10]=stride, [63:29]=special` | bank 从 rs2 移到 rs1[7:0]；mem_addr 移到 rs1 高位 |
| **MVOUT** (25) | `[7:0]=rd_bank`, `[63:8]=mem_addr` | `[9:0]=depth, [28:10]=stride, [63:29]=special` | 同上 |
| **MSET** (23) | `[7:0]=bank_id` | `[4:0]=row, [9:5]=col, [10]=alloc, ...` | bank 从 rs2 移到 rs1[7:0] |

#### Ball Domain 指令

| 指令 | 当前编码 | 新编码变更 |
|------|---------|-----------|
| **MATMUL_WARP16** (32) | op1=rs1[7:0], op2=rs1[15:8], wr=**rs2[7:0]** | wr 移到 **rs1[23:16]**；rs2 重排（iter 从 bit0 开始） |
| **RELU** (38) | op1=rs1[7:0], wr=**rs2[7:0]** | wr 移到 **rs1[23:16]** |
| **TRANSPOSE** (34) | op1=rs1[7:0], wr=**rs1[15:8]** | wr 移到 **rs1[23:16]** |
| **IM2COL** (33) | op1=rs1[7:0], wr=**rs1[15:8]** | wr 移到 **rs1[23:16]** |
| **CONCAT** (39) | op1=rs1[7:0], wr=**rs2[7:0]** | wr 移到 **rs1[23:16]** |
| **TRANSFER** (45) | op1=rs1[7:0], wr=**rs2[7:0]** | wr 移到 **rs1[23:16]** |

> BBFP_MUL(26), MATMUL_WS(27), ABFT_SYSTOLIC(42), CONV(43), CIM(44) 等双操作数指令同 MATMUL_WARP16 模式处理。

### 2.3 统一提取好处

GlobalDecoder 只需：
```
rd_bank_0 = rs1[7:0]
rd_bank_1 = rs1[15:8]
wr_bank   = rs1[23:16]   （Ball 指令）
          或 rs1[7:0]     （MVIN/MSET，写 bank 就是 bank_0）
```

加上一个简单的 valid 查找表（按 func7 索引），即可产出完全指令无感的 `BankAccessInfo`。

---

## 三、Bank Scoreboard 设计

### 3.1 位置

`framework/frontend/scoreboard/BankScoreboard.scala`

### 3.2 核心接口

```scala
class BankAccessInfo(bankIdLen: Int) extends Bundle {
  val rd_bank_0_valid = Bool()
  val rd_bank_0_id    = UInt(bankIdLen.W)
  val rd_bank_1_valid = Bool()
  val rd_bank_1_id    = UInt(bankIdLen.W)
  val wr_bank_valid   = Bool()
  val wr_bank_id      = UInt(bankIdLen.W)
}
```

Scoreboard **只接收 `BankAccessInfo`**，不关心具体指令类型、domain、func7。

### 3.3 内部数据结构

```scala
// 读计数器：多 bit，允许多条指令同时读同一 bank（RR 无冲突）
val bankRdCount = RegInit(VecInit(Seq.fill(bankNum)(0.U(cntWidth.W))))
// cntWidth = log2Ceil(rob_entries + 1)

// 写标志：1-bit，WAW 规则保证同一 bank 最多 1 个写者 in-flight
val bankWrBusy = RegInit(VecInit(Seq.fill(bankNum)(false.B)))
```

**为什么 wrCount 用 1-bit 就够？**
因为冲突检测规则中，写 bank X 要求 `bankWrBusy[X]==false`（WAW 阻塞），所以同一 bank 不可能有两条写指令同时 in-flight。

**为什么 rdCount 需要多 bit？**
多条指令同时读同一 bank 是允许的（RR 无冲突），因此需要计数器追踪有多少个并发读者，以便写操作判断 WAR 冲突。

### 3.4 冲突检测规则

```
新指令读 bank X → 要求 bankWrBusy[X] == false    （否则 RAW 冲突：正在被写）
新指令写 bank X → 要求 bankRdCount[X] == 0        （否则 WAR 冲突：正在被读）
                   且 bankWrBusy[X] == false       （否则 WAW 冲突：正在被写）
```

即：
```scala
def hasHazard(info: BankAccessInfo): Bool = {
  val rd0 = info.rd_bank_0_valid && bankWrBusy(info.rd_bank_0_id)
  val rd1 = info.rd_bank_1_valid && bankWrBusy(info.rd_bank_1_id)
  val wr  = info.wr_bank_valid && (
    bankRdCount(info.wr_bank_id) =/= 0.U ||
    bankWrBusy(info.wr_bank_id)
  )
  rd0 || rd1 || wr
}
```

### 3.5 计数器/标志更新

**发射时**（issue.fire）：
```scala
when(info.rd_bank_0_valid) { bankRdCount(info.rd_bank_0_id) += 1.U }
when(info.rd_bank_1_valid) { bankRdCount(info.rd_bank_1_id) += 1.U }
when(info.wr_bank_valid)   { bankWrBusy(info.wr_bank_id)   := true.B }
```

**完成时**（complete.fire）：
```scala
when(info.rd_bank_0_valid) { bankRdCount(info.rd_bank_0_id) -= 1.U }
when(info.rd_bank_1_valid) { bankRdCount(info.rd_bank_1_id) -= 1.U }
when(info.wr_bank_valid)   { bankWrBusy(info.wr_bank_id)   := false.B }
```

完成时需要从 ROB entry 中取回 `BankAccessInfo`，因此 **GlobalRobEntry 必须保存 `BankAccessInfo`**。

### 3.6 MSET 精确追踪

MSET 标记为 `wr_bank_valid=true, wr_bank_id=rs1[7:0]`。它只与同一 bank 的其他指令串行，不阻塞其他 bank。

### 3.7 GP Domain (RVV) 指令

GP 指令不访问 scratchpad bank，其 `BankAccessInfo` 全部 valid=false，不会被 scoreboard 阻塞，也不阻塞其他指令。

---

## 四、Fence 指令新语义

### 4.1 设计

Fence **进入 ROB**（当前实现中 fence 不进 ROB），但不发射到任何 domain。

ROB 中 fence 的作用：**发射屏障**。ROB 从 head 扫描待发射指令时，遇到 fence 就停止扫描——fence 之后的指令不会被考虑发射。

### 4.2 Fence 生命周期

1. Fence 进入 ROB，分配 entry，标记为特殊的 fence 类型
2. ROB 扫描发射时，以 fence 为边界，只扫描 fence 之前的指令
3. 当 fence 成为 head 且 fence 之前的所有指令都已完成时，fence 自动标记为 complete 并被 commit
4. Head 前进到 fence 之后，后续指令可以开始发射

### 4.3 与 Scoreboard 的关系

fence 之前的指令仍然受 scoreboard 管理——不同 bank 的指令可以乱序发射。fence 只是限制了扫描窗口的边界，确保 fence 之后的指令不会越过 fence 提前发射。

### 4.4 实际效果

- **无 fence 时**：ROB 中所有指令根据 bank 依赖自由发射，最大并行度
- **有 fence 时**：fence 前后的指令严格有序，fence 内部的指令仍可乱序
- 软件可以选择性使用 fence 来强制特定的执行顺序（调试、特殊语义等）

---

## 五、ROB 发射逻辑修改

### 5.1 当前逻辑

```scala
// 从 head 扫描，找第一个 valid && !issued && !complete 的指令
scanValid(i) := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
```

### 5.2 新逻辑

```scala
// 增加两个条件：(1) 无 bank 冲突 (2) 不在 fence 之后
// fenceBarrier(i) = true 表示该位置在某个未完成的 fence 之后
scanValid(i) := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
                && !hasHazard(robEntries(ptr).bankAccess)
                && !isBehindFence(i)
```

**isBehindFence 计算**：
从 head 开始扫描，一旦遇到一个 valid 且未 complete 的 fence entry，其后所有位置的 `isBehindFence` 标记为 true。

```scala
val fenceBarrier = Wire(Vec(rob_entries, Bool()))
var seenFence = false.B
for (i <- 0 until rob_entries) {
  val ptr = (headPtr + i.U) % rob_entries.U
  val isFence = robValid(ptr) && isFenceEntry(ptr) && !robComplete(ptr)
  seenFence = seenFence || isFence
  fenceBarrier(i) := seenFence  // fence 本身和之后的所有项都被屏蔽
}
```

### 5.3 Fence Entry 的自动完成

当 fence 成为 head（即 fence 之前的所有指令都已 commit）时，fence 自动标记为 issued + complete：

```scala
when (robValid(headPtr) && isFenceEntry(headPtr)) {
  robIssued(headPtr)   := true.B
  robComplete(headPtr) := true.B
}
```

---

## 六、需要修改的文件

### 硬件（Scala）

| 文件 | 改动内容 |
|------|---------|
| **新建** `framework/frontend/scoreboard/BankScoreboard.scala` | BankAccessInfo 定义、BankScoreboard 模块 |
| `framework/frontend/decoder/GobalDecoder.scala` | PostGDCmd 增加 BankAccessInfo 字段；增加 bank 提取逻辑和 valid 查表 |
| `framework/frontend/globalrs/GlobalROB.scala` | GlobalRobEntry 增加 BankAccessInfo 和 isFence 标志；集成 BankScoreboard；修改发射逻辑（hazard + fence barrier）；fence 自动完成逻辑 |
| `framework/frontend/globalrs/GlobalReservationStation.scala` | 删除旧的 fence 处理逻辑（fenceActive 等）；fence 改为进入 ROB 而非阻塞等待排空 |
| `framework/frontend/Frontend.scala` | 可能需要传递 bankNum 参数（已在 GlobalConfig 中） |
| `framework/memdomain/frontend/cmd_channel/decoder/DomainDecoder.scala` | bank_id 从 rs1[bankIdLen-1:0] 提取；mem_addr 从 rs1 高位提取；iter/stride 从 rs2 新位置提取 |
| `examples/toy/balldomain/DomainDecoder.scala` | wr_bank 统一从 rs1[23:16] 提取；更新 decode table |

### 软件（C 指令宏）

| 文件 | 改动内容 |
|------|---------|
| `bb-tests/.../isa/24_mvin.c` | bank_id 移到 rs1[7:0]，mem_addr 移到 rs1 高位 |
| `bb-tests/.../isa/25_mvout.c` | 同 MVIN |
| `bb-tests/.../isa/23_mset.c`（或含 mset 的文件） | bank_id 移到 rs1[7:0] |
| `bb-tests/.../isa/32_mul_warp16.c` | wr_bank 从 rs2[7:0] 移到 rs1[23:16] |
| `bb-tests/.../isa/38_relu.c` | wr_bank 从 rs2[7:0] 移到 rs1[23:16] |
| `bb-tests/.../isa/34_transpose.c` | wr_bank 从 rs1[15:8] 移到 rs1[23:16] |
| `bb-tests/.../isa/33_im2col.c` | wr_bank 从 rs1[15:8] 移到 rs1[23:16] |
| `bb-tests/.../isa/45_transfer.c` | wr_bank 从 rs2[7:0] 移到 rs1[23:16] |
| 其他 Ball 指令宏文件 | 类似处理 |
| 所有测试 .c 文件 | **删除 `bb_fence()` 调用** |

---

## 七、实施步骤

1. 定义 `BankAccessInfo` Bundle，新建 `BankScoreboard` 模块
2. 修改软件侧指令宏（MVIN/MVOUT/MSET/Ball 指令），统一 bank 编码到 rs1
3. 修改 `MemDomainDecoder`，适配新的 rs1/rs2 字段位置
4. 修改 `BallDomainDecoder`，wr_bank 统一从 rs1[23:16] 提取
5. 修改 `GlobalDecoder`，增加 `BankAccessInfo` 提取逻辑
6. 修改 `GlobalROB`，集成 BankScoreboard，实现 hazard 检测 + fence barrier 扫描逻辑
7. 修改 `GlobalReservationStation`，删除旧 fence 逻辑，fence 改为进入 ROB
8. 删除测试中的 `bb_fence()` 调用
9. 编译验证 + 运行测试

---

## 八、验证方案

1. Chisel 编译通过，无类型/语法错误
2. 运行全部 CTest（relu_test, transpose_test, mvin_mvout_test, transfer_test, tiled_matmul 等）
3. 所有测试在无 fence 的情况下结果正确
4. 可选：波形验证，确认不同 bank 的指令确实并行发射
5. 可选：加回 fence 的测试，确认 fence barrier 语义正确
