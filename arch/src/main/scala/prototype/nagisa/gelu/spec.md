# GELU加速单元设计规范

## 1. 概述 (Overview)

GELU (Gaussian Error Linear Unit) 加速单元是BuckyBall框架中的专用计算加速器，用于高效执行GELU激活函数运算。GELU是现代深度学习模型（如Transformer、BERT、GPT等）中广泛使用的非线性激活函数。

### 1.1 基本参数

- **数据格式**: 输入为INT8，输出为INT32
- **向量化处理**: 每次处理16个INT8元素（veclane=16）
- **流水线架构**: 多级流水线设计（ID, Load, Execute, Store）
- **计算方法**: 采用GELU近似算法（tanh公式）
- **存储接口**: 支持Scratchpad和Accumulator读写

### 1.2 数学定义

GELU激活函数的精确定义：

```
GELU(x) = x · Φ(x)
```

其中Φ(x)是标准正态分布的累积分布函数。

硬件实现采用tanh近似公式：

```
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

简化常数：
- √(2/π) ≈ 0.7978845608
- 0.044715

## 2. 系统架构 (Block Diagram)

### 2.1 顶层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        GELU Accelerator                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │              │    │              │    │              │       │
│  │   Control    │───▶│  Load Unit   │───▶│  Execute    │       │
│  │   Unit (ID)  │    │              │    │  Unit (EX)   │       │
│  │              │    │              │    │              │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │               │
│         │                   │                   │               │
│         │              ┌────▼────┐         ┌────▼────┐          │
│         │              │ SRAM    │         │ Store   │          │
│         │              │ Read    │         │ Unit    │          │
│         │              │ Arbiter │         │         │          │
│         │              └─────────┘         └────┬────┘          │
│         │                                       │               │
│  ┌──────▼────────────────────────────────┐      │               │
│  │         Command Interface             │      │               │
│  │   (Ball Bus / RoCC Interface)         │      │               │
│  └───────────────────────────────────────┘      │               │
│                                                 │               │
│  ┌──────────────────────────────────────┐       │               │
│  │         Status Monitor               │       │               │
│  │  (ready/valid/idle/init/running)     │       │               │
│  └──────────────────────────────────────┘       │               │
│                                                 │               │
└─────────────────────────────────────────────────┼───────────────┘
                                                  │
                                            ┌─────▼─────┐
                                            │  Memory   │
                                            │  System   │
                                            └───────────┘
```

### 2.2 流水线结构

```
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│  ID    │───▶│  Load  │───▶│   EX   │───▶│ Store  │
│ Stage  │    │ Stage  │    │ Stage  │    │ Stage  │
└────────┘    └────────┘    └────────┘    └────────┘
    │             │             │             │
    │             │             │             │
  Decode      Load Data    Compute GELU   Write Back
  Command     from SRAM    Approximation   to ACC/SRAM
```

### 2.3 计算单元架构

```
┌─────────────────────────────────────────────────────────┐
│              GELU Compute Pipeline (EX Stage)           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input x                                                │
│     │                                                   │
│     ├──────────┬──────────┬──────────┐                  │
│     │          │          │          │                  │
│     │       ┌──▼──┐     ┌──▼──┐    ┌──▼──┐              │
│     │       │ x²  │───▶│ x³  │───▶│ MUL │ (x³·0.044715)│
│     │       └─────┘     └─────┘    └──┬──┘              │
│     │                                 │                 │
│     │          ┌──────────────────────┘                 │
│     │          │                                        │
│     │       ┌──▼──┐                                     │
│     │       │ ADD │ (x + 0.044715·x³)                   │
│     │       └──┬──┘                                     │
│     │          │                                        │
│     │       ┌──▼──┐                                     │
│     │       │ MUL │ (0.7978845608 · ...)                │
│     │       └──┬──┘                                     │
│     │          │                                        │
│     │       ┌──▼──┐                                     │
│     │       │TANH │ (lookup table / polynomial approx)  │
│     │       └──┬──┘                                     │
│     │          │                                        │
│     │       ┌──▼──┐                                     │
│     │       │ADD+1│ (1 + tanh(...))                     │
│     │       └──┬──┘                                     │
│     │          │                                        │
│     └────┬─────┘                                        │
│          │                                              │
│       ┌──▼──┐                                           │
│       │ MUL │ x · (...)                                 │
│       └──┬──┘                                           │
│          │                                              │
│       ┌──▼──┐                                           │
│       │ MUL │ 0.5 · (...)                               │
│       └──┬──┘                                           │
│          │                                              │
│      Output GELU(x)                                     │
└─────────────────────────────────────────────────────────┘
```


## 3. 接口描述 (Interface Description)

GELU单元对外提供以下接口：
- **命令接口** (Command Interface): 接收GELU指令并返回完成响应
- **Scratchpad接口** (SRAM Interface): 访问INT8数据的存储器
- **Accumulator接口** (ACC Interface): 访问INT32数据的存储器
- **状态监控接口** (Status Interface): 输出当前运行状态信息
- **时钟和复位接口**: 提供时钟和复位信号

### 3.1 指令语义 (Instruction Semantics)

一条GELU指令的完整语义如下：

**指令含义**：对存储在Scratchpad或Accumulator中的向量执行GELU运算

**数据格式**：
- **Scratchpad模式** (`is_acc=0`)：INT8输入 → GELU → INT8输出
- **Accumulator模式** (`is_acc=1`)：INT32输入 → GELU → INT32输出
- **注意**：Scratchpad存储INT8，Accumulator存储INT32

**处理单位**：
- 每个向量 = 16个元素（veclane = 16）
- 每个SRAM地址存储1个INT8向量（16×8位 = 128位 = 16字节）
- 每个ACC地址存储1个INT32向量（16×32位 = 512位 = 64字节）

**指令参数说明**：

| 参数 | 含义 | 示例 |
|-----|------|------|
| `iter` | 要处理的向量个数 | iter=64 表示处理64个向量 |
| `op1_bank` | 输入数据所在的Bank号 | 0-3 |
| `op1_bank_addr` | 输入起始地址 | 0x100 |
| `wr_bank` | 输出数据写入的Bank号 | 0-3 |
| `wr_bank_addr` | 输出起始地址 | 0x200 |
| `is_acc` | 数据类型选择 | 0=SRAM(INT8)模式, 1=ACC(INT32)模式 |

**模式1：is_acc=0（SRAM模式，INT8）**

输入范围：
```
起始地址：SRAM[op1_bank][op1_bank_addr]
结束地址：SRAM[op1_bank][op1_bank_addr + iter - 1]
每地址：16个INT8元素（128位）
总元素数：iter × 16 个INT8元素
```

输出范围：
```
起始地址：SRAM[wr_bank][wr_bank_addr]
结束地址：SRAM[wr_bank][wr_bank_addr + iter - 1]
每地址：16个INT8元素（128位）
总元素数：iter × 16 个INT8元素
```

**模式2：is_acc=1（ACC模式，INT32）**

输入范围：
```
起始地址：ACC[op1_bank][op1_bank_addr]
结束地址：ACC[op1_bank][op1_bank_addr + iter - 1]
每地址：16个INT32元素（512位）
总元素数：iter × 16 个INT32元素
```

输出范围：
```
起始地址：ACC[wr_bank][wr_bank_addr]
结束地址：ACC[wr_bank][wr_bank_addr + iter - 1]
每地址：16个INT32元素（512位）
总元素数：iter × 16 个INT32元素
```

**示例1**：SRAM模式，处理INT8数据
```
iter = 64
op1_bank = 0, op1_bank_addr = 0x000
wr_bank = 1, wr_bank_addr = 0x000
is_acc = 0

输入：SRAM[0][0x000~0x03F] 的64个向量（1024个INT8元素）
输出：SRAM[1][0x000~0x03F] 的64个向量（1024个INT8元素）
```

**示例2**：ACC模式，处理INT32数据
```
iter = 64
op1_bank = 0, op1_bank_addr = 0x000
wr_bank = 1, wr_bank_addr = 0x000
is_acc = 1

输入：ACC[0][0x000~0x03F] 的64个向量（1024个INT32元素）
输出：ACC[1][0x000~0x03F] 的64个向量（1024个INT32元素）
```

**示例3**：单个向量处理
```
iter = 1
op1_bank = 0, op1_bank_addr = 0x100
wr_bank = 0, wr_bank_addr = 0x200
is_acc = 0

输入：SRAM[0][0x100] 的16个INT8元素
输出：SRAM[0][0x200] 的16个INT8元素
```

### 3.2 命令接口 (Command Interface)

GELU单元通过Ball Domain标准接口与系统交互：

| 信号名称 | 方向 | 位宽 | 描述 |
|---------|------|------|------|
| `cmdReq.valid` | Input | 1 | 命令请求有效信号 |
| `cmdReq.ready` | Output | 1 | 命令请求就绪信号 |
| `cmdReq.bits.rob_id` | Input | 10 | ROB (Reorder Buffer) 标识符 |
| `cmdReq.bits.iter` | Input | 10 | 向量迭代次数 (支持1-1024) |
| `cmdReq.bits.op1_bank` | Input | 2 | 操作数Bank选择 |
| `cmdReq.bits.op1_bank_addr` | Input | 12 | 操作数Bank内地址 |
| `cmdReq.bits.wr_bank` | Input | 2 | 写回Bank选择 |
| `cmdReq.bits.wr_bank_addr` | Input | 12 | 写回Bank内地址 |
| `cmdReq.bits.is_acc` | Input | 1 | 目标存储类型 (0=SRAM, 1=ACC) |

| 信号名称 | 方向 | 位宽 | 描述 |
|---------|------|------|------|
| `cmdResp.valid` | Output | 1 | 完成响应有效信号 |
| `cmdResp.ready` | Input | 1 | 完成响应就绪信号 |
| `cmdResp.bits.rob_id` | Output | 10 | 完成指令的ROB ID |
| `cmdResp.bits.commit` | Output | 1 | 提交标志 |

### 3.2 Scratchpad存储接口 (SRAM Interface)

支持多Bank并行访问的SRAM接口，存储INT8数据。Bank数量由配置参数`sp_banks`决定（典型值4）。

**读接口** (每Bank):

| 信号名称 | 方向 | 位宽 | 描述 |
|---------|------|------|------|
| `sramRead[i].req.valid` | Output | 1 | 读请求有效信号 |
| `sramRead[i].req.ready` | Input | 1 | 读请求就绪信号 |
| `sramRead[i].req.bits.addr` | Output | log2(entries) | 读地址 |
| `sramRead[i].resp.valid` | Input | 1 | 读响应有效信号 |
| `sramRead[i].resp.bits.data` | Input | 128 | 读数据（16个INT8 = 128位）|

**写接口** (每Bank):

| 信号名称 | 方向 | 位宽 | 描述 |
|---------|------|------|------|
| `sramWrite[i].valid` | Output | 1 | 写请求有效信号 |
| `sramWrite[i].ready` | Input | 1 | 写请求就绪信号 |
| `sramWrite[i].bits.addr` | Output | log2(entries) | 写地址 |
| `sramWrite[i].bits.data` | Output | 128 | 写数据（16个INT8 = 128位）|
| `sramWrite[i].bits.mask` | Output | 16 | 写掩码（按INT8元素）|

### 3.3 Accumulator存储接口 (ACC Interface)

Accumulator存储INT32数据。Bank数量由配置参数`acc_banks`决定（典型值2）。

**读接口** (每Bank):

| 信号名称 | 方向 | 位宽 | 描述 |
|---------|------|------|------|
| `accRead[i].req.valid` | Output | 1 | 读请求有效信号 |
| `accRead[i].req.ready` | Input | 1 | 读请求就绪信号 |
| `accRead[i].req.bits.addr` | Output | log2(acc_entries) | 读地址 |
| `accRead[i].resp.valid` | Input | 1 | 读响应有效信号 |
| `accRead[i].resp.bits.data` | Input | 512 | 读数据（16个INT32 = 512位）|

**写接口** (每Bank):

| 信号名称 | 方向 | 位宽 | 描述 |
|---------|------|------|------|
| `accWrite[i].valid` | Output | 1 | 写请求有效信号 |
| `accWrite[i].ready` | Input | 1 | 写请求就绪信号 |
| `accWrite[i].bits.addr` | Output | log2(acc_entries) | 写地址 |
| `accWrite[i].bits.data` | Output | 512 | 写数据（16个INT32 = 512位）|
| `accWrite[i].bits.mask` | Output | 16 | 写掩码（按INT32元素）|

### 3.4 状态监控接口 (Status Interface)

GELU单元提供状态监控接口，用于外部观察当前运行状态：

| 信号名称 | 方向 | 位宽 | 描述 |
|---------|------|------|------|
| `status.ready` | Output | 1 | 设备准备好接受新输入 |
| `status.valid` | Output | 1 | 设备有有效输出 |
| `status.idle` | Output | 1 | 空闲状态（无输入无输出）|
| `status.init` | Output | 1 | 初始化状态（有输入但无输出）|
| `status.running` | Output | 1 | 运行状态（已开始产生输出）|
| `status.complete` | Output | 1 | 完成信号（完全完成当前批次）|
| `status.iter` | Output | 32 | 已完成的批次迭代计数 |

**状态转换关系**：

```
idle (ready=1, valid=0, idle=1)
  ↓ [cmdReq.fire]
init (ready=0, valid=0, init=1)
  ↓ [开始产生输出]
running (ready=0, valid=1, running=1)
  ↓ [所有数据处理完成]
complete (complete=1)
  ↓ [cmdResp.fire]
idle (iter += 1)
```

**典型实现**：

```scala
// Status tracking registers
val iterCnt = RegInit(0.U(32.W))
val hasInput = RegInit(false.B)
val hasOutput = RegInit(false.B)

when(io.cmdReq.fire) {
  hasInput := true.B
}
when(io.cmdResp.fire) {
  hasOutput := false.B
  hasInput := false.B
  iterCnt := iterCnt + 1.U
}
when(io.cmdResp.valid && !hasOutput) {
  hasOutput := true.B
}

// Status signal assignments
io.status.ready := io.cmdReq.ready
io.status.valid := io.cmdResp.valid
io.status.idle := !hasInput && !hasOutput
io.status.init := hasInput && !hasOutput
io.status.running := hasOutput
io.status.complete := io.cmdResp.fire
io.status.iter := iterCnt
```

### 3.5 时钟和复位接口

| 信号名称 | 方向 | 描述 |
|---------|------|------|
| `clock` | Input | 全局时钟信号 |
| `reset` | Input | 全局同步复位信号 (高有效) |


## 4. 寄存器映射 (Register Map)

### 4.1 内部控制寄存器

GELU单元不直接暴露APB寄存器接口，而是通过Ball Domain命令接口进行控制。内部状态寄存器如下：

| 寄存器名称 | 位宽 | 复位值 | 描述 |
|-----------|------|--------|------|
| `state` | 3 | `idle` | 状态机状态: idle/load/exec/store/complete |
| `rob_id_reg` | 10 | 0 | 当前处理指令的ROB ID |
| `iter_reg` | 10 | 0 | 迭代次数寄存器 |
| `iter_cnt` | 10 | 0 | 迭代计数器 |
| `op1_bank_reg` | 2 | 0 | 操作数Bank寄存器 |
| `op1_addr_reg` | 12 | 0 | 操作数地址寄存器 |
| `wr_bank_reg` | 2 | 0 | 写回Bank寄存器 |
| `wr_addr_reg` | 12 | 0 | 写回地址寄存器 |
| `is_acc_reg` | 1 | 0 | 写回目标类型寄存器 |
| `load_cnt` | 4 | 0 | 加载计数器 (跟踪SRAM读延迟) |
| `exec_cnt` | 5 | 0 | 执行计数器 (跟踪流水线进度) |
| `iter_cnt` | 32 | 0 | 批次迭代计数器 (用于status.iter) |
| `has_input` | 1 | 0 | 输入状态标志 (用于status跟踪) |
| `has_output` | 1 | 0 | 输出状态标志 (用于status跟踪) |

### 4.2 状态机编码

| 状态名称 | 编码 | 描述 |
|---------|------|------|
| `idle` | 3'b000 | 空闲状态，等待命令 |
| `load` | 3'b001 | 加载数据状态 |
| `exec` | 3'b010 | 执行计算状态 |
| `store` | 3'b011 | 写回结果状态 |
| `complete` | 3'b100 | 完成响应状态 |


## 5. 功能描述 (Functional Description)

### 5.1 操作流程

#### 5.1.1 指令接收 (Idle → Load)

1. **空闲等待**: 状态机处于`idle`状态，监听`cmdReq.valid`信号
2. **指令解码**: 当`cmdReq.valid && cmdReq.ready`时，捕获指令参数：
   ```scala
   rob_id_reg := cmdReq.bits.rob_id
   iter_reg := cmdReq.bits.iter
   op1_bank_reg := cmdReq.bits.op1_bank
   op1_addr_reg := cmdReq.bits.op1_bank_addr
   wr_bank_reg := cmdReq.bits.wr_bank
   wr_addr_reg := cmdReq.bits.wr_bank_addr
   is_acc_reg := cmdReq.bits.is_acc
   ```
3. **状态转移**: 转移到`load`状态

#### 5.1.2 数据加载 (Load)

1. **发起读请求**: 向SRAM发起读请求
   ```scala
   sramRead(op1_bank_reg).req.valid := true.B
   sramRead(op1_bank_reg).req.bits.addr := op1_addr_reg + iter_cnt
   ```
2. **数据接收**: 等待`sramRead.resp.valid`，将数据存入缓冲寄存器
3. **迭代控制**: 每次加载完成后，`iter_cnt`递增
4. **状态转移**: 当所有数据加载完成时，转移到`exec`状态

#### 5.1.3 GELU计算 (Execute)

执行流水线化的GELU近似计算，每个向量元素并行处理：

**步骤1**: 计算x³
```scala
val x2 = x * x
val x3 = x2 * x
```

**步骤2**: 计算内层多项式
```scala
val poly = x + (x3 * 0.044715.F(32.BP))
```

**步骤3**: 缩放
```scala
val scaled = poly * 0.7978845608.F(32.BP)
```

**步骤4**: Tanh近似
```scala
val tanh_out = tanhApprox(scaled)
```

**步骤5**: 最终组合
```scala
val gelu_out = 0.5.F(32.BP) * x * (1.F(32.BP) + tanh_out)
```

#### 5.1.4 结果写回 (Store)

1. **目标选择**: 根据`is_acc_reg`决定写入SRAM或ACC
2. **写请求**:
   ```scala
   when(is_acc_reg) {
     accWrite(wr_bank_reg).valid := true.B
     accWrite(wr_bank_reg).bits.addr := wr_addr_reg + iter_cnt
     accWrite(wr_bank_reg).bits.data := gelu_result
   }.otherwise {
     sramWrite(wr_bank_reg).valid := true.B
     sramWrite(wr_bank_reg).bits.addr := wr_addr_reg + iter_cnt
     sramWrite(wr_bank_reg).bits.data := gelu_result
   }
   ```
3. **迭代控制**: 所有结果写回完成后，转移到`complete`状态

#### 5.1.5 完成响应 (Complete)

1. **发送完成信号**:
   ```scala
   cmdResp.valid := true.B
   cmdResp.bits.rob_id := rob_id_reg
   cmdResp.bits.commit := true.B
   ```
2. **状态复位**: 返回`idle`状态，准备接收下一条指令

### 5.2 Tanh近似算法

由于硬件实现完整的tanh函数代价高昂，采用分段线性近似或查找表方法：

#### 方案1: 查找表 (LUT)

- **输入范围**: [-4, 4]，分为256个区间
- **表项**: 每个区间存储斜率和截距
- **插值**: 线性插值计算精确值
- **误差**: < 0.01

#### 方案2: 分段多项式

将输入域划分为多个区间，每个区间用二阶多项式近似：

```
tanh(x) ≈ a₂x² + a₁x + a₀  (for x ∈ [x_min, x_max])
```

- **区间数**: 8-16个区间
- **系数存储**: 每区间3个系数 (a₀, a₁, a₂)
- **误差**: < 0.001

### 5.3 数据格式支持

#### 5.3.1 浮点数格式 (FP32)

- **符号位**: 1位
- **指数位**: 8位 (偏移127)
- **尾数位**: 23位
- **范围**: ±1.4E-45 至 ±3.4E+38
- **精度**: 约7位十进制

#### 5.3.2 半精度浮点 (FP16)

- **符号位**: 1位
- **指数位**: 5位 (偏移15)
- **尾数位**: 10位
- **优势**: 节省50%存储和带宽

#### 5.3.3 定点数格式 (INT8/INT16)

对于量化模型，支持定点数GELU计算：

- **量化参数**: scale, zero_point
- **计算流程**:
  1. 反量化: `x_fp = (x_int - zero_point) * scale`
  2. GELU计算: `y_fp = GELU(x_fp)`
  3. 量化: `y_int = round(y_fp / scale) + zero_point`

### 5.4 向量化处理

GELU单元支持向量级并行计算，典型配置为16通道：

```
Input Vector:  [x₀, x₁, x₂, ..., x₁₅]
                │   │   │       │
           ┌────┼───┼───┼───────┼────┐
           │ GELU Compute Array (16x) │
           └────┼───┼───┼───────┼────┘
                │   │   │       │
Output Vector: [y₀, y₁, y₂, ..., y₁₅]
```

每个通道独立计算，共享控制逻辑但使用独立的数据路径。

### 5.5 流水线控制

4级流水线允许不同阶段重叠执行：

| 周期 | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|-----|---------|---------|---------|---------|
| T0  | Decode  | -       | -       | -       |
| T1  | Decode  | Load    | -       | -       |
| T2  | Decode  | Load    | Exec    | -       |
| T3  | Decode  | Load    | Exec    | Store   |
| T4  | Idle    | Load    | Exec    | Store   |

**吞吐率**: 在连续向量处理时，可达1次GELU/周期（每次处理16个元素）


## 6. 时序特性 (Timing Characteristics)

### 6.1 延迟分析

| 操作阶段 | 周期数 | 说明 |
|---------|-------|------|
| 指令解码 (ID) | 1 | 命令参数捕获 |
| 数据加载 (Load) | 3-4 | SRAM读延迟 |
| GELU计算 (Exec) | 8-10 | 多级乘法和tanh近似 |
| 结果写回 (Store) | 2-3 | SRAM/ACC写延迟 |
| 完成响应 (Complete) | 1 | ROB通知 |
| **总延迟** | **15-19** | 典型值16周期 |

### 6.2 关键路径

最长组合逻辑路径出现在Execute阶段的乘法器链：

```
x → MUL(x²) → MUL(x³) → MUL(poly) → TANH → MUL(final)
```

**优化策略**:
1. 插入流水线寄存器，将8-10周期的计算分解为多个子阶段
2. 使用Booth编码乘法器减少关键路径
3. Tanh LUT采用双端口SRAM提高访问速度


## 7. 配置参数 (Configuration Parameters)

### 7.1 编译时参数

通过`CustomBuckyBallConfig`配置：

| 参数名称 | 类型 | 默认值 | 描述 |
|---------|------|--------|------|
| `veclane` | Int | 16 | 向量通道数 |
| `sp_banks` | Int | 4 | Scratchpad Bank数量 |
| `acc_banks` | Int | 2 | Accumulator Bank数量 |
| `spad_bank_entries` | Int | 1024 | 每个SRAM Bank条目数 |
| `acc_bank_entries` | Int | 512 | 每个ACC Bank条目数 |
| `inputType` | DataType | FP32 | 输入数据类型 |
| `outputType` | DataType | FP32 | 输出数据类型 |
| `gelu_pipeline_depth` | Int | 10 | GELU流水线深度 |
| `tanh_lut_size` | Int | 256 | Tanh查找表大小 |

### 7.2 运行时参数

通过Ball Domain指令传递：

| 参数 | 位宽 | 范围 | 描述 |
|-----|------|------|------|
| `iter` | 10 | 1-1024 | 向量迭代次数 |
| `op1_bank` | 2 | 0-3 | 输入数据Bank |
| `op1_bank_addr` | 12 | 0-4095 | 输入起始地址 |
| `wr_bank` | 2 | 0-3 | 输出数据Bank |
| `wr_bank_addr` | 12 | 0-4095 | 输出起始地址 |
| `is_acc` | 1 | 0-1 | 输出目标 (0=SRAM, 1=ACC) |


## 9. 验证方案 (Verification Plan)

### 9.1 功能验证

#### 9.1.1 单元测试

- **基本功能**: 单个元素GELU计算正确性
- **边界条件**: 零值、最大值、最小值、NaN、Inf
- **向量处理**: 16元素并行计算一致性
- **迭代处理**: 多次迭代的地址和数据正确性

#### 9.1.2 精度验证

与软件参考模型（PyTorch/NumPy）对比：

```python
import torch
import numpy as np

# Reference GELU
def gelu_ref(x):
    return torch.nn.functional.gelu(x)

# Test vectors
test_inputs = torch.randn(1000, 16)
golden_outputs = gelu_ref(test_inputs)

# Compare with hardware outputs
max_error = torch.max(torch.abs(hw_outputs - golden_outputs))
mean_error = torch.mean(torch.abs(hw_outputs - golden_outputs))

assert max_error < 1e-3, f"Max error {max_error} exceeds threshold"
```

### 9.3 集成验证

- **完整系统**: 在ToyBuckyBall环境中集成测试
- **编译器支持**: 验证MLIR lowering生成正确的GELU指令
- **端到端**: 运行完整的Transformer模型，验证功能和性能


## 10. 软件接口 (Software Interface)

### 10.1 MLIR Dialect扩展

在Buckyball Dialect中添加GELU操作：

```mlir
// MLIR IR
%output = buckyball.gelu %input : tensor<1024xf32>

// Lowering to hardware intrinsic
func.func @gelu_layer(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index

  // Issue GELU instruction to hardware
  buckyball.gelu.hw %arg0, %arg1, %c0, %c1024
    {op1_bank = 0, wr_bank = 1, is_acc = 0}

  return
}
```

### 10.2 C/C++ Intrinsics

提供底层硬件访问接口：

```c
// C intrinsic
void gelu_hw(
  float* input,        // Input vector address
  float* output,       // Output vector address
  int iter,            // Number of vectors (each 16 elements)
  int op1_bank,        // Input bank
  int op1_addr,        // Input address offset
  int wr_bank,         // Output bank
  int wr_addr,         // Output address offset
  bool is_acc          // Write to accumulator
) {
  // Encode instruction
  uint64_t inst = encode_gelu_inst(
    iter, op1_bank, op1_addr, wr_bank, wr_addr, is_acc
  );

  // Issue RoCC instruction
  ROCC_INSTRUCTION(GELU_OPCODE, inst);

  // Wait for completion (optional)
  wait_gelu_complete();
}
```

### 10.3 编译器优化

#### Fusion优化

将连续的GELU操作与其他算子融合：

```
// Before
%1 = matmul(%A, %B)
%2 = add(%1, %bias)
%3 = gelu(%2)

// After (fused)
%3 = matmul_add_gelu(%A, %B, %bias)
```

#### Tiling优化

大张量分块处理，优化内存访问：

```
// Original
%out = gelu(%in : tensor<10240xf32>)

// Tiled (64 iterations of 16 elements)
for i in 0..64:
  %tile_in = extract(%in, i*16, 16)
  %tile_out = gelu(%tile_in)
  insert(%out, %tile_out, i*16)
```


## 11. 使用示例 (Usage Examples)

### 11.1 基本用法

```scala
// Instantiate GELU unit
val geluUnit = Module(new GeluUnit)

// Connect to Ball Domain
geluUnit.io.cmdReq <> ballDomain.io.geluReq
ballDomain.io.geluResp <> geluUnit.io.cmdResp

// Connect to memory system
for (i <- 0 until sp_banks) {
  scratchpad.io.read(i) <> geluUnit.io.sramRead(i)
  scratchpad.io.write(i) <> geluUnit.io.sramWrite(i)
}

for (i <- 0 until acc_banks) {
  accumulator.io.read(i) <> geluUnit.io.accRead(i)
  accumulator.io.write(i) <> geluUnit.io.accWrite(i)
}

// Status monitoring (optional)
// Can be connected to performance counters or debug interface
val geluStatus = geluUnit.io.status
```

### 11.2 单次向量GELU

```c
// Process single vector (16 elements)
float input[16] = {-2.0, -1.5, ..., 2.0};
float output[16];

// Load input to SRAM bank 0, address 0x100
load_to_sram(0, 0x100, input, 16);

// Issue GELU instruction
gelu_hw(
  input, output,
  1,      // iter = 1 (single vector)
  0,      // op1_bank = 0
  0x100,  // op1_addr
  1,      // wr_bank = 1
  0x200,  // wr_addr
  false   // is_acc = false (write to SRAM)
);

// Read result from SRAM bank 1, address 0x200
read_from_sram(1, 0x200, output, 16);
```

### 11.3 批量处理

```c
// Process 1024 elements (64 vectors)
#define N 1024
#define VECLANE 16
#define ITERS (N / VECLANE)

float input[N], output[N];

// Initialize input data
for (int i = 0; i < N; i++) {
  input[i] = (float)i / 100.0 - 5.0;
}

// Load to SRAM
dma_to_sram(0, 0, input, N);

// Issue batched GELU
gelu_hw(
  NULL, NULL,
  ITERS,  // iter = 64
  0,      // op1_bank = 0
  0,      // op1_addr = 0
  0,      // wr_bank = 0 (in-place)
  0,      // wr_addr = 0
  false   // is_acc = false
);

// Read results
dma_from_sram(0, 0, output, N);
```

### 11.4 与MATMUL流水线

```c
// Transformer FFN layer: Y = GELU(XW + b)

// Step 1: Matrix multiplication
matmul_hw(X, W, XW, M, N, K);

// Step 2: Add bias
vecadd_hw(XW, b, XWb, M * N);

// Step 3: GELU activation
gelu_hw(XWb, Y, (M * N) / VECLANE, ...);
```

### 11.5 Status信号监控

Status信号可用于性能分析、调试和系统监控：

```scala
// 性能计数器集成
val perfCounters = Module(new PerfCounters)
perfCounters.io.gelu_idle := geluUnit.io.status.idle
perfCounters.io.gelu_running := geluUnit.io.status.running

// 调试时等待GELU完成
def waitGeluComplete(): Unit = {
  while (!geluUnit.io.status.idle) {
    // 可以输出中间状态
    if (geluUnit.io.status.init) {
      printf("GELU: Loading data...\n")
    } else if (geluUnit.io.status.running) {
      printf("GELU: Computing...\n")
    }
  }
  printf("GELU: Completed %d batches\n", geluUnit.io.status.iter)
}

// 流水线协调
when(geluUnit.io.status.ready && matmulUnit.io.status.complete) {
  // GELU ready and MATMUL finished, issue next GELU
  geluUnit.io.cmdReq.valid := true.B
}
```

**Status信号应用场景**：

1. **性能监控**: 统计各单元的利用率和空闲时间
2. **流水线调度**: 根据status协调多个加速单元的数据流
3. **调试分析**: 追踪指令执行进度和状态转换
4. **功耗管理**: 在idle状态下进行时钟门控优化
5. **错误检测**: 检测异常的状态转换或长时间停滞
