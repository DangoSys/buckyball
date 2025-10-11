# LayerNorm加速单元设计规范

## 1. 概述 (Overview)

LayerNorm (Layer Normalization) 加速单元是BuckyBall框架中的专用计算加速器，用于高效执行层归一化运算。LayerNorm是现代深度学习模型（如Transformer、BERT、GPT等）中广泛使用的归一化技术，能够稳定训练过程并提升模型性能。

### 1.1 基本参数

- **数据格式**: 输入为INT8/INT32，输出为INT8/INT32
- **向量化处理**: 每次处理16个元素（veclane=16）
- **流水线架构**: 多级流水线设计（ID, Load, Reduce, Normalize, Store）
- **计算方法**: 计算均值、方差，执行归一化和仿射变换
- **存储接口**: 支持Scratchpad和Accumulator读写
- **支持模式**: 可配置的归一化维度和可选的gamma/beta参数

### 1.2 数学定义

LayerNorm的标准定义：

```
给定输入向量 x = [x₀, x₁, ..., x_{N-1}]

1. 计算均值：
   μ = (1/N) * Σᵢ xᵢ

2. 计算方差：
   σ² = (1/N) * Σᵢ (xᵢ - μ)²

3. 归一化：
   x̂ᵢ = (xᵢ - μ) / √(σ² + ε)

4. 仿射变换（可选）：
   yᵢ = γᵢ * x̂ᵢ + βᵢ
```

其中：
- N 是归一化维度大小
- ε 是防止除零的小常数（典型值：1e-5）
- γ (gamma) 和 β (beta) 是可学习的仿射参数

**简化模式**：
当不使用仿射变换时，输出为 yᵢ = x̂ᵢ

## 2. 系统架构 (Block Diagram)

### 2.1 顶层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     LayerNorm Accelerator                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │              │    │              │    │              │      │
│  │   Control    │───▶│  Load Unit   │───▶│   Reduce    │      │
│  │   Unit (ID)  │    │              │    │  Unit (Mean) │      │
│  │              │    │              │    │              │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │              │
│         │                   │                   ▼              │
│         │              ┌────▼────┐         ┌────────────┐      │
│         │              │ SRAM    │         │  Variance  │      │
│         │              │ Read    │         │   Compute  │      │
│         │              │ Arbiter │         └──────┬─────┘      │
│         │              └─────────┘                │            │
│         │                                         ▼            │
│         │                                  ┌──────────────┐    │
│         │                                  │  Normalize   │    │
│         │                                  │    Unit      │    │
│         │                                  └──────┬───────┘    │
│         │                                         │            │
│         │                                  ┌──────▼───────┐    │
│         │                                  │   Affine     │    │
│         │                                  │ Transform    │    │
│         │                                  └──────┬───────┘    │
│         │                                         │            │
│  ┌──────▼────────────────────────────────┐        │            │
│  │         Command Interface             │        │            │
│  │   (Ball Bus / RoCC Interface)         │        │            │
│  └───────────────────────────────────────┘        │            │
│                                                   │            │
│  ┌──────────────────────────────────────┐         │            │
│  │         Status Monitor               │         │            │
│  │  (ready/valid/idle/init/running)     │         │            │
│  └──────────────────────────────────────┘         │            │
│                                                   │            │
└───────────────────────────────────────────────────┼────────────┘
                                                    │
                                              ┌─────▼─────┐
                                              │  Memory   │
                                              │  System   │
                                              └───────────┘
```

### 2.2 流水线结构

```
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│  ID    │───▶│  Load  │───▶│ Reduce │───▶│ Norm   │───▶│ Store  │
│ Stage  │    │ Stage  │    │ Stage  │    │ Stage  │    │ Stage  │
└────────┘    └────────┘    └────────┘    └────────┘    └────────┘
    │             │             │             │             │
    │             │             │             │             │
  Decode      Load Data    Compute Mean   Normalize &   Write Back
  Command     from SRAM    & Variance     Affine Tx    to ACC/SRAM
```

### 2.3 计算单元架构

```
┌─────────────────────────────────────────────────────────────────┐
│           LayerNorm Compute Pipeline (EX Stage)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input [x₀, x₁, ..., x_{N-1}]                                   │
│     │                                                           │
│     ├──────────────────────────────────┐                        │
│     │                                  │                        │
│  ┌──▼──────┐                     ┌─────▼─────┐                 │
│  │  Sum    │                     │  Square   │                 │
│  │ Reduce  │                     │  & Sum    │                 │
│  └──┬──────┘                     └─────┬─────┘                 │
│     │                                  │                        │
│  ┌──▼──────┐                     ┌─────▼─────┐                 │
│  │  DIV N  │ (mean μ)            │  Buffer   │                 │
│  └──┬──────┘                     └─────┬─────┘                 │
│     │                                  │                        │
│     ├──────────────────────────────────┤                        │
│     │                                  │                        │
│  ┌──▼──────────────────────────────────▼─────┐                 │
│  │  Variance: σ² = E[x²] - μ²                │                 │
│  └──┬──────────────────────────────────────┬──┘                │
│     │                                      │                   │
│  ┌──▼──────┐                               │                   │
│  │ RSQRT   │ (1/√(σ²+ε))                   │                   │
│  └──┬──────┘                               │                   │
│     │                                      │                   │
│     │    ┌─────────────────────────────────┘                   │
│     │    │                                                     │
│  ┌──▼────▼───────┐                                             │
│  │  Normalize    │                                             │
│  │  x̂ᵢ = (xᵢ-μ) * rsqrt                                       │
│  └──┬────────────┘                                             │
│     │                                                           │
│  ┌──▼────────────┐                                             │
│  │  Affine       │ (optional)                                  │
│  │  yᵢ = γᵢ*x̂ᵢ + βᵢ                                           │
│  └──┬────────────┘                                             │
│     │                                                           │
│  Output [y₀, y₁, ..., y_{N-1}]                                  │
└─────────────────────────────────────────────────────────────────┘
```


## 3. 接口描述 (Interface Description)

LayerNorm单元对外提供以下接口：
- **命令接口** (Command Interface): 接收LayerNorm指令并返回完成响应
- **Scratchpad接口** (SRAM Interface): 访问INT8数据的存储器
- **Accumulator接口** (ACC Interface): 访问INT32数据的存储器
- **状态监控接口** (Status Interface): 输出当前运行状态信息
- **时钟和复位接口**: 提供时钟和复位信号

### 3.1 指令语义 (Instruction Semantics)

一条LayerNorm指令的完整语义如下：

**指令含义**：对存储在Scratchpad或Accumulator中的向量执行LayerNorm运算

**数据格式**：
- **Scratchpad模式** (`is_acc=0`)：INT8输入 → LayerNorm → INT8输出
- **Accumulator模式** (`is_acc=1`)：INT32输入 → LayerNorm → INT32输出
- **注意**：Scratchpad存储INT8，Accumulator存储INT32

**处理单位**：
- 每个向量 = 16个元素（veclane = 16）
- 每个SRAM地址存储1个INT8向量（16×8位 = 128位 = 16字节）
- 每个ACC地址存储1个INT32向量（16×32位 = 512位 = 64字节）

**指令参数说明**：

| 参数 | 含义 | 示例 |
|-----|------|------|
| `iter` | 要处理的batch数量 | iter=64 表示处理64个独立的LayerNorm |
| `op1_bank` | 输入数据所在的Bank号 | 0-3 |
| `op1_bank_addr` | 输入起始地址 | 0x100 |
| `wr_bank` | 输出数据写入的Bank号 | 0-3 |
| `wr_bank_addr` | 输出起始地址 | 0x200 |
| `is_acc` | 数据类型选择 | 0=SRAM(INT8)模式, 1=ACC(INT32)模式 |
| `special[11:0]` | 归一化维度大小（向量数） | norm_dim，单位为向量数 |
| `special[23:12]` | Gamma参数地址（可选） | gamma_addr，0表示不使用 |
| `special[35:24]` | Beta参数地址（可选） | beta_addr，0表示不使用 |
| `special[37:36]` | Gamma/Beta所在Bank | param_bank |
| `special[38]` | 使能仿射变换 | use_affine (1=使用gamma/beta) |
| `special[39]` | 保留位 | reserved |

**Special字段编码**（40位）：

```
 39    38        37:36      35:24        23:12        11:0
┌──┬─────┬───────────┬────────────┬────────────┬──────────┐
│Rv│ Aff │ParamBank │ Beta Addr  │ Gamma Addr │ NormDim  │
└──┴─────┴───────────┴────────────┴────────────┴──────────┘

NormDim:    归一化维度（向量数），范围1-4096
GammaAddr:  Gamma参数起始地址（相对于ParamBank）
BetaAddr:   Beta参数起始地址（相对于ParamBank）
ParamBank:  Gamma/Beta参数所在Bank号
Aff:        仿射变换使能（1=使用gamma/beta，0=仅归一化）
Rv:         保留位
```

**处理语义**：

对于每个batch（共iter个batch）：
1. 从 `op1_bank[op1_bank_addr + batch_idx * norm_dim]` 读取norm_dim个向量
2. 计算这norm_dim个向量的均值和方差（跨 norm_dim * 16 个元素）
3. 执行归一化：`x̂ = (x - μ) / √(σ² + ε)`
4. 如果`use_affine=1`，执行仿射变换：`y = γ * x̂ + β`
5. 将结果写入 `wr_bank[wr_bank_addr + batch_idx * norm_dim]`

**模式1：is_acc=0（SRAM模式，INT8）**

输入范围（每个batch）：
```
起始地址：SRAM[op1_bank][op1_bank_addr + batch_idx * norm_dim]
结束地址：SRAM[op1_bank][op1_bank_addr + (batch_idx+1) * norm_dim - 1]
每地址：16个INT8元素（128位）
总元素数：norm_dim × 16 个INT8元素
```

输出范围（每个batch）：
```
起始地址：SRAM[wr_bank][wr_bank_addr + batch_idx * norm_dim]
结束地址：SRAM[wr_bank][wr_bank_addr + (batch_idx+1) * norm_dim - 1]
每地址：16个INT8元素（128位）
总元素数：norm_dim × 16 个INT8元素
```

**模式2：is_acc=1（ACC模式，INT32）**

输入范围（每个batch）：
```
起始地址：ACC[op1_bank][op1_bank_addr + batch_idx * norm_dim]
结束地址：ACC[op1_bank][op1_bank_addr + (batch_idx+1) * norm_dim - 1]
每地址：16个INT32元素（512位）
总元素数：norm_dim × 16 个INT32元素
```

输出范围（每个batch）：
```
起始地址：ACC[wr_bank][wr_bank_addr + batch_idx * norm_dim]
结束地址：ACC[wr_bank][wr_bank_addr + (batch_idx+1) * norm_dim - 1]
每地址：16个INT32元素（512位）
总元素数：norm_dim × 16 个INT32元素
```

**示例1**：单batch LayerNorm，归一化512个元素（32个向量）

```
iter = 1
op1_bank = 0, op1_bank_addr = 0x000
wr_bank = 1, wr_bank_addr = 0x000
is_acc = 0
special.norm_dim = 32  (32 vectors × 16 elements = 512 elements)
special.use_affine = 0

输入：SRAM[0][0x000~0x01F] 的32个向量（512个INT8元素）
输出：SRAM[1][0x000~0x01F] 的32个向量（512个INT8元素）
计算：对这512个元素计算均值和方差，执行归一化
```

**示例2**：批量LayerNorm with 仿射变换

```
iter = 64
op1_bank = 0, op1_bank_addr = 0x000
wr_bank = 1, wr_bank_addr = 0x000
is_acc = 1
special.norm_dim = 16  (16 vectors × 16 elements = 256 elements)
special.gamma_addr = 0x100
special.beta_addr = 0x110
special.param_bank = 2
special.use_affine = 1

输入：ACC[0][0x000~0x3FF] 的64个batch，每个batch 16个向量
Gamma: ACC[2][0x100~0x10F] 的16个向量（256个元素）
Beta:  ACC[2][0x110~0x11F] 的16个向量（256个元素）
输出：ACC[1][0x000~0x3FF] 的64个batch，每个batch 16个向量

对于每个batch i（i=0到63）：
  1. 读取ACC[0][i*16 ~ i*16+15]的256个元素
  2. 计算均值μ和方差σ²
  3. 归一化：x̂ = (x - μ) / √(σ² + ε)
  4. 仿射变换：y = γ * x̂ + β（element-wise）
  5. 写入ACC[1][i*16 ~ i*16+15]
```

**示例3**：小维度LayerNorm（Transformer hidden dim = 768）

```
iter = 128
norm_dim = 48  (48 vectors × 16 = 768 elements)
op1_bank = 0, op1_bank_addr = 0x000
wr_bank = 0, wr_bank_addr = 0x1800
is_acc = 1
special.norm_dim = 48
special.use_affine = 0

输入：ACC[0][0x000~0x17FF] 的128个序列，每个768维
输出：ACC[0][0x1800~0x2FFF] 的128个序列，每个768维
```

### 3.2 命令接口 (Command Interface)

LayerNorm单元通过Ball Domain标准接口与系统交互：

| 信号名称 | 方向 | 位宽 | 描述 |
|---------|------|------|------|
| `cmdReq.valid` | Input | 1 | 命令请求有效信号 |
| `cmdReq.ready` | Output | 1 | 命令请求就绪信号 |
| `cmdReq.bits.rob_id` | Input | 10 | ROB (Reorder Buffer) 标识符 |
| `cmdReq.bits.iter` | Input | 10 | Batch迭代次数 (支持1-1024) |
| `cmdReq.bits.op1_bank` | Input | 2 | 操作数Bank选择 |
| `cmdReq.bits.op1_bank_addr` | Input | 12 | 操作数Bank内地址 |
| `cmdReq.bits.wr_bank` | Input | 2 | 写回Bank选择 |
| `cmdReq.bits.wr_bank_addr` | Input | 12 | 写回Bank内地址 |
| `cmdReq.bits.is_acc` | Input | 1 | 目标存储类型 (0=SRAM, 1=ACC) |
| `cmdReq.bits.special` | Input | 40 | 特殊参数字段（编码见3.1节） |

| 信号名称 | 方向 | 位宽 | 描述 |
|---------|------|------|------|
| `cmdResp.valid` | Output | 1 | 完成响应有效信号 |
| `cmdResp.ready` | Input | 1 | 完成响应就绪信号 |
| `cmdResp.bits.rob_id` | Output | 10 | 完成指令的ROB ID |
| `cmdResp.bits.commit` | Output | 1 | 提交标志 |

### 3.3 Scratchpad存储接口 (SRAM Interface)

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

### 3.4 Accumulator存储接口 (ACC Interface)

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

### 3.5 状态监控接口 (Status Interface)

LayerNorm单元提供状态监控接口，用于外部观察当前运行状态：

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

### 3.6 时钟和复位接口

| 信号名称 | 方向 | 描述 |
|---------|------|------|
| `clock` | Input | 全局时钟信号 |
| `reset` | Input | 全局同步复位信号 (高有效) |


## 4. 寄存器映射 (Register Map)

### 4.1 内部控制寄存器

LayerNorm单元不直接暴露APB寄存器接口，而是通过Ball Domain命令接口进行控制。内部状态寄存器如下：

| 寄存器名称 | 位宽 | 复位值 | 描述 |
|-----------|------|--------|------|
| `state` | 3 | `idle` | 状态机状态: idle/load/reduce/norm/store/complete |
| `rob_id_reg` | 10 | 0 | 当前处理指令的ROB ID |
| `iter_reg` | 10 | 0 | Batch迭代次数寄存器 |
| `batch_cnt` | 10 | 0 | Batch计数器 |
| `op1_bank_reg` | 2 | 0 | 操作数Bank寄存器 |
| `op1_addr_reg` | 12 | 0 | 操作数地址寄存器 |
| `wr_bank_reg` | 2 | 0 | 写回Bank寄存器 |
| `wr_addr_reg` | 12 | 0 | 写回地址寄存器 |
| `is_acc_reg` | 1 | 0 | 写回目标类型寄存器 |
| `norm_dim_reg` | 12 | 0 | 归一化维度寄存器 |
| `gamma_addr_reg` | 12 | 0 | Gamma参数地址寄存器 |
| `beta_addr_reg` | 12 | 0 | Beta参数地址寄存器 |
| `param_bank_reg` | 2 | 0 | 参数Bank寄存器 |
| `use_affine_reg` | 1 | 0 | 仿射变换使能寄存器 |
| `vec_cnt` | 12 | 0 | 向量读取计数器 |
| `mean_acc` | 32 | 0 | 均值累加器 |
| `var_acc` | 32 | 0 | 方差累加器 |
| `mean_val` | 32 | 0 | 计算得到的均值 |
| `rsqrt_val` | 32 | 0 | 计算得到的归一化因子 1/√(σ²+ε) |
| `iter_cnt` | 32 | 0 | 批次迭代计数器 (用于status.iter) |
| `has_input` | 1 | 0 | 输入状态标志 (用于status跟踪) |
| `has_output` | 1 | 0 | 输出状态标志 (用于status跟踪) |

### 4.2 状态机编码

| 状态名称 | 编码 | 描述 |
|---------|------|------|
| `idle` | 3'b000 | 空闲状态，等待命令 |
| `load` | 3'b001 | 加载数据并计算统计量状态 |
| `reduce` | 3'b010 | 完成reduce，计算均值和方差 |
| `norm` | 3'b011 | 执行归一化和仿射变换状态 |
| `store` | 3'b100 | 写回结果状态 |
| `complete` | 3'b101 | 完成响应状态 |


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

   // 解码special字段
   norm_dim_reg := cmdReq.bits.special(11, 0)
   gamma_addr_reg := cmdReq.bits.special(23, 12)
   beta_addr_reg := cmdReq.bits.special(35, 24)
   param_bank_reg := cmdReq.bits.special(37, 36)
   use_affine_reg := cmdReq.bits.special(38)
   ```
3. **初始化**: 重置累加器和计数器
4. **状态转移**: 转移到`load`状态

#### 5.1.2 数据加载与归约 (Load & Reduce)

对于当前batch，依次加载norm_dim个向量并累加统计量：

**第一遍：计算均值**
```scala
// 对每个向量
for (vec_idx <- 0 until norm_dim) {
  // 1. 发起读请求
  val addr = op1_addr_reg + batch_cnt * norm_dim + vec_idx
  if (is_acc_reg) {
    accRead(op1_bank_reg).req.valid := true.B
    accRead(op1_bank_reg).req.bits.addr := addr
  } else {
    sramRead(op1_bank_reg).req.valid := true.B
    sramRead(op1_bank_reg).req.bits.addr := addr
  }

  // 2. 接收数据并累加
  when(memResp.valid) {
    val vec_data = memResp.bits.data
    val vec_sum = vec_data.asTypeOf(Vec(16, dataType)).reduce(_ + _)
    mean_acc := mean_acc + vec_sum

    // 同时保存数据到缓冲区（用于第二遍）
    dataBuffer(vec_idx) := vec_data
  }
}

// 计算均值
val total_elements = norm_dim * 16.U
mean_val := mean_acc / total_elements
```

**第二遍：计算方差**
```scala
// 从缓冲区读取数据，计算方差
for (vec_idx <- 0 until norm_dim) {
  val vec_data = dataBuffer(vec_idx)
  val elements = vec_data.asTypeOf(Vec(16, dataType))

  // 计算每个元素与均值的差的平方
  for (elem <- elements) {
    val diff = elem - mean_val
    var_acc := var_acc + diff * diff
  }
}

// 计算方差
val variance = var_acc / total_elements

// 计算归一化因子 rsqrt = 1 / √(variance + epsilon)
rsqrt_val := rsqrtApprox(variance + epsilon)
```

#### 5.1.3 归一化与仿射变换 (Normalize)

对缓冲区中的每个向量执行归一化：

```scala
for (vec_idx <- 0 until norm_dim) {
  val vec_data = dataBuffer(vec_idx).asTypeOf(Vec(16, dataType))

  // 归一化
  val normalized = VecInit(vec_data.map { x =>
    (x - mean_val) * rsqrt_val
  })

  // 仿射变换（可选）
  val output = if (use_affine_reg) {
    // 读取gamma和beta
    val gamma = gammaBuffer(vec_idx)
    val beta = betaBuffer(vec_idx)

    VecInit(normalized.zip(gamma).zip(beta).map {
      case ((x_norm, g), b) => g * x_norm + b
    })
  } else {
    normalized
  }

  // 存入输出缓冲
  outputBuffer(vec_idx) := output.asUInt
}
```

**加载Gamma和Beta（如果使用仿射变换）**：

```scala
when(use_affine_reg) {
  // 并行加载gamma和beta参数
  for (vec_idx <- 0 until norm_dim) {
    // 读取gamma
    val gamma_addr = gamma_addr_reg + vec_idx
    memRead(param_bank_reg, gamma_addr)
    gammaBuffer(vec_idx) := memResp.bits.data

    // 读取beta
    val beta_addr = beta_addr_reg + vec_idx
    memRead(param_bank_reg, beta_addr)
    betaBuffer(vec_idx) := memResp.bits.data
  }
}
```

#### 5.1.4 结果写回 (Store)

将输出缓冲中的数据写回内存：

```scala
for (vec_idx <- 0 until norm_dim) {
  val wr_addr = wr_addr_reg + batch_cnt * norm_dim + vec_idx

  when(is_acc_reg) {
    accWrite(wr_bank_reg).valid := true.B
    accWrite(wr_bank_reg).bits.addr := wr_addr
    accWrite(wr_bank_reg).bits.data := outputBuffer(vec_idx)
    accWrite(wr_bank_reg).bits.mask := Fill(16, 1.U)
  }.otherwise {
    sramWrite(wr_bank_reg).valid := true.B
    sramWrite(wr_bank_reg).bits.addr := wr_addr
    sramWrite(wr_bank_reg).bits.data := outputBuffer(vec_idx)
    sramWrite(wr_bank_reg).bits.mask := Fill(16, 1.U)
  }
}

// 更新batch计数器
batch_cnt := batch_cnt + 1.U

// 判断是否完成所有batch
when(batch_cnt === iter_reg) {
  state := complete
}.otherwise {
  state := load  // 处理下一个batch
}
```

#### 5.1.5 完成响应 (Complete)

```scala
cmdResp.valid := true.B
cmdResp.bits.rob_id := rob_id_reg
cmdResp.bits.commit := true.B

when(cmdResp.fire) {
  state := idle
  batch_cnt := 0.U
}
```

### 5.2 Reciprocal Square Root (RSQRT) 近似算法

由于硬件实现完整的平方根倒数代价高昂，采用Newton-Raphson迭代法或查找表方法：

#### 方案1: 查找表 + 线性插值 (LUT)

- **输入范围**: [0.001, 100]（覆盖常见方差范围）
- **LUT大小**: 256条目，存储初始近似值
- **插值**: 线性插值提高精度
- **误差**: < 0.1%

```scala
def rsqrtLUT(variance: UInt): UInt = {
  val v = variance + epsilon
  val exp = getExponent(v)
  val mantissa = getMantissa(v)

  // 查表获取初始值
  val idx = mantissa(22, 15)  // 取高8位作为索引
  val lut_val = rsqrtTable(idx)

  // 线性插值
  val offset = mantissa(14, 0)
  val slope = rsqrtTable(idx + 1) - rsqrtTable(idx)
  val result = lut_val + (slope * offset >> 15)

  // 调整指数
  adjustExponent(result, exp)
}
```

#### 方案2: Newton-Raphson 迭代

使用2-3次Newton-Raphson迭代快速收敛：

```
x_{n+1} = x_n * (3 - v * x_n²) / 2

其中 v = σ² + ε，x_0 从LUT获取初始近似
```

```scala
def rsqrtNewton(v: Float): Float = {
  var x = rsqrtLUT(v)  // 初始近似

  // 迭代3次
  for (i <- 0 until 3) {
    x = x * (3.0f - v * x * x) * 0.5f
  }

  x
}
```

**硬件实现特性**：
- **延迟**: LUT方法 3-4周期，Newton-Raphson 8-10周期
- **精度**: LUT < 1%, Newton-Raphson < 0.01%
- **资源**: LUT需要1KB SRAM，Newton-Raphson需要3个乘法器

### 5.3 数据格式支持

#### 5.3.1 INT32格式（定点数）

- **表示**: Q16.16定点数（16位整数，16位小数）
- **范围**: -32768.0 至 +32767.999
- **精度**: 2^-16 ≈ 0.000015

```scala
// 定点数乘法
def fixedMul(a: UInt, b: UInt): UInt = {
  val product = (a.asSInt * b.asSInt).asUInt
  product(47, 16)  // 保留高32位
}

// 定点数除法
def fixedDiv(a: UInt, b: UInt): UInt = {
  val dividend = Cat(a, 0.U(16.W))  // 左移16位
  dividend / b
}
```

#### 5.3.2 INT8格式（量化）

对于INT8输入，内部使用INT32累加器，输出时再量化回INT8：

```scala
// 输入反量化
def dequantize(x: SInt, scale: Float, zero_point: Int): Float = {
  (x.asSInt - zero_point) * scale
}

// 输出量化
def quantize(y: Float, scale: Float, zero_point: Int): SInt = {
  val q = (y / scale).round + zero_point
  q.min(127).max(-128).asSInt
}
```

#### 5.3.3 浮点数格式 (FP32) - 未来扩展

- **符号位**: 1位
- **指数位**: 8位 (偏移127)
- **尾数位**: 23位
- 支持标准IEEE 754浮点运算

### 5.4 向量化处理

LayerNorm单元支持向量级并行计算，典型配置为16通道：

```
Batch of Input Vectors: [V₀, V₁, V₂, ..., V_{norm_dim-1}]
                         │   │   │           │
                         └───┴───┴───────────┘
                                 │
                        ┌────────▼────────┐
                        │  Reduce Sum     │ (并行累加树)
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  Mean & Var     │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  Normalize      │ (16通道并行)
                        └────────┬────────┘
                                 │
Output Vectors:         [Y₀, Y₁, Y₂, ..., Y_{norm_dim-1}]
```

**并行累加树**：
- 使用树形归约结构，log2(16) = 4级
- 每级延迟1周期，总延迟4周期
- 支持INT8/INT32/FP32数据类型

### 5.5 流水线控制

5级流水线允许不同阶段重叠执行：

| 周期 | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|-----|---------|---------|---------|---------|---------|
| T0  | Decode  | -       | -       | -       | -       |
| T1  | -       | Load    | -       | -       | -       |
| T2  | -       | Load    | Reduce  | -       | -       |
| T3  | -       | Load    | Reduce  | Norm    | -       |
| T4  | -       | -       | Reduce  | Norm    | Store   |
| T5  | Decode  | Load    | Reduce  | Norm    | Store   |

**批处理优化**：
- 多个batch可以pipeline执行
- 当前batch在Normalize阶段时，下一个batch可以开始Load
- 理论吞吐率：每 (norm_dim + pipeline_depth) 周期完成1个batch


## 6. 时序特性 (Timing Characteristics)

### 6.1 延迟分析

| 操作阶段 | 周期数 | 说明 |
|---------|-------|------|
| 指令解码 (ID) | 1 | 命令参数捕获和special字段解码 |
| 数据加载 (Load) | norm_dim × 3 | 每个向量3周期SRAM读延迟 |
| 归约计算 (Reduce) | 4 + 20 | 树形归约4周期 + 方差计算20周期 |
| RSQRT计算 | 8-10 | Newton-Raphson迭代或LUT |
| 归一化 (Norm) | norm_dim × 2 | 每个向量2周期（norm + affine） |
| 参数加载 (Gamma/Beta) | norm_dim × 3 | 与Load并行执行（如果使用） |
| 结果写回 (Store) | norm_dim × 2 | 每个向量2周期写延迟 |
| 完成响应 (Complete) | 1 | ROB通知 |

**单batch总延迟**（典型norm_dim=32）：

```
T_total = 1 + (32×3) + 24 + 8 + (32×2) + (32×2) + 1
        = 1 + 96 + 24 + 8 + 64 + 64 + 1
        = 258 周期
```

**批处理吞吐率**（iter个batch）：

由于流水线重叠，实际周期数：
```
T_batch = T_decode + iter × (T_load + T_reduce + T_norm + T_store)
        ≈ iter × (norm_dim × 7 + 32)

对于iter=64, norm_dim=32:
T_batch ≈ 64 × (224 + 32) = 16384 周期
```

### 6.2 关键路径

最长组合逻辑路径：

**路径1: 累加树归约**
```
16个元素 → 4级加法树 → 累加器更新
```

**路径2: 方差计算**
```
元素 → SUB(x-μ) → MUL(diff²) → 累加
```

**路径3: 归一化**
```
元素 → SUB(x-μ) → MUL(rsqrt) → MUL(gamma) → ADD(beta)
```

**优化策略**:
1. 插入流水线寄存器，将长路径分解为多个子阶段
2. 使用Booth编码乘法器减少关键路径延迟
3. 采用快速累加器结构（Kogge-Stone加法器）
4. RSQRT查找表采用双端口SRAM提高访问速度
5. 双缓冲机制：加载下一个batch时处理当前batch


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
| `inputType` | DataType | INT32 | 输入数据类型 |
| `outputType` | DataType | INT32 | 输出数据类型 |
| `layernorm_pipeline_depth` | Int | 5 | LayerNorm流水线深度 |
| `rsqrt_method` | String | "newton" | RSQRT计算方法 (newton/lut) |
| `rsqrt_lut_size` | Int | 256 | RSQRT查找表大小 |
| `max_norm_dim` | Int | 4096 | 支持的最大归一化维度 |
| `epsilon` | Float | 1e-5 | 防止除零的小常数 |
| `enable_affine` | Boolean | true | 是否支持仿射变换 |
| `data_buffer_depth` | Int | 128 | 数据缓冲深度（向量数） |

### 7.2 运行时参数

通过Ball Domain指令传递：

| 参数 | 位宽 | 范围 | 描述 |
|-----|------|------|------|
| `iter` | 10 | 1-1024 | Batch迭代次数 |
| `op1_bank` | 2 | 0-3 | 输入数据Bank |
| `op1_bank_addr` | 12 | 0-4095 | 输入起始地址 |
| `wr_bank` | 2 | 0-3 | 输出数据Bank |
| `wr_bank_addr` | 12 | 0-4095 | 输出起始地址 |
| `is_acc` | 1 | 0-1 | 数据类型 (0=SRAM, 1=ACC) |
| `special.norm_dim` | 12 | 1-4096 | 归一化维度（向量数） |
| `special.gamma_addr` | 12 | 0-4095 | Gamma参数地址 |
| `special.beta_addr` | 12 | 0-4095 | Beta参数地址 |
| `special.param_bank` | 2 | 0-3 | 参数所在Bank |
| `special.use_affine` | 1 | 0-1 | 使能仿射变换 |


## 8. 验证方案 (Verification Plan)

### 8.1 功能验证

#### 8.1.1 单元测试

- **基本功能**: 单个batch的LayerNorm计算正确性
- **边界条件**: 零方差、极大/极小值、norm_dim=1到4096
- **向量处理**: 16元素并行计算一致性
- **批处理**: 多batch迭代的地址和数据正确性
- **仿射变换**: Gamma和Beta应用的正确性
- **数据格式**: INT8/INT32模式切换

#### 8.1.2 精度验证

与软件参考模型（PyTorch/NumPy）对比：

```python
import torch
import numpy as np

# Reference LayerNorm
def layernorm_ref(x, gamma=None, beta=None, eps=1e-5):
    layer_norm = torch.nn.LayerNorm(x.shape[-1], eps=eps)
    if gamma is not None:
        layer_norm.weight.data = gamma
    if beta is not None:
        layer_norm.bias.data = beta
    return layer_norm(x)

# Test vectors
batch_size = 64
norm_dim = 512
test_inputs = torch.randn(batch_size, norm_dim)
gamma = torch.randn(norm_dim)
beta = torch.randn(norm_dim)

golden_outputs = layernorm_ref(test_inputs, gamma, beta)

# Compare with hardware outputs
max_error = torch.max(torch.abs(hw_outputs - golden_outputs))
mean_error = torch.mean(torch.abs(hw_outputs - golden_outputs))
relative_error = mean_error / torch.mean(torch.abs(golden_outputs))

print(f"Max Error: {max_error:.6f}")
print(f"Mean Error: {mean_error:.6f}")
print(f"Relative Error: {relative_error:.2%}")

assert relative_error < 0.01, f"Error {relative_error} exceeds threshold"
```

#### 8.1.3 性能验证

- **吞吐率**: 测量实际周期数，验证是否达到理论值
- **延迟**: 单batch端到端延迟测量
- **内存带宽**: SRAM/ACC访问带宽利用率
- **流水线效率**: 多batch时的流水线填充率

### 8.2 集成验证

- **完整系统**: 在ToyBuckyBall环境中集成测试
- **编译器支持**: 验证MLIR lowering生成正确的LayerNorm指令
- **端到端**: 运行完整的Transformer模型，验证功能和性能
- **多Ball协作**: 与MATMUL、GELU等Ball单元配合测试


## 9. 软件接口 (Software Interface)

### 9.1 MLIR Dialect扩展

在Buckyball Dialect中添加LayerNorm操作：

```mlir
// MLIR IR - 基本LayerNorm
%output = buckyball.layernorm %input : tensor<64x512xf32>

// MLIR IR - 带仿射变换的LayerNorm
%output = buckyball.layernorm %input, %gamma, %beta
  : tensor<64x512xf32>, tensor<512xf32>, tensor<512xf32>

// Lowering to hardware intrinsic
func.func @layernorm_layer(
  %arg0: memref<64x512xf32>,  // input
  %arg1: memref<512xf32>,     // gamma
  %arg2: memref<512xf32>,     // beta
  %arg3: memref<64x512xf32>   // output
) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index  // norm_dim in vectors (512/16)

  // Issue LayerNorm instruction to hardware
  buckyball.layernorm.hw %arg0, %arg1, %arg2, %arg3
    {
      iter = 64,
      norm_dim = 32,
      op1_bank = 0,
      wr_bank = 1,
      is_acc = 1,
      use_affine = 1
    }

  return
}
```

### 9.2 C/C++ Intrinsics

提供底层硬件访问接口：

```c
// C intrinsic for LayerNorm
void layernorm_hw(
  void* input,         // Input data address
  void* output,        // Output data address
  void* gamma,         // Gamma parameters (optional, NULL if not used)
  void* beta,          // Beta parameters (optional, NULL if not used)
  int batch,           // Number of batches
  int norm_dim,        // Normalization dimension (in vectors)
  int op1_bank,        // Input bank
  int op1_addr,        // Input address offset
  int wr_bank,         // Output bank
  int wr_addr,         // Output address offset
  int param_bank,      // Parameter bank (for gamma/beta)
  int gamma_addr,      // Gamma address offset
  int beta_addr,       // Beta address offset
  bool is_acc,         // Use accumulator
  bool use_affine      // Use affine transformation
) {
  // Encode special field
  uint64_t special =
    (norm_dim & 0xFFF) |
    ((gamma_addr & 0xFFF) << 12) |
    ((beta_addr & 0xFFF) << 24) |
    ((param_bank & 0x3) << 36) |
    (use_affine ? (1ULL << 38) : 0);

  // Encode instruction
  uint64_t inst = encode_layernorm_inst(
    batch, op1_bank, op1_addr, wr_bank, wr_addr, is_acc, special
  );

  // Issue RoCC instruction
  ROCC_INSTRUCTION(LAYERNORM_OPCODE, inst);

  // Wait for completion
  wait_layernorm_complete();
}

// Simplified interface - no affine transformation
void layernorm_simple(
  float* input,
  float* output,
  int batch,
  int dim          // Dimension in elements (will be converted to vectors)
) {
  int norm_dim = (dim + 15) / 16;  // Convert to vector count

  layernorm_hw(
    input, output, NULL, NULL,
    batch, norm_dim,
    0, 0, 1, 0,      // Bank and address config
    0, 0, 0,         // Param config (not used)
    true, false      // ACC mode, no affine
  );
}
```

### 9.3 编译器优化

#### Operator Fusion

将LayerNorm与相邻算子融合：

```
// Before
%1 = matmul(%Q, %K)
%2 = softmax(%1)
%3 = matmul(%2, %V)
%4 = layernorm(%3)

// After (fused)
%4 = attention_layernorm(%Q, %K, %V)
```

#### Memory Layout Optimization

优化数据布局以减少bank conflict：

```mlir
// Original: [batch, seq_len, hidden_dim]
// Optimized: [batch, seq_len/16, 16, hidden_dim/16, 16]
// 将数据重排为向量友好的格式
```

#### Tiling Strategy

针对大batch size进行分块处理：

```c
// Original: batch=1024, norm_dim=32
layernorm_hw(input, output, gamma, beta, 1024, 32, ...);

// Tiled: 分4次，每次256个batch
for (int i = 0; i < 4; i++) {
  layernorm_hw(
    input + i * 256 * 512,
    output + i * 256 * 512,
    gamma, beta,
    256, 32, ...  // iter=256
  );
}
```


## 10. 使用示例 (Usage Examples)

### 10.1 基本用法

```scala
// Instantiate LayerNorm unit
val layernormUnit = Module(new LayerNormUnit)

// Connect to Ball Domain
layernormUnit.io.cmdReq <> ballDomain.io.layernormReq
ballDomain.io.layernormResp <> layernormUnit.io.cmdResp

// Connect to memory system
for (i <- 0 until sp_banks) {
  scratchpad.io.read(i) <> layernormUnit.io.sramRead(i)
  scratchpad.io.write(i) <> layernormUnit.io.sramWrite(i)
}

for (i <- 0 until acc_banks) {
  accumulator.io.read(i) <> layernormUnit.io.accRead(i)
  accumulator.io.write(i) <> layernormUnit.io.accWrite(i)
}

// Status monitoring (optional)
val layernormStatus = layernormUnit.io.status
```

### 10.2 单batch LayerNorm（无仿射变换）

```c
// Process single sequence: [1, 768]
// 768 elements = 48 vectors (768 / 16)

#define HIDDEN_DIM 768
#define VECLANE 16
#define NORM_DIM (HIDDEN_DIM / VECLANE)  // 48

float input[HIDDEN_DIM];
float output[HIDDEN_DIM];

// Initialize input
for (int i = 0; i < HIDDEN_DIM; i++) {
  input[i] = randn();
}

// Load to ACC bank 0
dma_to_acc(0, 0, input, HIDDEN_DIM);

// Issue LayerNorm instruction
layernorm_hw(
  NULL, NULL, NULL, NULL,
  1,           // iter = 1 (single batch)
  NORM_DIM,    // norm_dim = 48
  0, 0,        // op1_bank=0, op1_addr=0
  1, 0,        // wr_bank=1, wr_addr=0
  0, 0, 0,     // param config (not used)
  true,        // is_acc = true
  false        // use_affine = false
);

// Read result from ACC bank 1
dma_from_acc(1, 0, output, HIDDEN_DIM);
```

### 10.3 批量LayerNorm with Gamma/Beta

```c
// Process batch: [64, 512]
// 64 batches, each with 512 elements = 32 vectors

#define BATCH 64
#define HIDDEN_DIM 512
#define NORM_DIM 32

float input[BATCH][HIDDEN_DIM];
float output[BATCH][HIDDEN_DIM];
float gamma[HIDDEN_DIM];
float beta[HIDDEN_DIM];

// Initialize data
init_data(input, gamma, beta);

// Load data to memory
dma_to_acc(0, 0, input, BATCH * HIDDEN_DIM);
dma_to_acc(2, 0x100, gamma, HIDDEN_DIM);
dma_to_acc(2, 0x120, beta, HIDDEN_DIM);

// Issue LayerNorm with affine transformation
layernorm_hw(
  NULL, NULL, NULL, NULL,
  BATCH,       // iter = 64
  NORM_DIM,    // norm_dim = 32
  0, 0,        // op1_bank=0, op1_addr=0
  1, 0,        // wr_bank=1, wr_addr=0
  2,           // param_bank=2
  0x100,       // gamma_addr
  0x120,       // beta_addr
  true,        // is_acc = true
  true         // use_affine = true
);

// Read results
dma_from_acc(1, 0, output, BATCH * HIDDEN_DIM);
```

### 10.4 Transformer Layer中的LayerNorm

```c
// Transformer encoder layer:
// output = LayerNorm(x + Attention(x))

#define SEQ_LEN 128
#define HIDDEN_DIM 768
#define NORM_DIM 48

float x[SEQ_LEN][HIDDEN_DIM];
float attn_out[SEQ_LEN][HIDDEN_DIM];
float ln_out[SEQ_LEN][HIDDEN_DIM];
float gamma[HIDDEN_DIM], beta[HIDDEN_DIM];

// Step 1: Multi-head attention
multihead_attention_hw(x, attn_out, SEQ_LEN, HIDDEN_DIM);

// Step 2: Residual connection (x + attn_out)
vecadd_hw(x, attn_out, attn_out, SEQ_LEN * HIDDEN_DIM);

// Step 3: Layer normalization
layernorm_hw(
  attn_out, ln_out, gamma, beta,
  SEQ_LEN,      // iter = 128
  NORM_DIM,     // norm_dim = 48
  0, 0,         // Input from bank 0
  1, 0,         // Output to bank 1
  2, 0x100, 0x120,  // Gamma/Beta in bank 2
  true, true    // ACC mode, use affine
);

// ln_out now contains the output of the layer
```

### 10.5 性能监控

```scala
// Performance counter integration
val perfCounters = Module(new PerfCounters)
perfCounters.io.layernorm_idle := layernormUnit.io.status.idle
perfCounters.io.layernorm_running := layernormUnit.io.status.running

// Debug: wait for LayerNorm completion
def waitLayerNormComplete(): Unit = {
  while (!layernormUnit.io.status.idle) {
    if (layernormUnit.io.status.init) {
      printf("LayerNorm: Loading data...\n")
    } else if (layernormUnit.io.status.running) {
      printf("LayerNorm: Computing...\n")
    }
  }
  printf("LayerNorm: Completed %d batches\n", layernormUnit.io.status.iter)
}

// Pipeline coordination
when(layernormUnit.io.status.ready && matmulUnit.io.status.complete) {
  // LayerNorm ready and MATMUL finished, issue LayerNorm
  layernormUnit.io.cmdReq.valid := true.B
}

// Measure cycles
val startCycle = RegInit(0.U(64.W))
val endCycle = RegInit(0.U(64.W))

when(layernormUnit.io.cmdReq.fire) {
  startCycle := cycleCounter
}
when(layernormUnit.io.cmdResp.fire) {
  endCycle := cycleCounter
  val elapsedCycles = endCycle - startCycle
  printf("LayerNorm completed in %d cycles\n", elapsedCycles)
}
```

### 10.6 不同norm_dim场景

```c
// Scenario 1: BERT-base (hidden=768)
layernorm_hw(..., 1, 48, ...);  // 768/16 = 48 vectors

// Scenario 2: GPT-2 (hidden=1024)
layernorm_hw(..., 1, 64, ...);  // 1024/16 = 64 vectors

// Scenario 3: Small model (hidden=256)
layernorm_hw(..., 1, 16, ...);  // 256/16 = 16 vectors

// Scenario 4: Large batch (batch=256, hidden=512)
layernorm_hw(..., 256, 32, ...);  // 256 batches, 512/16 = 32 vectors each
```


## 11. 性能分析与优化建议

### 11.1 性能瓶颈分析

**内存带宽瓶颈**：
- LayerNorm需要两遍读取数据（计算均值和方差）
- 每个batch需要 `2 × norm_dim` 次内存读取
- 带宽需求：`2 × norm_dim × (128 or 512 bits) / T_batch`

**计算瓶颈**：
- RSQRT计算（8-10周期）可能成为瓶颈
- 方差计算需要大量乘法运算

**优化建议**：
1. **单遍算法**：使用Welford's online algorithm合并均值和方差计算
2. **数据重用**：增大data buffer，减少内存访问
3. **并行度提升**：支持多个batch并行处理
4. **预取机制**：在处理当前batch时预取下一个batch数据

### 11.2 Welford在线算法

可以在单遍扫描中同时计算均值和方差：

```scala
// Welford's online algorithm
var n = 0
var mean = 0.0
var M2 = 0.0  // Sum of squared differences

for (x <- data) {
  n += 1
  val delta = x - mean
  mean += delta / n
  val delta2 = x - mean
  M2 += delta * delta2
}

val variance = M2 / n
```

**优势**：
- 只需一遍读取数据
- 减少50%内存访问
- 提升整体吞吐率

**实现复杂度**：
- 需要更复杂的流水线控制
- 增加除法器资源消耗


## 12. 附录

### 12.1 epsilon取值

不同框架的默认epsilon值：

| 框架 | Epsilon | 说明 |
|------|---------|------|
| PyTorch | 1e-5 | `torch.nn.LayerNorm` |
| TensorFlow | 1e-3 | `tf.keras.layers.LayerNormalization` |
| Hugging Face | 1e-12 | BERT模型 |
| NVIDIA Apex | 1e-5 | 混合精度训练 |

**建议**：使用1e-5作为默认值，允许通过配置参数调整。

### 12.2 数值稳定性

防止数值溢出/下溢的策略：

1. **方差计算**：使用 `E[(x-μ)²]` 而非 `E[x²] - μ²`
2. **RSQRT输入范围检查**：确保 `σ² + ε > 0`
3. **饱和运算**：INT8输出时进行饱和限制在[-128, 127]
4. **动态范围调整**：对极大/极小输入值进行预处理

### 12.3 参考资料

1. Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv preprint arXiv:1607.06450 (2016).
2. Welford, B. P. "Note on a method for calculating corrected sums of squares and products." Technometrics 4.3 (1962): 419-420.
3. BuckyBall Framework Documentation
4. RISC-V RoCC Interface Specification
5. Ball Domain Architecture Guide
