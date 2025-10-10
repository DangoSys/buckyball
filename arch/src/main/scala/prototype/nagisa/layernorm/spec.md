# LayerNorm加速单元设计规范

## 1. 概述 (Overview)

LayerNorm (Layer Normalization) 加速单元是BuckyBall框架中的专用计算加速器，用于高效执行层归一化运算。LayerNorm是现代深度学习模型（如Transformer、BERT、GPT等）中广泛使用的归一化技术，对于稳定训练过程和提升模型性能至关重要。

### 1.1 基本参数

- **数据格式**: 输入为FP32，输出为FP32
- **向量化处理**: 每次处理16个FP32元素（veclane=16）
- **流水线架构**: 多级流水线设计（ID, Load, Compute, Store）
- **计算方法**: 采用两步法（均值方差计算 + 归一化变换）
- **存储接口**: 支持Scratchpad和Accumulator读写
- **参数存储**: 内置gamma/beta参数缓存

### 1.2 数学定义

LayerNorm的精确定义：

对于输入向量 x ∈ ℝᴴ：

**步骤1: 计算均值和方差**
```
μ = (1/H) Σᵢ xᵢ
σ² = (1/H) Σᵢ (xᵢ - μ)²
```

**步骤2: 归一化变换**
```
x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
yᵢ = γᵢ · x̂ᵢ + βᵢ
```

其中：
- ε ≈ 1e-5 为数值稳定性常数
- γ ∈ ℝᴴ, β ∈ ℝᴴ 为可学习参数
- H 为特征维度（隐藏层大小）

## 2. 系统架构 (Block Diagram)

### 2.1 顶层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     LayerNorm Accelerator                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │              │    │              │    │              │       │
│  │   Control    │───▶│  Load Unit   │───▶│  Compute    │       │
│  │   Unit (ID)  │    │              │    │  Unit (CMP)  │       │
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
│  ┌──────────────────────────────────────┐       │               │
│  │     Parameter Cache (γ/β)            │       │               │
│  │    (SRAM-based LUT storage)          │       │               │
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
│  ID    │───▶│  Load  │───▶│  CMP   │───▶│ Store  │
│ Stage  │    │ Stage  │    │ Stage  │    │ Stage  │
└────────┘    └────────┘    └────────┘    └────────┘
    │             │             │             │
    │             │             │             │
  Decode      Load Data    Compute Stats    Write Back
  Command     from SRAM    & Normalize      to SRAM/ACC
  & Params    & γ/β        (μ, σ², x̂, y)    & Update Stats
```

### 2.3 计算单元架构

```
┌──────────────────────────────────────────────────────────────┐
│              LayerNorm Compute Pipeline (CMP Stage)          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Vector: [x₀, x₁, ..., x₁₅]                           │
│     │      │      │        │                                │
│     ├──────┼──────┼────────┼──┐                            │
│     │      │      │        │  │ Parallel Processing (16x) │
│     ▼      ▼      ▼        ▼  │                            │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐│                            │
│  │x₀² │  │x₁² │  │... │  │x₁₅²││  Step 1: Square           │
│  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘│                            │
│    │       │       │       │  │                            │
│    ▼       ▼       ▼       ▼  │                            │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐│                            │
│  │SUM │◄─┼──┼──┼──┼──┼──┼──┼──┼┘  Step 2: Sum & Mean        │
│  └─┬──┘  └────┘  └────┘  └────┘│                            │
│    │                            │                            │
│    ▼                            │                            │
│  ┌────┐                         │  μ = Σxᵢ/H                 │
│  │DIV │◄────────────────────────┘  Step 3: Divide by H       │
│  └─┬──┘                                                      │
│    │  Mean (μ)                                               │
│    ▼                                                          │
│  ┌────┐                                                      │
│  │SUB │◄──────────────────────┐  Step 4: Subtract mean      │
│  └─┬──┘                      │  (xᵢ - μ)                    │
│    │                         │                              │
│    ▼                         ▼                              │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐                            │
│  │x̂₀² │  │x̂₁² │  │... │  │x̂₁₅²│                            │
│  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘                            │
│    │       │       │       │                              │
│    ▼       ▼       ▼       ▼                              │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐                            │
│  │SUM │◄─┼──┼──┼──┼──┼──┼──┼──┼┘  Step 5: Sum squares       │
│  └─┬──┘  └────┘  └────┘  └────┘│                            │
│    │                            │  σ² = Σ(xᵢ-μ)²/H           │
│    ▼                            │                            │
│  ┌────┐                         │  Step 6: Variance          │
│  │DIV │◄────────────────────────┘                            │
│  └─┬──┘                                                      │
│    │  Variance (σ²)                                          │
│    ▼                                                          │
│  ┌────┐                                                      │
│  │ADD │ (σ² + ε)              Step 7: Add epsilon           │
│  └─┬──┘                                                      │
│    │                                                         │
│    ▼                                                         │
│  ┌────┐                                                      │
│  │SQRT│                      Step 8: Square root            │
│  └─┬──┘                                                      │
│    │  Std Dev (√(σ²+ε))                                      │
│    ▼                                                          │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐                            │
│  │DIV │  │DIV │  │... │  │DIV │  Step 9: Normalize         │
│  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘  x̂ᵢ = (xᵢ-μ)/σ             │
│    │       │       │       │                              │
│    ▼       ▼       ▼       ▼                              │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐                            │
│  │MUL │  │MUL │  │... │  │MUL │  Step 10: Scale γ          │
│  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘                            │
│    │       │       │       │                              │
│    ▼       ▼       ▼       ▼                              │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐                            │
│  │ADD │  │ADD │  │... │  │ADD │  Step 11: Shift β          │
│  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘                            │
│    │       │       │       │                              │
│    ▼       ▼       ▼       ▼                              │
│ Output: [y₀, y₁, ..., y₁₅]                                 │
│ yᵢ = γᵢ·x̂ᵢ + βᵢ                                            │
└──────────────────────────────────────────────────────────────┘
```

## 3. 接口描述 (Interface Description)

LayerNorm单元对外提供以下接口：
- **命令接口** (Command Interface): 接收LayerNorm指令并返回完成响应
- **Scratchpad接口** (SRAM Interface): 访问FP32输入数据的存储器
- **Accumulator接口** (ACC Interface): 访问FP32输出数据的存储器
- **参数接口** (Parameter Interface): 访问gamma/beta参数存储器
- **状态监控接口** (Status Interface): 输出当前运行状态信息
- **时钟和复位接口**: 提供时钟和复位信号

### 3.1 指令语义 (Instruction Semantics)

一条LayerNorm指令的完整语义如下：

**指令含义**：对存储在Scratchpad或Accumulator中的向量序列执行LayerNorm运算

**数据格式**：
- **输入**: FP32向量序列
- **输出**: FP32归一化向量序列
- **参数**: FP32 gamma/beta向量（每个向量16个元素）

**处理单位**：
- 每个向量 = 16个FP32元素（veclane = 16）
- 每个SRAM地址存储1个FP32向量（16×32位 = 512位 = 64字节）
- 每个ACC地址存储1个FP32向量（16×32位 = 512位 = 64字节）

**指令参数说明**：

| 参数 | 含义 | 示例 |
|-----|------|------|
| `iter` | 要处理的向量个数 | iter=64 表示处理64个向量 |
| `op1_bank` | 输入数据所在的Bank号 | 0-3 |
| `op1_bank_addr` | 输入起始地址 | 0x100 |
| `wr_bank` | 输出数据写入的Bank号 | 0-3 |
| `wr_bank_addr` | 输出起始地址 | 0x200 |
| `param_bank` | 参数(gamma/beta)所在的Bank号 | 0-1 |
| `param_bank_addr` | 参数起始地址 | 0x000 |
| `is_acc` | 数据类型选择 | 0=SRAM模式, 1=ACC模式 |

**输入范围**：
```
起始地址：SRAM[op1_bank][op1_bank_addr]
结束地址：SRAM[op1_bank][op1_bank_addr + iter - 1]
每地址：16个FP32元素（512位）
总元素数：iter × 16 个FP32元素
```

**输出范围**：
```
起始地址：SRAM[wr_bank][wr_bank_addr]
结束地址：SRAM[wr_bank][wr_bank_addr + iter - 1]
每地址：16个FP32元素（512位）
总元素数：iter × 16 个FP32元素
```

**参数范围**：
```
gamma向量：SRAM[param_bank][param_bank_addr]
beta向量： SRAM[param_bank][param_bank_addr + 1]
每参数向量：16个FP32元素（512位）
```

**示例1**：基本LayerNorm运算
```
iter = 64
op1_bank = 0, op1_bank_addr = 0x000
wr_bank = 1, wr_bank_addr = 0x000
param_bank = 2, param_bank_addr = 0x000
is_acc = 0

输入：SRAM[0][0x000~0x03F] 的64个向量（1024个FP32元素）
参数：SRAM[2][0x000] gamma, SRAM[2][0x001] beta
输出：SRAM[1][0x000~0x03F] 的64个向量（1024个FP32元素）
```

### 3.2 命令接口 (Command Interface)

LayerNorm单元通过Ball Domain标准接口与系统交互：

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
| `cmdReq.bits.param_bank` | Input | 1 | 参数Bank选择 |
| `cmdReq.bits.param_bank_addr` | Input | 12 | 参数Bank内地址 |
| `cmdReq.bits.is_acc` | Input | 1 | 目标存储类型 (0=SRAM, 1=ACC) |

| 信号名称 | 方向 | 位宽 | 描述 |
|---------|------|------|------|
| `cmdResp.valid` | Output | 1 | 完成响应有效信号 |
| `cmdResp.ready` | Input | 1 | 完成响应就绪信号 |
| `cmdResp.bits.rob_id` | Output | 10 | 完成指令的ROB ID |
| `cmdResp.bits.commit` | Output | 1 | 提交标志 |

### 3.3 Special字段定义

LayerNorm指令使用special字段传递额外参数：

| 位段 | 含义 | 说明 |
|-----|------|------|
| `special[0]` | 保留模式 | 0=标准LayerNorm, 1=RMSNorm |
| `special[7:1]` | epsilon值 | 指数部分编码，默认-5 (1e-5) |
| `special[15:8]` | 保留 | 未来扩展 |
| `special[39:16]` | 用户自定义 | 保留给用户使用 |

### 3.4 存储接口

**Scratchpad存储接口** (SRAM Interface):
- `sramRead`: `Vec[SramReadIO]` - 读取输入向量和参数
- `sramWrite`: `Vec[SramWriteIO]` - 写入输出向量
- 数据宽度：512位（16×FP32）

**Accumulator存储接口** (ACC Interface):
- `accRead`: `Vec[AccReadIO]` - 读取输入向量（ACC模式）
- `accWrite`: `Vec[AccWriteIO]` - 写入输出向量（ACC模式）
- 数据宽度：512位（16×FP32）

### 3.5 状态监控接口 (Status Interface)

| 信号名称 | 方向 | 位宽 | 描述 |
|---------|------|------|------|
| `status.ready` | Output | 1 | 设备准备好接受新输入 |
| `status.valid` | Output | 1 | 设备有有效输出 |
| `status.idle` | Output | 1 | 空闲状态 |
| `status.init` | Output | 1 | 初始化状态 |
| `status.load` | Output | 1 | 加载数据状态 |
| `status.compute` | Output | 1 | 计算状态 |
| `status.store` | Output | 1 | 存储状态 |
| `status.complete` | Output | 1 | 完成信号 |
| `status.iter` | Output | 32 | 已完成的批次迭代计数 |
| `status.vector_count` | Output | 16 | 当前处理的向量计数 |

## 4. 功能描述 (Functional Description)

### 4.1 操作流程

#### 4.1.1 指令接收 (Idle → Load)

1. **空闲等待**: 状态机处于`idle`状态，监听`cmdReq.valid`信号
2. **指令解码**: 捕获所有指令参数
3. **参数预取**: 从参数Bank加载gamma/beta向量到内部缓存
4. **状态转移**: 转移到`load`状态

#### 4.1.2 数据加载 (Load)

1. **发起读请求**: 向输入Bank发起读请求
2. **数据接收**: 等待响应，将数据存入向量寄存器阵列
3. **迭代控制**: 每次加载完成后更新地址
4. **状态转移**: 数据加载完成后转移到`compute`状态

#### 4.1.3 LayerNorm计算 (Compute)

执行流水线化的LayerNorm计算，分为两个阶段：

**阶段1: 统计量计算**
```scala
// 并行计算所有元素
val sum = VecInit(io.input.map(_.asTypeOf(FP32))).reduce(_ + _)
val mean = sum / H.U.asTypeOf(FP32)

// 计算方差
val diff_sq = VecInit(io.input.map(x => {
  val diff = x - mean
  diff * diff
}))
val var_sum = diff_sq.reduce(_ + _)
val variance = var_sum / H.U.asTypeOf(FP32)
val std_dev = sqrt(variance + epsilon)
```

**阶段2: 归一化变换**
```scala
// 并行归一化所有元素
val normalized = VecInit(io.input.zip(gamma_cached).zip(beta_cached).map {
  case ((x, gamma), beta) =>
    val x_hat = (x - mean) / std_dev
    gamma * x_hat + beta
})
```

#### 4.1.4 结果写回 (Store)

1. **目标选择**: 根据`is_acc`决定写入SRAM或ACC
2. **写请求**: 发起写请求到目标Bank
3. **完成确认**: 等待写响应确认
4. **状态转移**: 所有结果写回后转移到`complete`状态

#### 4.1.5 完成响应 (Complete)

1. **发送完成信号**: 通过`cmdResp`接口发送完成响应
2. **状态复位**: 返回`idle`状态，准备下一条指令

### 4.2 向量化处理

LayerNorm单元支持16通道并行计算：

```
Input Vector:  [x₀, x₁, x₂, ..., x₁₅]
                │   │   │       │
           ┌────┼───┼───┼───────┼────┐
           │ LayerNorm Compute Array (16x) │
           └────┼───┼───┼───────┼────┘
                │   │   │       │
Output Vector: [y₀, y₁, y₂, ..., y₁₅]
```

每个通道包含：
- 输入寄存器（FP32）
- 平方单元（x²）
- 减法单元（x - μ）
- 除法单元（/σ）
- 乘法单元（×γ）
- 加法单元（+β）

### 4.3 参数缓存机制

**Gamma/Beta缓存**：
- 2×16个FP32寄存器阵列
- 指令开始时从参数Bank加载
- 整个批次计算期间保持不变
- 支持双端口读写

**缓存更新策略**：
- 冷启动：从外部SRAM加载
- 热命中：使用缓存数据
- 一致性：由软件保证参数有效性

### 4.4 数值稳定性

**Epsilon处理**：
- 默认值：1e-5（FP32表示）
- 可配置：通过special字段设置
- 保护：防止除零错误

**溢出保护**：
- 中间结果使用FP32高精度
- 方差计算采用稳定算法
- 平方根使用查表+线性插值

## 5. 配置参数 (Configuration Parameters)

### 5.1 编译时参数

通过`CustomBuckyBallConfig`配置：

| 参数名称 | 类型 | 默认值 | 描述 |
|---------|------|--------|------|
| `veclane` | Int | 16 | 向量通道数 |
| `sp_banks` | Int | 4 | Scratchpad Bank数量 |
| `acc_banks` | Int | 2 | Accumulator Bank数量 |
| `param_banks` | Int | 2 | 参数Bank数量 |
| `spad_bank_entries` | Int | 1024 | 每个SRAM Bank条目数 |
| `acc_bank_entries` | Int | 512 | 每个ACC Bank条目数 |
| `param_entries` | Int | 256 | 参数缓存条目数 |
| `epsilon` | FP32 | 1e-5 | 数值稳定性常数 |
| `pipeline_depth` | Int | 8 | 计算流水线深度 |

### 5.2 运行时参数

通过Ball Domain指令传递：

| 参数 | 位宽 | 范围 | 描述 |
|-----|------|------|------|
| `iter` | 10 | 1-1024 | 向量迭代次数 |
| `op1_bank` | 2 | 0-3 | 输入数据Bank |
| `op1_bank_addr` | 12 | 0-4095 | 输入起始地址 |
| `wr_bank` | 2 | 0-3 | 输出数据Bank |
| `wr_bank_addr` | 12 | 0-4095 | 输出起始地址 |
| `param_bank` | 1 | 0-1 | 参数Bank |
| `param_bank_addr` | 12 | 0-4095 | 参数起始地址 |
| `is_acc` | 1 | 0-1 | 输出目标 (0=SRAM, 1=ACC) |

### 5.3 Special字段配置

| 位段 | 配置值 | 描述 |
|-----|--------|------|
| `special[0]` | 0 | 标准LayerNorm模式 |
| `special[0]` | 1 | RMSNorm模式（无均值减除） |
| `special[7:1]` | -5 | epsilon = 1e-5 |
| `special[7:1]` | -6 | epsilon = 1e-6 |
| `special[7:1]` | -4 | epsilon = 1e-4 |

## 6. 时序特性 (Timing Characteristics)

### 6.1 延迟分析

| 操作阶段 | 周期数 | 说明 |
|---------|-------|------|
| 指令解码 (ID) | 1 | 命令参数捕获 |
| 参数加载 | 2 | gamma/beta参数加载 |
| 数据加载 (Load) | 3-4 | SRAM读延迟 |
| 统计计算 | 4-6 | 均值方差计算 |
| 归一化变换 | 3-4 | 并行归一化计算 |
| 结果写回 (Store) | 2-3 | SRAM/ACC写延迟 |
| 完成响应 (Complete) | 1 | ROB通知 |
| **总延迟** | **16-21** | 典型值18周期 |

### 6.2 关键路径

最长组合逻辑路径出现在统计计算阶段：

```
Input → Square → Tree Add → Divide → Subtract → Square → Tree Add → Divide → Sqrt → Divide
```

**优化策略**：
1. 树形加法器减少求和延迟
2. 流水线寄存器分解长路径
3. 查找表加速平方根计算
4. 并行处理16个通道

### 6.3 吞吐率

- **峰值吞吐率**: 1向量/周期（连续处理）
- **有效吞吐率**: 0.8-0.9向量/周期（考虑启动延迟）
- **内存带宽**: 每向量需要1读 + 1写 + 0.03参数读（摊销）

## 7. ISA定制 (ISA Customization)

### 7.1 指令编码

LayerNorm指令使用RoCC格式，func7 = 0x23 (35)：

```
func7 = 0x23  (35)  // LayerNorm opcode
rs1 = {op1_bank[1:0], op1_bank_addr[11:0]}  // 14位
rs2 = {special[39:0], iter[9:0], wr_bank_addr[11:0], wr_bank[1:0], param_bank_addr[11:0], param_bank[0]}
```

### 7.2 Special字段定义

| 位段 | 名称 | 类型 | 描述 |
|-----|------|------|------|
| [0] | norm_mode | bool | 0=LayerNorm, 1=RMSNorm |
| [7:1] | epsilon_exp | int7 | epsilon指数部分（2的幂） |
| [15:8] | reserved | uint8 | 保留供未来使用 |
| [39:16] | user_defined | uint24 | 用户自定义字段 |

### 7.3 软件接口示例

```c
// C intrinsic for LayerNorm
void layernorm_hw(
  float* input,      // Input vector address
  float* output,     // Output vector address
  float* gamma,      // Gamma parameters
  float* beta,       // Beta parameters
  int iter,          // Number of vectors
  int op1_bank,      // Input bank
  int op1_addr,      // Input address
  int wr_bank,       // Output bank
  int wr_addr,       // Output address
  int param_bank,    // Parameter bank
  int param_addr,    // Parameter address
  bool is_acc,       // Use accumulator
  int norm_mode,     // 0=LayerNorm, 1=RMSNorm
  int epsilon_exp    // Epsilon exponent
) {
  uint64_t rs1 = ((op1_bank & 0x3) << 12) | (op1_addr & 0xFFF);
  uint64_t rs2 = ((norm_mode & 0x1) << 40) |
                 ((epsilon_exp & 0x7F) << 33) |
                 ((iter & 0x3FF) << 23) |
                 ((wr_addr & 0xFFF) << 11) |
                 ((wr_bank & 0x3) << 9) |
                 ((param_addr & 0xFFF) << 1) |
                 (param_bank & 0x1);

  ROCC_INSTRUCTION(0x23, rs1, rs2);
  wait_layernorm_complete();
}
```

## 8. 验证方案 (Verification Plan)

### 8.1 功能验证

#### 8.1.1 单元测试

- **基本功能**: 单个向量LayerNorm计算正确性
- **边界条件**: 零值、极大值、极小值、NaN、Inf
- **参数测试**: gamma=1, beta=0的恒等变换
- **精度测试**: epsilon对数值稳定性的影响

#### 8.1.2 精度验证

与软件参考模型（PyTorch）对比：

```python
import torch
import torch.nn as nn

# Reference LayerNorm
def layernorm_ref(x, gamma, beta, eps=1e-5):
    return nn.functional.layer_norm(x, (16,), gamma, beta, eps)

# Test vectors
test_inputs = torch.randn(64, 16)  # 64 vectors, 16 elements each
gamma = torch.ones(16)
beta = torch.zeros(16)
golden_outputs = layernorm_ref(test_inputs, gamma, beta)

# Compare with hardware outputs
max_error = torch.max(torch.abs(hw_outputs - golden_outputs))
mean_error = torch.mean(torch.abs(hw_outputs - golden_outputs))

assert max_error < 1e-4, f"Max error {max_error} exceeds threshold"
assert mean_error < 1e-5, f"Mean error {mean_error} exceeds threshold"
```

### 8.2 性能验证

- **吞吐率测试**: 连续向量处理性能
- **延迟测试**: 单个向量的处理延迟
- **流水线效率**: 不同批次大小的效率分析
- **内存带宽**: 内存访问模式分析

### 8.3 集成验证

- **完整系统**: 在ToyBuckyBall环境中集成测试
- **编译器支持**: 验证MLIR lowering生成正确的LayerNorm指令
- **端到端**: 运行完整的Transformer模型，验证功能和性能

## 9. 软件接口 (Software Interface)

### 9.1 MLIR Dialect扩展

在Buckyball Dialect中添加LayerNorm操作：

```mlir
// MLIR IR
%output = buckyball.layernorm %input, %gamma, %beta
          {epsilon = 1.0e-5, norm_mode = "layernorm"}
          : tensor<64x16xf32>, tensor<16xf32>, tensor<16xf32>

// Lowering to hardware intrinsic
func.func @layernorm_layer(%input: memref<64x16xf32>,
                          %output: memref<64x16xf32>,
                          %gamma: memref<16xf32>,
                          %beta: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index

  // Issue LayerNorm instruction to hardware
  buckyball.layernorm.hw %input, %output, %gamma, %beta, %c0, %c64
    {op1_bank = 0, wr_bank = 1, param_bank = 2,
     op1_addr = 0, wr_addr = 0, param_addr = 0,
     is_acc = 0, norm_mode = 0, epsilon = -5}

  return
}
```

### 9.2 高级API

```python
# Python API
import buckyball as bb

def layernorm_layer(x, gamma=None, beta=None, epsilon=1e-5):
    """LayerNorm layer using BuckyBall hardware accelerator"""
    B, L, H = x.shape  # batch, length, hidden

    # Flatten to 2D for hardware processing
    x_2d = x.reshape(-1, H)  # [B*L, H]
    vectors = x_2d.shape[0]

    # Default parameters
    if gamma is None:
        gamma = torch.ones(H)
    if beta is None:
        beta = torch.zeros(H)

    # Allocate memory in scratchpad
    input_addr = bb.sram_alloc(vectors * H * 4)  # FP32
    output_addr = bb.sram_alloc(vectors * H * 4)
    gamma_addr = bb.sram_alloc(H * 4)
    beta_addr = bb.sram_alloc(H * 4)

    # Transfer data
    bb.sram_write(input_addr, x_2d.numpy())
    bb.sram_write(gamma_addr, gamma.numpy())
    bb.sram_write(beta_addr, beta.numpy())

    # Issue LayerNorm instruction
    bb.layernorm(
        input_addr=input_addr,
        output_addr=output_addr,
        gamma_addr=gamma_addr,
        beta_addr=beta_addr,
        vectors=vectors,
        epsilon=epsilon
    )

    # Read result
    result = bb.sram_read(output_addr, shape=(vectors, H))

    # Reshape back
    return torch.from_numpy(result).reshape(B, L, H)
```

## 10. 使用示例 (Usage Examples)

### 10.1 基本LayerNorm

```c
// Basic LayerNorm on 1024 elements (64 vectors of 16 elements)
#define N 1024
#define H 16
#define VECTORS (N / H)

float input[N], output[N];
float gamma[H] = {1.0, 1.0, ..., 1.0};  // Identity scale
float beta[H] = {0.0, 0.0, ..., 0.0};   // Zero shift

// Initialize input data
for (int i = 0; i < N; i++) {
    input[i] = (float)(i % H) * 0.1f;  // Simple pattern
}

// Allocate and transfer data
dma_to_sram(0, 0x000, input, N);
dma_to_sram(2, 0x000, gamma, H);
dma_to_sram(2, 0x040, beta, H);

// Issue LayerNorm instruction
layernorm_hw(
    NULL, NULL, NULL, NULL,  // Hardware manages addresses
    VECTORS,                 // 64 vectors
    0, 0x000,               // Input: bank 0, addr 0x000
    1, 0x000,               // Output: bank 1, addr 0x000
    2, 0x000,               // Params: bank 2, addr 0x000
    false,                  // SRAM mode
    0,                      // Standard LayerNorm
    -5                      // epsilon = 1e-5
);

// Read results
dma_from_sram(1, 0x000, output, N);

// Verify: output should have zero mean and unit variance per vector
for (int v = 0; v < VECTORS; v++) {
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = 0; i < H; i++) {
        sum += output[v*H + i];
        sum_sq += output[v*H + i] * output[v*H + i];
    }
    float mean = sum / H;
    float variance = sum_sq / H - mean * mean;
    printf("Vector %d: mean=%.6f, variance=%.6f\n", v, mean, variance);
}
```

### 10.2 Transformer中的LayerNorm

```c
// Multi-head attention LayerNorm
// Y = LayerNorm(X + Attention(X))

#define SEQ_LEN 512
#define HIDDEN 1024
#define HEADS 16
#define HEAD_DIM (HIDDEN / HEADS)

float attention_output[SEQ_LEN * HIDDEN];
float residual[SEQ_LEN * HIDDEN];
float layernorm_output[SEQ_LEN * HIDDEN];

// Step 1: Compute attention (not shown)
// multi_head_attention(X, attention_output);

// Step 2: Add residual connection
for (int i = 0; i < SEQ_LEN * HIDDEN; i++) {
    residual[i] = X[i] + attention_output[i];
}

// Step 3: Apply LayerNorm in tiles
// Process 16 elements at a time (vector width)
#define TILE_SIZE 16
int tiles_per_row = HIDDEN / TILE_SIZE;
int total_tiles = SEQ_LEN * tiles_per_row;

// Transfer residual to SRAM
dma_to_sram(0, 0x000, residual, SEQ_LEN * HIDDEN);

// Load LayerNorm parameters (learned during training)
dma_to_sram(2, 0x000, layernorm_gamma, HIDDEN);
dma_to_sram(2, 0x400, layernorm_beta, HIDDEN);

// Issue LayerNorm instruction
layernorm_hw(
    NULL, NULL, NULL, NULL,
    total_tiles,            // Total number of 16-element vectors
    0, 0x000,               // Input: residual data
    1, 0x000,               // Output: normalized data
    2, 0x000,               // Parameters: gamma/beta
    false,                  // SRAM mode
    0,                      // Standard LayerNorm
    -5                      // epsilon = 1e-5
);

// Read LayerNorm output
dma_from_sram(1, 0x000, layernorm_output, SEQ_LEN * HIDDEN);

// Step 4: Continue with feed-forward network
// feed_forward(layernorm_output, ff_output);
```

### 10.3 RMSNorm变体

```c
// RMSNorm: y = x * γ / RMS(x), where RMS(x) = sqrt(mean(x²))
// More efficient than LayerNorm, used in modern LLMs

#define N 1024
#define H 16
#define VECTORS (N / H)

float input[N], output[N];
float gamma[H] = {1.0, 1.0, ..., 1.0};

// No beta needed for RMSNorm

// Initialize input with random data
init_random_data(input, N);

// Transfer data
dma_to_sram(0, 0x000, input, N);
dma_to_sram(2, 0x000, gamma, H);

// Issue RMSNorm instruction (norm_mode = 1)
layernorm_hw(
    NULL, NULL, NULL, NULL,
    VECTORS,
    0, 0x000,               // Input
    1, 0x000,               // Output
    2, 0x000,               // Gamma only
    false,                  // SRAM mode
    1,                      // RMSNorm mode
    -5                      // epsilon = 1e-5
);

// Read results
dma_from_sram(1, 0x000, output, N);

// Verify RMS normalization
for (int v = 0; v < VECTORS; v++) {
    float sum_sq = 0.0f;
    for (int i = 0; i < H; i++) {
        sum_sq += input[v*H + i] * input[v*H + i];
    }
    float rms = sqrt(sum_sq / H);
    printf("Vector %d: RMS=%.6f\n", v, rms);
}
```

### 10.4 性能监控

```scala
// Performance monitoring integration
val perfCounters = Module(new PerfCounters)
perfCounters.io.layernorm_cycles := RegNext(cycleCount)
perfCounters.io.layernorm_vectors := RegNext(vectorCount)

// Runtime profiling
def profileLayerNorm(): Unit = {
  val startTime = cycleCount
  val startVectors = layernormUnit.io.status.vector_count

  // Issue LayerNorm instruction
  issueLayerNorm()

  // Wait for completion
  while (!layernormUnit.io.status.complete) {
    // Monitor intermediate states
    when(layernormUnit.io.status.load) {
      printf("LN: Loading data...\n")
    }
    when(layernormUnit.io.status.compute) {
      printf("LN: Computing stats...\n")
    }
    when(layernormUnit.io.status.store) {
      printf("LN: Storing results...\n")
    }
  }

  val endTime = cycleCount
  val endVectors = layernormUnit.io.status.vector_count
  val elapsed = endTime - startTime
  val vectorsProcessed = endVectors - startVectors

  printf("LayerNorm Profile: %d cycles, %d vectors, %.2f cycles/vector\n",
         elapsed, vectorsProcessed,
         elapsed.toFloat / vectorsProcessed.toFloat)
}
```
