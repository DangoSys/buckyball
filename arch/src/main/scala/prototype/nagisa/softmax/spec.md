# Softmax加速单元设计规范

## 1. 概述 (Overview)

Softmax加速单元是BuckyBall框架中的专用计算加速器，用于高效执行Softmax激活函数运算。Softmax是深度学习模型（如Transformer、BERT、GPT等）中广泛使用的归一化激活函数，特别是在注意力机制(Attention)中起到关键作用。

### 1.1 基本参数

- **数据格式**: 输入为INT8，内部计算FP32，输出为INT32
- **向量化处理**: 每次处理16个INT8元素（veclane=16）
- **流水线架构**: 多级流水线设计（ID, FindMax, ComputeExp, Normalize, Store）
- **计算方法**: 采用数值稳定的Softmax算法（减最大值）
- **存储接口**: 支持Scratchpad和Accumulator读写

### 1.2 数学定义

Softmax函数的精确定义：

```
Softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

数值稳定的实现方式：

```
max_x = max(x_1, x_2, ..., x_n)
Softmax(x_i) = exp(x_i - max_x) / Σ exp(x_j - max_x)
```

这种方法通过减去最大值来防止指数运算溢出，确保数值稳定性。

## 2. 系统架构 (Block Diagram)

### 2.1 顶层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Softmax Accelerator                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │              │    │   FindMax    │    │  ComputeExp  │       │
│  │   Control    │───▶│     Unit     │───▶│   & Sum      │       │
│  │   Unit (ID)  │    │              │    │     Unit     │       │
│  │              │    │              │    │              │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │               │
│         │                   │                   │               │
│         │              ┌────▼────┐         ┌────▼────┐          │
│         │              │ SRAM    │         │Normalize│          │
│         │              │ Read    │         │  Unit   │          │
│         │              │ Arbiter │         │(Divide) │          │
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
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│  ID    │───▶│FindMax │───▶│ExpSum  │───▶│ Norm   │───▶│ Store  │
│ Stage  │    │ Stage  │    │ Stage  │    │ Stage  │    │ Stage  │
└────────┘    └────────┘    └────────┘    └────────┘    └────────┘
    │             │             │             │             │
    │             │             │             │             │
  Decode      Find Max     Compute exp()   Normalize    Write Back
  Command     from input   and accumulate   by dividing   to ACC/SRAM
              vectors       sum             sum total
```

### 2.3 计算单元架构

```
┌─────────────────────────────────────────────────────────┐
│          Softmax Compute Pipeline (Multi-Stage)         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input [x₀, x₁, ..., x₁₅]                               │
│     │                                                   │
│     │  Stage 1: Find Maximum                            │
│     ├──────────┬──────────┬──────────┐                  │
│     │          │          │          │                  │
│     ▼          ▼          ▼          ▼                  │
│  ┌────────────────────────────────┐                     │
│  │  Parallel Comparator Tree     │                     │
│  │    (16-way reduction)          │                     │
│  └────────────┬───────────────────┘                     │
│               │                                         │
│            max_val                                      │
│               │                                         │
│               │  Stage 2: Compute exp(x - max) & Sum    │
│     ┌─────────┴────────┐                                │
│     │                  │                                │
│     ▼                  ▼                                │
│  [x₀-max]          [x₁-max] ...                         │
│     │                  │                                │
│     ▼                  ▼                                │
│  exp(x₀-max)       exp(x₁-max) ...                      │
│     │                  │                                │
│     └────────┬─────────┘                                │
│              │                                          │
│              ▼                                          │
│        ┌─────────┐                                      │
│        │Reduction│  Σ exp(xᵢ-max)                       │
│        │  Tree   │                                      │
│        └────┬────┘                                      │
│             │                                           │
│          sum_exp                                        │
│             │                                           │
│             │  Stage 3: Normalization (Division)        │
│     ┌───────┴────────┐                                  │
│     │                │                                  │
│     ▼                ▼                                  │
│  exp(x₀-max)/sum  exp(x₁-max)/sum ...                   │
│     │                │                                  │
│     ▼                ▼                                  │
│  Output [y₀, y₁, ..., y₁₅]                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```


## 3. 接口描述 (Interface Description)

Softmax单元对外提供以下接口：
- **命令接口** (Command Interface): 接收Softmax指令并返回完成响应
- **Scratchpad接口** (SRAM Interface): 访问INT8数据的存储器
- **Accumulator接口** (ACC Interface): 访问INT32数据的存储器
- **状态监控接口** (Status Interface): 输出当前运行状态信息
- **时钟和复位接口**: 提供时钟和复位信号

### 3.1 指令语义 (Instruction Semantics)

一条Softmax指令的完整语义如下：

**指令含义**：对存储在Scratchpad或Accumulator中的向量沿指定维度执行Softmax运算

**数据格式**：
- **Scratchpad模式** (`is_acc=0`)：INT8输入 → Softmax → INT8输出
- **Accumulator模式** (`is_acc=1`)：INT32输入 → Softmax → INT32输出
- **注意**：内部计算使用FP32保证精度

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
| `special[9:0]` | Softmax维度长度 | dim_len=16表示沿16个元素做Softmax |
| `special[19:10]` | Batch大小 | batch=4表示有4个独立的Softmax组 |
| `special[20]` | 归一化模式 | 0=标准Softmax, 1=LogSoftmax |
| `special[39:21]` | 保留位 | 用于未来扩展 |

**Special字段编码**：

```
special[39:0]:
  [9:0]   - dim_len:  Softmax维度长度（1-1024）
  [19:10] - batch:    批次大小（1-1024）
  [20]    - log_mode: 0=Softmax, 1=LogSoftmax
  [39:21] - reserved: 保留
```

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

**示例1**：SRAM模式，单个Softmax（沿16个元素）
```
iter = 1
op1_bank = 0, op1_bank_addr = 0x000
wr_bank = 1, wr_bank_addr = 0x000
is_acc = 0
special[9:0] = 16 (dim_len)
special[19:10] = 1 (batch)
special[20] = 0 (标准Softmax)

输入：SRAM[0][0x000] 的16个INT8元素
输出：SRAM[1][0x000] 的16个INT8元素（Softmax归一化后）
```

**示例2**：ACC模式，批量Softmax处理
```
iter = 64
op1_bank = 0, op1_bank_addr = 0x000
wr_bank = 1, wr_bank_addr = 0x000
is_acc = 1
special[9:0] = 64 (dim_len，沿64个向量做Softmax)
special[19:10] = 16 (batch，16组独立的Softmax)
special[20] = 0

输入：ACC[0][0x000~0x03F] 的64个向量（1024个INT32元素）
处理：每64个向量作为一组，对其执行Softmax（沿dim=64的维度）
输出：ACC[1][0x000~0x03F] 的64个向量（1024个INT32元素）
```

**示例3**：LogSoftmax模式
```
iter = 16
op1_bank = 0, op1_bank_addr = 0x100
wr_bank = 0, wr_bank_addr = 0x200
is_acc = 0
special[9:0] = 16 (dim_len)
special[19:10] = 16 (batch)
special[20] = 1 (LogSoftmax模式)

输入：SRAM[0][0x100~0x10F] 的16个向量（256个INT8元素）
输出：SRAM[0][0x200~0x20F] 的16个向量（256个INT8元素）
计算：log(Softmax(x)) = (x - max) - log(Σexp(x - max))
```

### 3.2 命令接口 (Command Interface)

Softmax单元通过Ball Domain标准接口与系统交互：

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
| `cmdReq.bits.special` | Input | 40 | 定制参数（维度、batch等） |

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

Softmax单元提供状态监控接口，用于外部观察当前运行状态：

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

Softmax单元不直接暴露APB寄存器接口，而是通过Ball Domain命令接口进行控制。内部状态寄存器如下：

| 寄存器名称 | 位宽 | 复位值 | 描述 |
|-----------|------|--------|------|
| `state` | 3 | `idle` | 状态机状态: idle/findmax/expsum/normalize/store/complete |
| `rob_id_reg` | 10 | 0 | 当前处理指令的ROB ID |
| `iter_reg` | 10 | 0 | 迭代次数寄存器 |
| `iter_cnt` | 10 | 0 | 迭代计数器 |
| `op1_bank_reg` | 2 | 0 | 操作数Bank寄存器 |
| `op1_addr_reg` | 12 | 0 | 操作数地址寄存器 |
| `wr_bank_reg` | 2 | 0 | 写回Bank寄存器 |
| `wr_addr_reg` | 12 | 0 | 写回地址寄存器 |
| `is_acc_reg` | 1 | 0 | 写回目标类型寄存器 |
| `dim_len_reg` | 10 | 0 | Softmax维度长度 |
| `batch_reg` | 10 | 0 | 批次大小 |
| `log_mode_reg` | 1 | 0 | LogSoftmax模式标志 |
| `max_val_reg` | 32 | 0 | 当前最大值寄存器（FP32） |
| `sum_exp_reg` | 32 | 0 | 指数和寄存器（FP32） |
| `vec_buffer` | 128×32 | 0 | 向量缓冲区（存储中间结果） |
| `load_cnt` | 4 | 0 | 加载计数器 |
| `exec_cnt` | 5 | 0 | 执行计数器 |
| `iter_cnt` | 32 | 0 | 批次迭代计数器 (用于status.iter) |
| `has_input` | 1 | 0 | 输入状态标志 |
| `has_output` | 1 | 0 | 输出状态标志 |

### 4.2 状态机编码

| 状态名称 | 编码 | 描述 |
|---------|------|------|
| `idle` | 3'b000 | 空闲状态，等待命令 |
| `findmax` | 3'b001 | 查找最大值状态 |
| `expsum` | 3'b010 | 计算指数和累加状态 |
| `normalize` | 3'b011 | 归一化除法状态 |
| `store` | 3'b100 | 写回结果状态 |
| `complete` | 3'b101 | 完成响应状态 |


## 5. 功能描述 (Functional Description)

### 5.1 操作流程

#### 5.1.1 指令接收 (Idle → FindMax)

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

   // 解析special字段
   dim_len_reg := cmdReq.bits.special(9, 0)
   batch_reg := cmdReq.bits.special(19, 10)
   log_mode_reg := cmdReq.bits.special(20)
   ```
3. **状态转移**: 转移到`findmax`状态

#### 5.1.2 查找最大值 (FindMax)

1. **遍历读取**: 逐个向量从SRAM/ACC读取数据
   ```scala
   sramRead(op1_bank_reg).req.valid := true.B
   sramRead(op1_bank_reg).req.bits.addr := op1_addr_reg + read_cnt
   ```
2. **并行比较**: 使用比较器树找到每个向量的最大值
   ```scala
   val max_tree = Module(new MaxReductionTree(16))
   max_tree.io.in := input_vector
   val vec_max = max_tree.io.out
   ```
3. **全局最大**: 在dim_len范围内更新全局最大值
   ```scala
   when(vec_max > max_val_reg) {
     max_val_reg := vec_max
   }
   ```
4. **缓冲存储**: 将输入向量缓存到`vec_buffer`以供后续使用
5. **状态转移**: 所有向量扫描完成后，转移到`expsum`状态

#### 5.1.3 计算指数和累加 (ExpSum)

1. **减去最大值**: 对缓冲区中的每个元素减去全局最大值
   ```scala
   val x_shifted = vec_buffer(i) - max_val_reg
   ```
2. **计算指数**: 使用exp近似单元计算exp(x - max)
   ```scala
   val exp_val = expApprox(x_shifted)
   vec_buffer(i) := exp_val  // 更新缓冲区
   ```
3. **累加求和**: 累加所有exp值
   ```scala
   sum_exp_reg := sum_exp_reg + exp_val
   ```
4. **状态转移**: 完成后转移到`normalize`状态

#### 5.1.4 归一化 (Normalize)

1. **标准Softmax模式** (log_mode=0):
   ```scala
   val softmax_out = vec_buffer(i) / sum_exp_reg
   ```

2. **LogSoftmax模式** (log_mode=1):
   ```scala
   val log_sum = log(sum_exp_reg)
   val logsoftmax_out = (input_vector(i) - max_val_reg) - log_sum
   ```

3. **数据转换**: 将FP32结果转换为INT8/INT32输出格式
4. **状态转移**: 转移到`store`状态

#### 5.1.5 结果写回 (Store)

1. **目标选择**: 根据`is_acc_reg`决定写入SRAM或ACC
2. **写请求**:
   ```scala
   when(is_acc_reg) {
     accWrite(wr_bank_reg).valid := true.B
     accWrite(wr_bank_reg).bits.addr := wr_addr_reg + write_cnt
     accWrite(wr_bank_reg).bits.data := softmax_result
   }.otherwise {
     sramWrite(wr_bank_reg).valid := true.B
     sramWrite(wr_bank_reg).bits.addr := wr_addr_reg + write_cnt
     sramWrite(wr_bank_reg).bits.data := softmax_result
   }
   ```
3. **迭代控制**: 所有结果写回完成后，转移到`complete`状态

#### 5.1.6 完成响应 (Complete)

1. **发送完成信号**:
   ```scala
   cmdResp.valid := true.B
   cmdResp.bits.rob_id := rob_id_reg
   cmdResp.bits.commit := true.B
   ```
2. **状态复位**: 返回`idle`状态，准备接收下一条指令

### 5.2 Exp近似算法

硬件实现exp函数采用分段多项式或查找表方法：

#### 方案1: 查找表 (LUT)

- **输入范围**: [-16, 16]，分为512个区间
- **表项**: 每个区间存储exp值的斜率和截距
- **插值**: 线性插值计算精确值
- **误差**: < 0.001

#### 方案2: 分段多项式

将输入域划分为多个区间，每个区间用二阶多项式近似：

```
exp(x) ≈ a₂x² + a₁x + a₀  (for x ∈ [x_min, x_max])
```

- **区间数**: 16个区间
- **系数存储**: 每区间3个系数 (a₀, a₁, a₂)
- **误差**: < 0.0005

#### 方案3: 指数位操作法

利用浮点数表示的特性快速计算exp：

```scala
// exp(x) = 2^(x/ln(2))
val x_scaled = x * 1.44269504  // 1/ln(2)
val exp_approx = (127 + x_scaled.floor) << 23  // FP32指数域
```

### 5.3 数值稳定性保证

Softmax计算中的数值稳定性问题主要来自指数运算的溢出：

#### 问题示例：
```
输入: x = [1000, 1001, 1002]
直接计算: exp(1000) ≈ 2×10^434 (溢出!)
```

#### 解决方案：减最大值
```
max_x = 1002
x' = [-2, -1, 0]
exp(x') = [0.135, 0.368, 1.0]
sum = 1.503
softmax = [0.090, 0.245, 0.665]  ✓ 数值稳定
```

### 5.4 向量化处理

Softmax单元支持向量级并行计算，典型配置为16通道：

```
Input Vector:  [x₀, x₁, x₂, ..., x₁₅]
                │   │   │       │
           ┌────┼───┼───┼───────┼────┐
           │ Parallel Max Finder (16x) │
           └────┼───┼───┼───────┼────┘
                │   │   │       │
           ┌────┼───┼───┼───────┼────┐
           │    Exp Compute (16x)     │
           └────┼───┼───┼───────┼────┘
                │   │   │       │
           ┌────┼───┼───┼───────┼────┐
           │ Parallel Divider (16x)   │
           └────┼───┼───┼───────┼────┘
                │   │   │       │
Output Vector: [y₀, y₁, y₂, ..., y₁₅]
```

### 5.5 批量处理优化

对于多个独立的Softmax操作（如多头注意力），可以批量并行处理：

```
Batch 0: [x₀₀, x₀₁, ..., x₀₁₅] → Softmax → [y₀₀, y₀₁, ..., y₀₁₅]
Batch 1: [x₁₀, x₁₁, ..., x₁₅] → Softmax → [y₁₀, y₁₁, ..., y₁₅]
...
Batch N: [xₙ₀, xₙ₁, ..., xₙ₁₅] → Softmax → [yₙ₀, yₙ₁, ..., yₙ₁₅]
```

每个batch的Softmax计算相互独立，可以复用相同的硬件流水线。


## 6. 时序特性 (Timing Characteristics)

### 6.1 延迟分析

| 操作阶段 | 周期数 | 说明 |
|---------|-------|------|
| 指令解码 (ID) | 1 | 命令参数捕获 |
| 查找最大值 (FindMax) | dim_len/16 | 每周期处理1个向量 |
| 数据加载 | 3-4 | SRAM读延迟（流水线重叠）|
| 计算Exp并累加 (ExpSum) | dim_len/16 × 6 | Exp近似+累加 |
| 归一化除法 (Normalize) | dim_len/16 × 8 | 除法器延迟较高 |
| 结果写回 (Store) | dim_len/16 | 每周期写回1个向量 |
| 完成响应 (Complete) | 1 | ROB通知 |
| **总延迟 (dim=16)** | **~30** | 典型值 |
| **总延迟 (dim=256)** | **~250** | 大维度情况 |

### 6.2 吞吐率

**单次Softmax**:
- 处理16个元素: ~30周期
- 处理256个元素: ~250周期

**批量处理**:
- 多个独立Softmax可流水线处理
- 稳态吞吐率: 1 Softmax / (dim_len/16 + overhead) 周期

### 6.3 关键路径

最长组合逻辑路径出现在归一化阶段的除法器：

```
exp_val → Divider → normalized_output
```

**优化策略**:
1. 使用高性能除法器IP（如Goldschmidt迭代）
2. 将除法流水线化为多个子阶段
3. 对于特定应用，可使用倒数近似+乘法替代除法


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
| `inputType` | DataType | INT8 | 输入数据类型 |
| `outputType` | DataType | INT32 | 输出数据类型 |
| `softmax_buffer_depth` | Int | 128 | Softmax向量缓冲区深度 |
| `exp_lut_size` | Int | 512 | Exp查找表大小 |
| `div_pipeline_depth` | Int | 8 | 除法器流水线深度 |

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
| `special[9:0]` | 10 | 1-1024 | Softmax维度长度 |
| `special[19:10]` | 10 | 1-1024 | Batch大小 |
| `special[20]` | 1 | 0-1 | LogSoftmax模式 |


## 8. 验证方案 (Verification Plan)

### 8.1 功能验证

#### 8.1.1 单元测试

- **基本功能**: 单个向量Softmax计算正确性
- **边界条件**: 全零、全相同值、极大值、极小值、NaN
- **数值稳定性**: 大数值输入不溢出
- **向量处理**: 16元素并行计算一致性
- **维度变化**: 不同dim_len的正确性

#### 8.1.2 精度验证

与软件参考模型（PyTorch/NumPy）对比：

```python
import torch
import numpy as np

# Reference Softmax
def softmax_ref(x, dim=-1):
    return torch.nn.functional.softmax(x, dim=dim)

# Test vectors
test_inputs = torch.randn(16, 64)
golden_outputs = softmax_ref(test_inputs, dim=-1)

# Compare with hardware outputs
max_error = torch.max(torch.abs(hw_outputs - golden_outputs))
mean_error = torch.mean(torch.abs(hw_outputs - golden_outputs))

assert max_error < 1e-3, f"Max error {max_error} exceeds threshold"
```

#### 8.1.3 数值稳定性测试

```python
# Large value test (should not overflow)
x_large = torch.tensor([1000.0, 1001.0, 1002.0])
y_hw = softmax_hw(x_large)
y_ref = torch.softmax(x_large, dim=0)
assert torch.allclose(y_hw, y_ref, atol=1e-3)

# Small value test (should not underflow)
x_small = torch.tensor([-1000.0, -1001.0, -1002.0])
y_hw = softmax_hw(x_small)
y_ref = torch.softmax(x_small, dim=0)
assert torch.allclose(y_hw, y_ref, atol=1e-3)
```

### 8.2 性能验证

- **延迟测量**: 记录不同dim_len下的实际周期数
- **吞吐率**: 批量处理的平均吞吐率
- **资源利用**: 流水线占用率、存储器带宽利用率

### 8.3 集成验证

- **完整系统**: 在ToyBuckyBall环境中集成测试
- **编译器支持**: 验证MLIR lowering生成正确的Softmax指令
- **端到端**: 运行完整的Transformer Attention层，验证功能和性能


## 9. 软件接口 (Software Interface)

### 9.1 MLIR Dialect扩展

在Buckyball Dialect中添加Softmax操作：

```mlir
// MLIR IR
%output = buckyball.softmax %input {dim = -1} : tensor<16x64xf32>

// Lowering to hardware intrinsic
func.func @softmax_layer(%arg0: memref<16x64xf32>, %arg1: memref<16x64xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index

  // Issue Softmax instruction to hardware
  buckyball.softmax.hw %arg0, %arg1, %c0, %c16
    {op1_bank = 0, wr_bank = 1, is_acc = 0,
     dim_len = 64, batch = 16, log_mode = 0}

  return
}
```

### 9.2 C/C++ Intrinsics

提供底层硬件访问接口：

```c
// C intrinsic
void softmax_hw(
  void* input,         // Input vector address
  void* output,        // Output vector address
  int iter,            // Number of vectors
  int op1_bank,        // Input bank
  int op1_addr,        // Input address offset
  int wr_bank,         // Output bank
  int wr_addr,         // Output address offset
  bool is_acc,         // Write to accumulator
  int dim_len,         // Softmax dimension length
  int batch,           // Batch size
  bool log_mode        // LogSoftmax mode
) {
  // Encode special field
  uint64_t special = (dim_len & 0x3FF) |
                     ((batch & 0x3FF) << 10) |
                     ((log_mode ? 1 : 0) << 20);

  // Encode instruction
  uint64_t inst = encode_softmax_inst(
    iter, op1_bank, op1_addr, wr_bank, wr_addr, is_acc, special
  );

  // Issue RoCC instruction
  ROCC_INSTRUCTION(SOFTMAX_OPCODE, inst);

  // Wait for completion
  wait_softmax_complete();
}
```

### 9.3 编译器优化

#### Fusion优化

将Softmax与其他算子融合：

```
// Before
%1 = matmul(%Q, %K_T)  // Attention scores
%2 = softmax(%1)       // Attention weights
%3 = matmul(%2, %V)    // Attention output

// After (partial fusion)
%2 = matmul_softmax(%Q, %K_T)  // Fused
%3 = matmul(%2, %V)
```

#### Tiling优化

大张量分块处理，优化内存访问：

```
// Original
%out = softmax(%in : tensor<128x512xf32>, dim=1)

// Tiled (process 16 rows at a time)
for i in 0..8:
  %tile_in = extract(%in, i*16, 16x512)
  %tile_out = softmax(%tile_in, dim=1)
  insert(%out, %tile_out, i*16)
```


## 10. 使用示例 (Usage Examples)

### 10.1 基本用法

```scala
// Instantiate Softmax unit
val softmaxUnit = Module(new SoftmaxUnit)

// Connect to Ball Domain
softmaxUnit.io.cmdReq <> ballDomain.io.softmaxReq
ballDomain.io.softmaxResp <> softmaxUnit.io.cmdResp

// Connect to memory system
for (i <- 0 until sp_banks) {
  scratchpad.io.read(i) <> softmaxUnit.io.sramRead(i)
  scratchpad.io.write(i) <> softmaxUnit.io.sramWrite(i)
}

for (i <- 0 until acc_banks) {
  accumulator.io.read(i) <> softmaxUnit.io.accRead(i)
  accumulator.io.write(i) <> softmaxUnit.io.accWrite(i)
}

// Status monitoring
val softmaxStatus = softmaxUnit.io.status
```

### 10.2 单次Softmax

```c
// Process single softmax (16 elements)
float input[16] = {1.0, 2.0, 3.0, ..., 16.0};
float output[16];

// Load input to SRAM bank 0, address 0x100
load_to_sram(0, 0x100, input, 16);

// Issue Softmax instruction
softmax_hw(
  input, output,
  1,      // iter = 1 (single vector)
  0,      // op1_bank = 0
  0x100,  // op1_addr
  1,      // wr_bank = 1
  0x200,  // wr_addr
  false,  // is_acc = false
  16,     // dim_len = 16
  1,      // batch = 1
  false   // log_mode = false
);

// Read result from SRAM bank 1, address 0x200
read_from_sram(1, 0x200, output, 16);
```

### 10.3 批量处理（多头注意力）

```c
// Process 8 attention heads, each with 64 sequence positions
#define NUM_HEADS 8
#define SEQ_LEN 64
#define VECLANE 16
#define TOTAL_VECS ((NUM_HEADS * SEQ_LEN) / VECLANE)

float attention_scores[NUM_HEADS][SEQ_LEN];
float attention_weights[NUM_HEADS][SEQ_LEN];

// Load to SRAM
dma_to_sram(0, 0, attention_scores, NUM_HEADS * SEQ_LEN);

// Issue batched Softmax (each head independently)
for (int head = 0; head < NUM_HEADS; head++) {
  softmax_hw(
    NULL, NULL,
    SEQ_LEN / VECLANE,  // iter = 4 (64/16)
    0,                  // op1_bank = 0
    head * SEQ_LEN,     // op1_addr
    0,                  // wr_bank = 0 (in-place)
    head * SEQ_LEN,     // wr_addr
    false,              // is_acc = false
    SEQ_LEN,            // dim_len = 64
    1,                  // batch = 1 (per head)
    false               // log_mode = false
  );
}

// Read results
dma_from_sram(0, 0, attention_weights, NUM_HEADS * SEQ_LEN);
```

### 10.4 LogSoftmax模式（用于分类）

```c
// Classification layer with 1000 classes
#define NUM_CLASSES 1000
#define NUM_SAMPLES 16

float logits[NUM_SAMPLES][NUM_CLASSES];
float log_probs[NUM_SAMPLES][NUM_CLASSES];

// Load logits to ACC
dma_to_acc(0, 0, logits, NUM_SAMPLES * NUM_CLASSES);

// Process each sample
softmax_hw(
  NULL, NULL,
  (NUM_SAMPLES * NUM_CLASSES) / VECLANE,
  0,                  // op1_bank = 0
  0,                  // op1_addr = 0
  1,                  // wr_bank = 1
  0,                  // wr_addr = 0
  true,               // is_acc = true (ACC mode)
  NUM_CLASSES,        // dim_len = 1000
  NUM_SAMPLES,        // batch = 16
  true                // log_mode = true (LogSoftmax)
);

// Read log probabilities
dma_from_acc(1, 0, log_probs, NUM_SAMPLES * NUM_CLASSES);
```

### 10.5 与Attention流水线集成

```c
// Transformer Self-Attention: Softmax(QK^T/sqrt(d))V

// Step 1: Q × K^T matrix multiplication
matmul_hw(Q, K_T, QK, seq_len, seq_len, head_dim);

// Step 2: Scale by 1/sqrt(d)
vecscale_hw(QK, 1.0/sqrt(head_dim), seq_len * seq_len);

// Step 3: Softmax over last dimension
softmax_hw(QK, attention_weights,
           seq_len,           // iter
           0, 0,              // op1_bank, op1_addr
           1, 0,              // wr_bank, wr_addr
           false,             // is_acc
           seq_len,           // dim_len
           1,                 // batch
           false);            // log_mode

// Step 4: Attention weights × V
matmul_hw(attention_weights, V, output, seq_len, head_dim, seq_len);
```

### 10.6 Status信号监控

```scala
// 性能计数器集成
val perfCounters = Module(new PerfCounters)
perfCounters.io.softmax_idle := softmaxUnit.io.status.idle
perfCounters.io.softmax_running := softmaxUnit.io.status.running

// 调试时等待Softmax完成
def waitSoftmaxComplete(): Unit = {
  while (!softmaxUnit.io.status.idle) {
    if (softmaxUnit.io.status.init) {
      printf("Softmax: Finding max and loading data...\n")
    } else if (softmaxUnit.io.status.running) {
      printf("Softmax: Computing exp and normalizing...\n")
    }
  }
  printf("Softmax: Completed %d batches\n", softmaxUnit.io.status.iter)
}

// 流水线协调
when(softmaxUnit.io.status.ready && matmulUnit.io.status.complete) {
  // Softmax ready and MATMUL finished, issue Softmax
  softmaxUnit.io.cmdReq.valid := true.B
}
```


## 11. 设计考虑与优化

### 11.1 内存带宽优化

Softmax需要多次访问相同数据（FindMax、ExpSum、Normalize），内存带宽是瓶颈：

**优化策略**：
1. **片上缓冲**: 使用大容量`vec_buffer`缓存中间结果，减少SRAM访问
2. **Bank交错**: 输入输出使用不同Bank，避免读写冲突
3. **流水线重叠**: FindMax阶段就开始缓存数据，ExpSum直接从缓冲读取

### 11.2 面积与性能权衡

**高性能配置** (面积大，延迟低)：
- 16个并行除法器
- 大缓冲区（支持dim=1024）
- 高精度exp LUT（512项）

**低面积配置** (面积小，延迟高)：
- 1-2个除法器，时分复用
- 小缓冲区（支持dim=256）
- 简化exp近似（多项式）

### 11.3 精度与量化

**FP32模式**: 最高精度，适合训练
**FP16模式**: 平衡精度和性能，适合推理
**INT8模式**: 需要仔细设计量化方案：
```
// INT8量化Softmax
1. Dequantize: x_fp = (x_int - zero_point) * scale
2. Softmax: y_fp = Softmax(x_fp)
3. Quantize: y_int = round(y_fp / out_scale) + out_zero_point
```

### 11.4 温度参数支持（可选扩展）

某些应用（如采样生成）需要温度参数调节Softmax分布：

```
Softmax(x/T) where T is temperature
```

可在special字段中增加温度参数：
```
special[30:21] - temperature (10-bit fixed-point, default=1.0)
```


## 12. 应用场景

### 12.1 Transformer Attention

Softmax是Attention机制的核心组件：
```
Attention(Q,K,V) = Softmax(QK^T/√d)V
```

### 12.2 分类层

输出层的Softmax将logits转换为概率分布：
```
P(class_i) = Softmax(logits)_i
```

### 12.3 强化学习

策略网络输出动作概率分布：
```
π(a|s) = Softmax(policy_logits(s))
```

### 12.4 序列生成

语言模型的下一个token预测：
```
P(token|context) = Softmax(logits)
```


## 13. 未来扩展

### 13.1 Fused Softmax

与其他操作融合，减少内存访问：
- **Attention Fusion**: QK^T + Scale + Softmax + Mask
- **LayerNorm-Softmax**: 前置归一化后直接Softmax

### 13.2 稀疏Softmax

仅对非零或top-k元素计算Softmax，降低计算量。

### 13.3 近似Softmax

使用更快速的近似算法（如Polynomial Softmax）以降低延迟。

### 13.4 多维度Softmax

支持对任意维度（不仅是最后一维）执行Softmax。


## 14. 参考文献

1. Vaswani et al., "Attention Is All You Need", NeurIPS 2017
2. NVIDIA, "Efficient Softmax Approximation for GPUs", ICML 2018
3. Google TPU, "Systolic Array Based Softmax Implementation"
4. ARM, "Optimizing Softmax for Edge Devices", 2020
