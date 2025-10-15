# 任务：用 BuckyBall 框架从头实现 Gemmini NPU

## 目标
**用 BuckyBall 框架从头实现 Gemmini，能够执行除 mvin/mvout 外的所有 Gemmini 指令。**

- ✅ 使用 BuckyBall 的框架（memdomain 提供 DMA/内存）
- ✅ 实现 Gemmini 的所有计算指令
- ✅ 兼容 Gemmini ISA（指令编码、参数格式）
- ❌ 不需要实现 mvin/mvout（框架已提供）

## 执行指示（Master Agent）

**请立即开始实现，不要过度收集信息！**

1. 快速查询 Gemmini 架构（1-2次 Deepwiki）
2. 立即开始调用 spec_agent 和 code_agent 实现第一个 Ball
3. 逐个实现所有 Ball，不要等到"完全了解"才开始

## Gemmini ISA 完整指令集

参考：`arch/thirdparty/chipyard/generators/gemmini/src/main/scala/gemmini/GemminiISA.scala`

### ❌ 不需要实现（框架已提供 DMA）

| 指令 | funct 值 | 说明 |
|------|----------|------|
| `LOAD_CMD` (mvin) | 2 | DRAM → Scratchpad/Accumulator，由框架 memdomain 提供 |
| `LOAD2_CMD` (mvin2) | 1 | 独立配置的 mvin，由框架 memdomain 提供 |
| `LOAD3_CMD` (mvin3) | 14 | 独立配置的 mvin，由框架 memdomain 提供 |
| `STORE_CMD` (mvout) | 3 | Scratchpad/Accumulator → DRAM，由框架 memdomain 提供 |

### ✅ 需要实现的指令（计算 + 配置）

#### 1. 配置指令
| 指令 | funct 值 | 说明 | 实现位置 |
|------|----------|------|----------|
| `CONFIG_CMD` | 0 | 配置指令（通过 rs1[2:0] 区分子类型）| 各 Ball 或系统层 |
| - `CONFIG_EX` | rs1[2:0]=0 | 配置执行单元（dataflow, activation, transpose）| MatMulBall |
| - `CONFIG_LOAD` | rs1[2:0]=1 | 配置加载参数（stride, scale）| 系统层 |
| - `CONFIG_STORE` | rs1[2:0]=2 | 配置存储参数（stride, activation, pooling）| 系统层 |
| - `CONFIG_NORM` | rs1[2:0]=3 | 配置归一化参数 | NormBall |
| `FLUSH_CMD` | 7 | TLB 刷新 | 系统层 |
| `CLKGATE_EN` | 22 | 时钟门控使能 | 系统层（可选）|

#### 2. 计算指令 ⭐
| 指令 | funct 值 | 说明 | 实现位置 |
|------|----------|------|----------|
| `PRELOAD_CMD` | 6 | 预加载矩阵到寄存器 | MatMulBall |
| `COMPUTE_AND_FLIP_CMD` | 4 | 计算并翻转累加器 bank | MatMulBall |
| `COMPUTE_AND_STAY_CMD` | 5 | 计算保持累加器 bank | MatMulBall |

#### 3. 循环优化指令
| 指令 | funct 值 | 说明 | 实现位置 |
|------|----------|------|----------|
| `LOOP_WS` | 8 | Weight-stationary 矩阵乘循环 | MatMulBall（扩展）|
| `LOOP_WS_CONFIG_*` | 9-13 | 循环配置（bounds, addrs, strides）| MatMulBall（扩展）|
| `LOOP_CONV_WS` | 15 | 卷积循环 | Im2colBall + MatMulBall |
| `LOOP_CONV_WS_CONFIG_*` | 16-21 | 卷积循环配置 | Im2colBall（扩展）|

**实现优先级：**
1. **第一阶段（必须）**：基础计算指令（PRELOAD, COMPUTE, CONFIG_EX）
2. **第二阶段（必须）**：配置指令（CONFIG_NORM, CONFIG_LOAD/STORE）
3. **第三阶段（可选）**：循环优化指令（LOOP_WS, LOOP_CONV_WS）

## 系统设计要求

### ⚠️ 实现路径（重要）

**新的 Gemmini NPU 必须在以下路径实现：**

```
arch/src/main/scala/prototype/gemmini/
├── matmul/
│   ├── spec.md
│   ├── MatMulUnit.scala
│   ├── MatMulCtrlUnit.scala
│   ├── MatMulLoadUnit.scala
│   ├── MatMulExUnit.scala
│   └── MatMulStoreUnit.scala
├── im2col/
│   ├── spec.md
│   ├── Im2colUnit.scala
│   └── Im2colCtrlUnit.scala
├── transpose/
│   ├── spec.md
│   ├── TransposeUnit.scala
│   └── TransposeCtrlUnit.scala
└── norm/
    ├── spec.md
    ├── NormUnit.scala
    └── NormCtrlUnit.scala
```

**注意：**
- ✅ 只实现**4个计算 Ball**（MatMul, Im2col, Transpose, Norm）
- ❌ 不要实现 DMA/内存搬运（框架已提供）
- ❌ Scratchpad 和 Accumulator 由框架提供

**禁止操作以下路径：**
- ❌ `arch/src/main/scala/examples/toy/` - 这是参考示例，不要修改！
- ❌ `arch/src/main/scala/prototype/nagisa/` - 这是现有 Ball，不要修改！

**只允许：**
- ✅ 在 `arch/src/main/scala/prototype/gemmini/` 下创建新文件
- ✅ 在系统注册文件中**追加**新内容（不修改已有内容）

### 参考实现（仅供学习，不要修改）
- ToyBuckyBall: `arch/src/main/scala/examples/toy/`
- Ball 示例: `arch/src/main/scala/prototype/nagisa/`

### 需要实现的 Ball

**⚠️ 重要架构说明：**
- **Ball = 计算单元**：只负责计算，不负责 DMA/内存搬运
- **MemDomain/DMA**：由框架提供，Ball 通过 Blink 协议使用
- **Ball 通过 Blink 接口**：读取 scratchpad/accumulator，执行计算，写回结果

需要实现以下**4个计算 Ball**：

1. **MatMulBall**: 矩阵乘法计算（Systolic Array）
   - 对应 Gemmini 的 matmul.preload/compute 指令
   - 实现脉动阵列矩阵乘法：C = A × B + D
   - 支持 weight stationary / output stationary 数据流
   - **参考**：`gemmini/Mesh.scala`, `gemmini/PE.scala`, `gemmini/Tile.scala`

2. **Im2colBall**: 图像到列转换（卷积支持）
   - 对应 Gemmini 的卷积数据重排
   - 将卷积窗口展开为列，支持 stride/padding/kernel size
   - 将卷积转换为矩阵乘法
   - **参考**：`gemmini/Im2Col.scala`

3. **TransposeBall**: 矩阵转置
   - 对应 Gemmini 的转置操作
   - 实现流水线矩阵转置（PipelinedTransposer）
   - 用于数据重排和格式转换
   - **参考**：`gemmini/Transposer.scala`, `gemmini/TransposePreloadUnroller.scala`

4. **NormBall**: 归一化与激活函数
   - 对应 Gemmini 的 Normalizer + Activation
   - 归一化：mean/variance/stddev、LayerNorm、softmax
   - 激活函数：ReLU、ReLU6、GELU
   - **参考**：`gemmini/Normalizer.scala`, `gemmini/Activation.scala`, `gemmini/AccumulatorScale.scala`

**框架已提供（不需要实现）：**
- ❌ DMA：数据搬运（mvin/mvout）由框架的 memdomain 提供
- ❌ Scratchpad：暂存器内存，由框架提供
- ❌ Accumulator：累加器内存，由框架提供
- ❌ 内存控制器：由框架提供
- ❌ TLB/缓存管理：由框架提供

### NPU 系统结构

```
GemminiNPU/
├── DomainDecoder           # 解码 Gemmini 指令
├── BallRSModule            # 指令缓冲
├── BBus                    # 指令路由
├── Ball Units (计算单元):
│   ├── MatMulBall          # 矩阵乘法计算
│   ├── Im2colBall          # 图像到列转换
│   ├── TransposeBall       # 矩阵转置
│   └── NormBall            # 归一化操作
└── [使用框架提供的]
    ├── MemDomain            # 内存域（框架）
    ├── Scratchpad           # 暂存器内存（框架）
    ├── Accumulator          # 累加器内存（框架）
    ├── DMA                  # 数据搬运（框架）
    └── MemRouter            # 内存仲裁（框架）
```

**架构要点：**
- Ball 通过 Blink 接口从 scratchpad/accumulator 读取数据
- Ball 执行计算后写回 scratchpad/accumulator
- 数据的 DRAM ↔ scratchpad/accumulator 搬运由框架的 memdomain 处理

## 实现步骤

### 阶段 1: 规划（master_agent）
1. 使用 Deepwiki 查询 Gemmini 架构和 ToyBuckyBall 实现
2. 规划需要实现的 Ball 列表
3. 分配任务给 spec_agent 和 code_agent

### 阶段 2: 各 Ball 的 Spec（spec_agent）
为每个 Ball 编写 spec.md：
- 参考 Gemmini 的对应功能
- 定义 Blink 接口
- 明确指令参数映射

### 阶段 3: 实现各 Ball（code_agent）
1. 实现 Chisel 硬件模块
2. 定义 ISA 映射（使用 Gemmini 的 funct 值）
3. 注册到系统（DomainDecoder、BBus、RS）
4. 调用 verify_agent 测试

### 阶段 4: 系统集成（master_agent）
1. 创建 GemminiNPU 顶层模块
2. 集成所有 Ball
3. 配置路由和仲裁
4. 端到端测试

## 验证要求

### 功能测试
- 每个指令的基本功能
- 指令组合（如 preload + compute）
- 数据搬运正确性

### 对照测试
参考 Gemmini 的测试用例：
- 矩阵乘法精度
- 卷积操作
- 端到端 DNN 层

## 交付物

1. **Spec 文档**
   - 每个 Ball 的 spec.md
   - 系统架构文档

2. **Chisel 实现**
   - 各 Ball 的 Scala 代码
   - GemminiNPU 顶层模块

3. **ISA 定义**
   - 指令编码映射
   - 软件 API

4. **测试用例**
   - CTest 测试程序
   - Verilator 仿真结果

5. **集成文档**
   - 如何使用 GemminiNPU
   - 与 Gemmini 的差异说明

## 注意事项

1. **兼容性优先**：指令编码尽量与 Gemmini 一致
2. **模块化设计**：每个 Ball 独立，便于调试
3. **参考现有实现**：充分利用 ToyBuckyBall 和现有 Ball
4. **渐进式开发**：先实现核心指令，再扩展循环指令
5. **充分使用 Deepwiki**：遇到不懂的查询 `ucb-bar/gemmini` 和 `DangoSys/buckyball`
