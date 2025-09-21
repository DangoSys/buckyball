# 矩阵转置加速器

## 概述

该目录实现了 BuckyBall 的矩阵转置加速器，用于矩阵转置操作。位于 `arch/src/main/scala/prototype/transpose` 下，作为矩阵转置加速器，支持流水线化的转置操作。

实现的核心组件：
- **Transpose.scala**: 流水线化转置器实现

## 代码结构

```
transpose/
└── Transpose.scala  - 流水线转置器
```

### 模块职责

**Transpose.scala** (转置实现层)
- 实现 PipelinedTransposer 模块
- 管理矩阵数据的读取、转置和写回
- 提供 Ball 域命令接口

## 模块说明

### Transpose.scala

**主要功能**: 实现流水线化的矩阵转置操作

**状态机定义**:
```scala
val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
val state = RegInit(idle)
```

**存储结构**:
```scala
// 矩阵存储寄存器 (veclane x veclane)
val regArray = Reg(Vec(b.veclane, Vec(b.veclane, UInt(b.inputType.getWidth.W))))
```

**计数器管理**:
```scala
val readCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))
val respCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))
val writeCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))
```

**指令寄存器**:
```scala
val robid_reg = RegInit(0.U(10.W))    // ROB ID
val waddr_reg = RegInit(0.U(10.W))    // 写地址
val wbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))  // 写bank
val raddr_reg = RegInit(0.U(10.W))    // 读地址
val rbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))  // 读bank
val iter_reg = RegInit(0.U(10.W))     // 迭代计数
```

**接口定义**:
```scala
val io = IO(new Bundle {
  val cmdReq = Flipped(Decoupled(new BallRsIssue))
  val cmdResp = Decoupled(new BallRsComplete)
  val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(...)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(...)))
})
```

**处理流程**:
1. **idle**: 等待命令，解析转置参数
2. **sRead**: 按行读取矩阵数据到寄存器阵列
3. **sWrite**: 按列写回转置后的数据
4. **complete**: 发送完成信号

**转置算法**:
- 使用 veclane×veclane 的寄存器阵列存储矩阵
- 按行读取，按列写回实现转置
- 支持任意大小矩阵的分块转置

## 使用方法

### 实现细节

**状态机**:
```scala
val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
```
- `idle`: 等待指令
- `sRead`: 读取矩阵数据
- `sWrite`: 写入转置结果
- `complete`: 完成并响应

**寄存器阵列**:
```scala
val regArray = Reg(Vec(b.veclane, Vec(b.veclane, UInt(b.inputType.getWidth.W))))
```
使用 veclane×veclane 的寄存器阵列缓存矩阵数据。

**转置操作**:
- 读取阶段：按行读取数据存入 `regArray(row)(col)`
- 写入阶段：按列读取 `regArray(i)(col)` 组成新行写出

### 配置参数

**矩阵大小**: 由 b.veclane 参数决定
**数据位宽**: 由 b.inputType.getWidth 决定
**Bank 配置**: 支持多 bank SRAM 访问

### 注意事项

1. **矩阵大小限制**: 最大支持 veclane×veclane 的矩阵
2. **内存带宽**: 转置操作对内存带宽要求较高
3. **寄存器开销**: 需要 veclane² 个寄存器存储矩阵
4. **地址计算**: 转置后的地址计算需要正确处理
5. **流水线控制**: 读写计数器需要正确同步
