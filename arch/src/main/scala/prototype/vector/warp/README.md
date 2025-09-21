# 线程束模块 (Warp)

## 概述

线程束模块实现了向量处理单元中的线程束管理功能，位于 `prototype/vector/warp` 路径下。该模块将多个线程组织成网格结构，实现并行计算和数据流管理。

## 文件结构

```
warp/
├── MeshWarp.scala    - 网格线程束实现
└── VecBall.scala     - 向量球形处理器
```

## 核心组件

### MeshWarp - 网格线程束

MeshWarp 实现了一个 32 线程的网格结构，包含 16 个乘法线程和 16 个级联线程：

```scala
class MeshWarp(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new MeshWarpInput))
    val out = Decoupled(new MeshWarpOutput)
  })
}
```

#### 输入输出接口

```scala
class MeshWarpInput extends Bundle {
  val op1 = Vec(16, UInt(8.W))        // 第一个操作数向量
  val op2 = Vec(16, UInt(8.W))        // 第二个操作数向量
  val thread_id = UInt(10.W)          // 线程标识符
}

class MeshWarpOutput extends Bundle {
  val res = Vec(16, UInt(32.W))       // 结果向量
}
```

#### 线程配置

网格中的线程按以下规则配置：

```scala
val threadMap = (0 until 32).map { i =>
  val threadName = i.toString
  val opType = if (i < 16) "mul" else "cascade"
  val bond = if (opType == "mul") {
    BondParam("vvv", inputWidth = 8, outputWidth = 32)
  } else {
    BondParam("vvv", inputWidth = 32, outputWidth = 32)
  }
  val op = OpParam(opType, bond)
  val thread = ThreadParam(16, s"attr$threadName", threadName, op)
  threadName -> thread
}.toMap
```

线程分配：
- 线程 0-15：乘法操作线程 (8位输入 → 32位输出)
- 线程 16-31：级联操作线程 (32位输入 → 32位输出)

#### 数据流连接

网格中的数据流按以下方式连接：

```scala
// 连接mul线程的输出到cascade线程的输入
casBond.in.bits.in1 := mulBond.out.bits.out
mulBond.out.ready   := casBond.in.ready

// 级联连接cascade线程
if (i == 0) {
  casBond.in.bits.in2 := VecInit(Seq.fill(16)(0.U(32.W)))
} else {
  casBond.in.bits.in2 := prevCasBond.out.bits.out
}
```

数据流路径：
1. 输入数据 → 乘法线程 (thread 0-15)
2. 乘法结果 → 级联线程 (thread 16-31)
3. 级联线程间串行连接
4. 最终结果从 thread 31 输出

### VecBall - 向量球形处理器

VecBall 是 MeshWarp 的封装器，提供状态管理和迭代控制：

```scala
class VecBall(implicit p: Parameters) extends Module {
  val io = IO(new VecBallIO())
}
```

#### 接口定义

```scala
class VecBallIO extends BallIO {
  val op1In = Flipped(Valid(Vec(16, UInt(8.W))))    // 操作数1输入
  val op2In = Flipped(Valid(Vec(16, UInt(8.W))))    // 操作数2输入
  val rstOut = Decoupled(Vec(16, UInt(32.W)))       // 结果输出
}

class BallIO extends Bundle {
  val iterIn = Flipped(Decoupled(UInt(10.W)))       // 迭代次数输入
  val iterOut = Valid(UInt(10.W))                   // 当前迭代输出
}
```

#### 状态管理

VecBall 维护以下内部状态：

```scala
val start  = RegInit(false.B)      // 开始标志
val arrive = RegInit(false.B)      // 到达标志
val done   = RegInit(false.B)      // 完成标志
val iter   = RegInit(0.U(10.W))    // 总迭代次数
val iterCounter = RegInit(0.U(10.W)) // 当前迭代计数
```

#### 线程调度

VecBall 使用轮转调度分配线程：

```scala
val threadId = RegInit(0.U(4.W))
when (io.op1In.valid && io.op2In.valid && threadId < 15.U) {
  threadId := threadId + 1.U
} .elsewhen (io.op1In.valid && io.op2In.valid && threadId === 15.U) {
  threadId := 0.U
}
```

## 使用方法

### 创建 MeshWarp 实例

```scala
val meshWarp = Module(new MeshWarp()(p))

// 连接输入
meshWarp.io.in.valid := inputValid
meshWarp.io.in.bits.op1 := operand1
meshWarp.io.in.bits.op2 := operand2
meshWarp.io.in.bits.thread_id := selectedThread

// 连接输出
outputValid := meshWarp.io.out.valid
result := meshWarp.io.out.bits.res
meshWarp.io.out.ready := outputReady
```

### 创建 VecBall 实例

```scala
val vecBall = Module(new VecBall()(p))

// 设置迭代次数
vecBall.io.iterIn.valid := iterValid
vecBall.io.iterIn.bits := totalIterations

// 输入数据
vecBall.io.op1In.valid := dataValid
vecBall.io.op1In.bits := inputVector1
vecBall.io.op2In.valid := dataValid
vecBall.io.op2In.bits := inputVector2

// 获取结果
outputReady := vecBall.io.rstOut.ready
when(vecBall.io.rstOut.valid) {
  result := vecBall.io.rstOut.bits
}
```

## 计算模式

### 向量乘法累加

MeshWarp 实现的计算模式：

1. **乘法阶段**：16 个乘法线程并行计算 `op1[i] * op2[i]`
2. **累加阶段**：16 个级联线程串行累加乘法结果
3. **输出阶段**：输出最终的累加向量

### 迭代处理

VecBall 支持多次迭代处理：

1. 设置迭代次数 `iterIn`
2. 循环输入数据对
3. 监控迭代计数 `iterOut`
4. 检查完成状态

## 性能特性

- **并行度**：16 个乘法操作并行执行
- **流水线**：支持连续数据流处理
- **吞吐量**：每周期可处理一个 16 元素向量对
- **延迟**：乘法 + 级联的组合延迟

## 相关模块

- [线程模块](../thread/README.md) - 提供基础线程实现
- [向量操作模块](../op/README.md) - 提供乘法和级联操作
- [绑定模块](../bond/README.md) - 提供数据接口
