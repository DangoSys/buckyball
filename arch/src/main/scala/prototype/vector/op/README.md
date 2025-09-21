# 向量操作模块 (Vector Operations)

## 概述

向量操作模块实现了向量处理单元中的具体计算操作，位于 `prototype/vector/op` 路径下。该模块提供了不同类型的向量运算实现，包括乘法操作和级联操作。

## 文件结构

```
op/
├── cascade.scala    - 级联加法操作
└── mul.scala       - 乘法操作
```

## 核心组件

### CascadeOp - 级联加法操作

CascadeOp 实现向量元素的逐元素加法操作：

```scala
class CascadeOp(implicit p: Parameters) extends Module {
  val lane = p(ThreadKey).get.lane
  val bondParam = p(ThreadBondKey).get
  val outputWidth = bondParam.outputWidth

  val io = IO(new VVV()(p))
}
```

#### 操作逻辑

```scala
val reg1 = RegInit(VecInit(Seq.fill(lane)(0.U(outputWidth.W))))
val valid1 = RegInit(false.B)

when (io.in.valid) {
  valid1 := true.B
  reg1 := io.in.bits.in1.zip(io.in.bits.in2).map { case (a, b) => a + b }
}
```

**功能说明**：
- 接收两个输入向量 `in1` 和 `in2`
- 执行逐元素加法：`out[i] = in1[i] + in2[i]`
- 使用寄存器缓存计算结果
- 支持流水线操作

#### 流控机制

```scala
io.in.ready := io.out.ready

when (io.out.ready && valid) {
  io.out.valid := true.B
  io.out.bits.out := reg1
}.otherwise {
  io.out.valid := false.B
  io.out.bits.out := VecInit(Seq.fill(lane)(0.U(outputWidth.W)))
}
```

### MulOp - 乘法操作

MulOp 实现向量乘法操作，支持广播模式：

```scala
class MulOp(implicit p: Parameters) extends Module {
  val lane = p(ThreadKey).get.lane
  val bondParam = p(ThreadBondKey).get
  val inputWidth = bondParam.inputWidth

  val io = IO(new VVV()(p))
}
```

#### 操作逻辑

```scala
val reg1 = RegInit(VecInit(Seq.fill(lane)(0.U(inputWidth.W))))
val reg2 = RegInit(VecInit(Seq.fill(lane)(0.U(inputWidth.W))))
val cnt = RegInit(0.U(log2Ceil(lane).W))
val active = RegInit(false.B)

when (io.in.valid) {
  reg1 := io.in.bits.in1
  reg2 := io.in.bits.in2
  cnt := 0.U
  active := true.B
}
```

**功能说明**：
- 接收两个输入向量并缓存到寄存器
- 使用计数器 `cnt` 控制输出序列
- 实现广播乘法：`out[i] = reg1[cnt] * reg2[i]`

#### 序列输出

```scala
for (i <- 0 until lane) {
  io.out.bits.out(i) := reg1(cnt) * reg2(i)
}

when (active && io.out.ready) {
  cnt := cnt + 1.U
  when (cnt === (lane-1).U) {
    active := false.B
  }
}
```

**输出模式**：
- 每个周期输出一组乘法结果
- `reg1[cnt]` 与 `reg2` 的所有元素相乘
- 计数器递增，实现序列输出

## 操作特质

### CanHaveCascadeOp - 级联操作特质

```scala
trait CanHaveCascadeOp { this: BaseThread =>
  val cascadeOp = params(ThreadOpKey).filter(_.OpType == "cascade").map { opParam =>
    Module(new CascadeOp()(params))
  }

  def getCascadeOp = cascadeOp
}
```

### CanHaveMulOp - 乘法操作特质

```scala
trait CanHaveMulOp { this: BaseThread =>
  val mulOp = params(ThreadOpKey).filter(_.OpType == "mul").map { opParam =>
    Module(new MulOp()(params))
  }

  def getMulOp = mulOp
}
```

## 使用方法

### 在线程中使用操作

```scala
class CasThread(implicit p: Parameters) extends BaseThread
  with CanHaveCascadeOp
  with CanHaveVVVBond {

  // 连接操作和绑定
  for {
    op <- cascadeOp
    bond <- vvvBond
  } {
    op.io.in <> bond.in
    op.io.out <> bond.out
  }
}
```

### 配置操作参数

```scala
val opParam = OpParam(
  OpType = "cascade",                    // 操作类型
  bondType = BondParam(
    bondType = "vvv",
    inputWidth = 32,
    outputWidth = 32
  )
)
```

## 操作类型对比

### CascadeOp vs MulOp

| 特性 | CascadeOp | MulOp |
|------|-----------|-------|
| 操作类型 | 逐元素加法 | 广播乘法 |
| 输入位宽 | 任意 | 通常较小 |
| 输出位宽 | 任意 | 通常较大 |
| 延迟 | 1 周期 | lane 周期 |
| 吞吐量 | 每周期 1 组 | 每 lane 周期 1 组 |
| 资源消耗 | 加法器 × lane | 乘法器 × lane |

### 应用场景

**CascadeOp 适用于**：
- 向量加法运算
- 累加操作
- 数据合并

**MulOp 适用于**：
- 矩阵向量乘法
- 卷积运算
- 缩放操作

## 数据流模式

### CascadeOp 数据流

```
输入: [a0, a1, ..., an], [b0, b1, ..., bn]
      ↓
计算: [a0+b0, a1+b1, ..., an+bn]
      ↓
输出: [c0, c1, ..., cn] (1 周期)
```

### MulOp 数据流

```
输入: [a0, a1, ..., an], [b0, b1, ..., bn]
      ↓
周期0: [a0*b0, a0*b1, ..., a0*bn]
周期1: [a1*b0, a1*b1, ..., a1*bn]
...
周期n: [an*b0, an*b1, ..., an*bn]
```

## 扩展操作

### 添加新操作

可以通过类似的模式添加新的向量操作：

```scala
class SubOp(implicit p: Parameters) extends Module {
  val io = IO(new VVV()(p))

  // 实现减法操作
  io.out.bits.out := io.in.bits.in1.zip(io.in.bits.in2).map {
    case (a, b) => a - b
  }
}

trait CanHaveSubOp { this: BaseThread =>
  val subOp = params(ThreadOpKey).filter(_.OpType == "sub").map { _ =>
    Module(new SubOp()(params))
  }
}
```

### 复杂操作

对于更复杂的操作，可以组合多个基础操作：

```scala
class FMAOp(implicit p: Parameters) extends Module {
  // 融合乘加操作: out = a * b + c
  val mulOp = Module(new MulOp())
  val addOp = Module(new CascadeOp())

  // 连接操作流水线
  addOp.io.in.bits.in1 <> mulOp.io.out.bits.out
  // ...
}
```

## 性能优化

### 流水线优化

- 使用寄存器缓存中间结果
- 支持连续数据流处理
- 最小化组合逻辑延迟

### 资源优化

- 根据操作类型选择合适的硬件资源
- 支持资源共享和复用
- 可配置的并行度

## 相关模块

- [绑定模块](../bond/README.md) - 提供数据接口
- [线程模块](../thread/README.md) - 提供操作的执行环境
- [向量处理单元](../README.md) - 上层向量处理器
