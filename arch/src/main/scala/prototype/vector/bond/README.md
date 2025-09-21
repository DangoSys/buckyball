# 绑定模块 (Bond)

## 概述

绑定模块实现了向量处理单元中的数据接口和同步机制，位于 `prototype/vector/bond` 路径下。该模块定义了线程间的数据传递接口，支持不同类型的数据绑定模式。

## 文件结构

```
bond/
├── BondWrapper.scala    - 绑定包装器基类
└── vvv.scala           - VVV 绑定实现
```

## 核心组件

### VVV - 向量到向量绑定

VVV (Vector-Vector-Vector) 绑定实现了双输入向量到单输出向量的数据接口：

```scala
class VVV(implicit p: Parameters) extends Bundle {
  val lane = p(ThreadKey).get.lane
  val bondParam = p(ThreadBondKey).get
  val inputWidth = bondParam.inputWidth
  val outputWidth = bondParam.outputWidth

  // Input interface (Flipped Decoupled)
  val in = Flipped(Decoupled(new Bundle {
    val in1 = Vec(lane, UInt(inputWidth.W))
    val in2 = Vec(lane, UInt(inputWidth.W))
  }))

  // Decoupled output interface
  val out = Decoupled(new Bundle {
    val out = Vec(lane, UInt(outputWidth.W))
  })
}
```

#### 接口说明

**输入接口**：
- `in.bits.in1`: 第一个输入向量，位宽为 `inputWidth`
- `in.bits.in2`: 第二个输入向量，位宽为 `inputWidth`
- `in.valid`: 输入数据有效信号
- `in.ready`: 输入就绪信号

**输出接口**：
- `out.bits.out`: 输出向量，位宽为 `outputWidth`
- `out.valid`: 输出数据有效信号
- `out.ready`: 输出就绪信号

#### 参数配置

VVV 绑定的参数通过配置系统获取：

```scala
val lane = p(ThreadKey).get.lane                    // 向量通道数
val bondParam = p(ThreadBondKey).get                // 绑定参数
val inputWidth = bondParam.inputWidth               // 输入位宽
val outputWidth = bondParam.outputWidth             // 输出位宽
```

### CanHaveVVVBond - VVV 绑定特质

CanHaveVVVBond 特质为线程提供 VVV 绑定功能：

```scala
trait CanHaveVVVBond { this: BaseThread =>
  val vvvBond = params(ThreadBondKey).filter(_.bondType == "vvv").map { bondParam =>
    IO(new VVV()(params))
  }

  def getVVVBond = vvvBond
}
```

#### 使用方式

线程类通过混入该特质获得 VVV 绑定能力：

```scala
class MulThread(implicit p: Parameters) extends BaseThread
  with CanHaveMulOp
  with CanHaveVVVBond {

  // 连接操作和绑定
  for {
    op <- mulOp
    bond <- vvvBond
  } {
    op.io.in <> bond.in
    op.io.out <> bond.out
  }
}
```

### BondWrapper - 绑定包装器

BondWrapper 提供了基于 Diplomacy 的绑定封装：

```scala
abstract class BondWrapper(implicit p: Parameters) extends LazyModule {
  val bondName = "vvv"

  def to[T](name: String)(body: => T): T = {
    LazyScope(s"bond_to_${name}", s"Bond_${bondName}_to_${name}") { body }
  }

  def from[T](name: String)(body: => T): T = {
    LazyScope(s"bond_from_${name}", s"Bond_${bondName}_from_${name}") { body }
  }
}
```

#### 作用域管理

BondWrapper 提供了命名作用域管理功能：
- `to()`: 创建输出方向的绑定作用域
- `from()`: 创建输入方向的绑定作用域

## 绑定类型

### VVV 绑定模式

VVV 绑定支持以下数据流模式：

1. **双输入单输出**：两个向量输入，一个向量输出
2. **位宽转换**：支持输入和输出位宽不同
3. **向量并行**：支持多通道并行数据传输

### 数据流控制

VVV 绑定使用 Decoupled 接口进行流控：

```scala
// 生产者端
producer.io.out.valid := dataReady
producer.io.out.bits.in1 := inputVector1
producer.io.out.bits.in2 := inputVector2

// 消费者端
consumer.io.in.ready := canAcceptData
when(consumer.io.in.fire) {
  processData(consumer.io.in.bits.out)
}
```

## 配置参数

### 绑定参数

绑定参数通过 `BondParam` 定义：

```scala
case class BondParam(
  bondType: String,           // 绑定类型 ("vvv")
  inputWidth: Int = 8,        // 输入位宽
  outputWidth: Int = 32       // 输出位宽
)
```

### 配置示例

```scala
val bondConfig = BondParam(
  bondType = "vvv",
  inputWidth = 8,
  outputWidth = 32
)

val threadConfig = ThreadParam(
  lane = 16,
  attr = "vector",
  threadName = "mul_thread",
  Op = OpParam("mul", bondConfig)
)
```

## 使用方法

### 创建 VVV 绑定

```scala
// 在线程中使用 VVV 绑定
class CustomThread(implicit p: Parameters) extends BaseThread
  with CanHaveVVVBond {

  // 获取绑定接口
  for (bond <- vvvBond) {
    // 连接输入
    bond.in.valid := inputValid
    bond.in.bits.in1 := inputVector1
    bond.in.bits.in2 := inputVector2

    // 连接输出
    outputValid := bond.out.valid
    outputVector := bond.out.bits.out
    bond.out.ready := outputReady
  }
}
```

### 绑定连接

```scala
// 连接两个模块的绑定接口
val producer = Module(new ProducerThread())
val consumer = Module(new ConsumerThread())

// 直接连接绑定接口
for {
  prodBond <- producer.vvvBond
  consBond <- consumer.vvvBond
} {
  consBond.in <> prodBond.out
}
```

## 同步机制

### 握手协议

VVV 绑定使用标准的 Decoupled 握手协议：

1. **数据准备**：生产者设置 `valid` 和 `bits`
2. **接收就绪**：消费者设置 `ready`
3. **数据传输**：当 `valid && ready` 时完成传输
4. **状态更新**：双方更新内部状态

### 背压处理

绑定接口支持背压机制：

```scala
// 当下游未就绪时，上游会等待
when(!downstream.ready) {
  upstream.valid := false.B
  // 保持数据不变
}
```

## 扩展性

### 新绑定类型

可以通过类似的模式定义新的绑定类型：

```scala
// 单输入单输出绑定
class VV(implicit p: Parameters) extends Bundle {
  val in = Flipped(Decoupled(Vec(lane, UInt(inputWidth.W))))
  val out = Decoupled(Vec(lane, UInt(outputWidth.W)))
}

// 对应的特质
trait CanHaveVVBond { this: BaseThread =>
  val vvBond = params(ThreadBondKey).filter(_.bondType == "vv").map { _ =>
    IO(new VV()(params))
  }
}
```

### 参数化支持

绑定模块支持完全参数化配置：

- 向量通道数可配置
- 输入输出位宽可配置
- 绑定类型可扩展

## 相关模块

- [线程模块](../thread/README.md) - 提供绑定的使用环境
- [向量操作模块](../op/README.md) - 绑定的数据处理逻辑
- [向量处理单元](../README.md) - 上层向量处理器
