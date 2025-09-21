# 线程模块 (Thread)

## 概述

线程模块实现了向量处理单元中的线程抽象，位于 `prototype/vector/thread` 路径下。该模块定义了线程的基本结构和具体实现，通过组合不同的操作 (Op) 和绑定 (Bond) 来构建特定功能的线程。

## 文件结构

```
thread/
├── BaseThread.scala    - 线程基类定义
├── CasThread.scala     - 级联操作线程
└── MulThread.scala     - 乘法操作线程
```

## 核心组件

### BaseThread - 线程基类

BaseThread 是所有线程的基类，定义了线程的基本参数和配置：

```scala
class BaseThread(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {})
  val params = p
  val threadMap = p(ThreadMapKey)
  val threadParam = threadMap.getOrElse(
    p(ThreadKey).get.threadName,
    throw new Exception(s"ThreadParam not found for threadName: ${p(ThreadKey).get.threadName}")
  )
  val opParam = p(ThreadOpKey).get
  val bondParam = p(ThreadBondKey).get
}
```

### 参数定义

线程模块使用以下参数结构：

```scala
case class ThreadParam(lane: Int, attr: String, threadName: String, Op: OpParam)
case class OpParam(OpType: String, bondType: BondParam)
case class BondParam(bondType: String, inputWidth: Int = 8, outputWidth: Int = 32)
```

参数说明：
- `lane`: 向量通道数量
- `threadName`: 线程名称标识
- `OpType`: 操作类型 ("cascade", "mul")
- `bondType`: 绑定类型 ("vvv")
- `inputWidth`: 输入数据位宽，默认 8 位
- `outputWidth`: 输出数据位宽，默认 32 位

## 具体线程实现

### CasThread - 级联操作线程

CasThread 实现级联加法操作，组合了 CascadeOp 和 VVVBond：

```scala
class CasThread(implicit p: Parameters) extends BaseThread
  with CanHaveCascadeOp
  with CanHaveVVVBond {

  // 连接CascadeOp和VVVBond
  for {
    op <- cascadeOp
    bond <- vvvBond
  } {
    op.io.in <> bond.in
    op.io.out <> bond.out
  }
}
```

功能：对两个输入向量执行逐元素加法操作。

### MulThread - 乘法操作线程

MulThread 实现乘法操作，组合了 MulOp 和 VVVBond：

```scala
class MulThread(implicit p: Parameters) extends BaseThread
  with CanHaveMulOp
  with CanHaveVVVBond {

  // 连接MulOp和VVVBond
  for {
    op <- mulOp
    bond <- vvvBond
  } {
    op.io.in <> bond.in
    op.io.out <> bond.out
  }
}
```

功能：实现向量乘法操作，支持逐周期输出结果。

## 配置系统

线程模块使用 Chipyard 的配置系统进行参数化：

```scala
case object ThreadKey extends Field[Option[ThreadParam]](None)
case object ThreadOpKey extends Field[Option[OpParam]](None)
case object ThreadBondKey extends Field[Option[BondParam]](None)
case object ThreadMapKey extends Field[Map[String, ThreadParam]](Map.empty)
```

配置键说明：
- `ThreadKey`: 当前线程参数
- `ThreadOpKey`: 操作参数
- `ThreadBondKey`: 绑定参数
- `ThreadMapKey`: 线程映射表

## 使用方法

### 创建线程实例

```scala
// 配置参数
val threadParam = ThreadParam(
  lane = 4,
  attr = "vector",
  threadName = "mul_thread",
  Op = OpParam("mul", BondParam("vvv", 8, 32))
)

// 创建线程
val mulThread = Module(new MulThread()(
  new Config((site, here, up) => {
    case ThreadKey => Some(threadParam)
    case ThreadOpKey => Some(threadParam.Op)
    case ThreadBondKey => Some(threadParam.Op.bondType)
  })
))
```

### 连接接口

线程通过 VVV 绑定接口进行数据交互：

```scala
// 输入数据
mulThread.io.in.valid := inputValid
mulThread.io.in.bits.in1 := inputVector1
mulThread.io.in.bits.in2 := inputVector2

// 输出数据
outputValid := mulThread.io.out.valid
outputVector := mulThread.io.out.bits.out
mulThread.io.out.ready := outputReady
```

## 相关模块

- [向量操作模块](../op/README.md) - 提供具体的计算操作
- [绑定模块](../bond/README.md) - 提供数据接口和同步机制
- [向量处理单元](../README.md) - 上层向量处理器
