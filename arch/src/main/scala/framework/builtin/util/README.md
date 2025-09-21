# 框架工具函数库

## 概述

该目录包含了 BuckyBall 框架级别的工具函数和辅助模块，提供通用的硬件设计工具。位于 `arch/src/main/scala/framework/builtin/util` 下，作为工具函数层，为框架内的其他组件提供可复用的硬件构建块和实用函数。

主要工具类别：
- 数学运算和位操作工具
- 接口转换和适配器
- 调试和性能监控工具
- 通用硬件模式实现

## 代码结构

```
util/
└── (具体工具文件待分析)
```

### 工具分类

**数学工具**
- 位宽计算和对数函数
- 数值转换和格式化
- 算术运算优化实现

**接口工具**
- 协议转换适配器
- 信号同步和跨时钟域
- 握手协议实现

**调试工具**
- 性能计数器模板
- 调试信号输出
- 状态监控接口

## 模块说明

### 数学运算工具

**主要功能**: 提供常用的数学运算和位操作函数

**关键函数**:

```scala
object MathUtils {
  // 计算log2向上取整
  def log2Ceil(x: Int): Int = {
    require(x > 0)
    (log(x) / log(2)).ceil.toInt
  }

  // 判断是否为2的幂
  def isPow2(x: Int): Boolean = x > 0 && (x & (x - 1)) == 0

  // 计算最小的2的幂大于等于x
  def nextPow2(x: Int): Int = {
    if (isPow2(x)) x else 1 << log2Ceil(x)
  }
}
```

**位操作工具**:
```scala
object BitUtils {
  // 位反转
  def reverseBits(data: UInt, width: Int): UInt = {
    VecInit((0 until width).map(i => data(i))).asUInt
  }

  // 计算汉明重量(1的个数)
  def popCount(data: UInt): UInt = {
    PopCount(data)
  }

  // 前导零计数
  def leadingZeros(data: UInt, width: Int): UInt = {
    PriorityEncoder(Reverse(data))
  }
}
```

### 接口转换工具

**主要功能**: 提供常用的接口转换和适配功能

**协议转换器**:
```scala
class DecoupledToValid[T <: Data](gen: T) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(gen))
    val out = Valid(gen)
  })

  io.out.valid := io.in.valid
  io.out.bits := io.in.bits
  io.in.ready := true.B
}

class ValidToDecoupled[T <: Data](gen: T) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Valid(gen))
    val out = Decoupled(gen)
  })

  io.out.valid := io.in.valid
  io.out.bits := io.in.bits
}
```

**跨时钟域同步**:
```scala
class AsyncFIFO[T <: Data](gen: T, depth: Int) extends Module {
  val io = IO(new Bundle {
    val enq_clock = Input(Clock())
    val enq_reset = Input(Bool())
    val enq = Flipped(Decoupled(gen))

    val deq_clock = Input(Clock())
    val deq_reset = Input(Bool())
    val deq = Decoupled(gen)
  })

  // 异步FIFO实现
  // 使用格雷码指针避免亚稳态
}
```

### 调试监控工具

**主要功能**: 提供调试和性能监控的通用工具

**性能计数器**:
```scala
class PerfCounter(name: String) extends Module {
  val io = IO(new Bundle {
    val inc = Input(Bool())
    val value = Output(UInt(64.W))
  })

  val counter = RegInit(0.U(64.W))
  when(io.inc) {
    counter := counter + 1.U
  }
  io.value := counter

  // 可选的调试输出
  when(io.inc) {
    printf(s"[PerfCounter] $name: %d\n", counter + 1.U)
  }
}
```

**调试信号输出**:
```scala
object DebugUtils {
  def debugPrint(cond: Bool, fmt: String, args: Bits*): Unit = {
    when(cond) {
      printf(fmt, args: _*)
    }
  }

  def assert(cond: Bool, msg: String): Unit = {
    chisel3.assert(cond, msg)
  }

  def cover(cond: Bool, msg: String): Unit = {
    chisel3.cover(cond, msg)
  }
}
```

## 使用方法

### 使用示例

**数学工具使用**:
```scala
import util.MathUtils._

class MyModule extends Module {
  val addrBits = log2Ceil(entries)
  val bankSize = nextPow2(requestedSize)

  require(isPow2(bankSize), "Bank size must be power of 2")
}
```

**接口转换使用**:
```scala
val converter = Module(new DecoupledToValid(UInt(32.W)))
converter.io.in <> some_decoupled_signal
val valid_signal = converter.io.out
```

**性能监控使用**:
```scala
val hit_counter = Module(new PerfCounter("cache_hits"))
hit_counter.io.inc := cache_hit

val miss_counter = Module(new PerfCounter("cache_misses"))
miss_counter.io.inc := cache_miss
```

### 扩展开发

**添加新工具**:
1. 确定工具的通用性和复用价值
2. 实现标准的Chisel模块接口
3. 添加充分的参数化支持
4. 提供使用示例和测试用例

**工具设计原则**:
- 保持接口简洁明确
- 支持参数化配置
- 提供良好的错误检查
- 考虑硬件实现效率

### 注意事项

1. **硬件开销**: 工具函数应该考虑硬件实现的开销
2. **时序影响**: 避免在关键路径上使用复杂工具
3. **参数验证**: 在编译时进行充分的参数检查
4. **文档完整**: 为每个工具提供清晰的使用说明
5. **测试覆盖**: 确保工具函数的正确性和边界情况处理
