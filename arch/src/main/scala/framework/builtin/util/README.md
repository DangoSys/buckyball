# Framework Utility Library

## Overview

This directory contains framework-level utility functions and helper modules for BuckyBall, providing general-purpose hardware design tools. Located in `arch/src/main/scala/framework/builtin/util`, it serves as a utility layer, offering reusable hardware building blocks and utility functions for other framework components.

Main utility categories:
- Mathematical operations and bit manipulation tools
- Interface conversion and adapters
- Debug and performance monitoring tools
- Common hardware pattern implementations

## Code Structure

```
util/
└── (specific utility files to be analyzed)
```

### Utility Categories

**Math Tools**
- Bit width calculation and logarithm functions
- Numeric conversion and formatting
- Optimized arithmetic operation implementations

**Interface Tools**
- Protocol conversion adapters
- Signal synchronization and clock domain crossing
- Handshake protocol implementations

**Debug Tools**
- Performance counter templates
- Debug signal output
- State monitoring interfaces

## Module Description

### Mathematical Operations Tools

**Main functionality**: Provides common mathematical operations and bit manipulation functions

**Key functions**:

```scala
object MathUtils {
  // Calculate log2 ceiling
  def log2Ceil(x: Int): Int = {
    require(x > 0)
    (log(x) / log(2)).ceil.toInt
  }

  // Check if power of 2
  def isPow2(x: Int): Boolean = x > 0 && (x & (x - 1)) == 0

  // Calculate smallest power of 2 >= x
  def nextPow2(x: Int): Int = {
    if (isPow2(x)) x else 1 << log2Ceil(x)
  }
}
```

**Bit manipulation tools**:
```scala
object BitUtils {
  // Bit reversal
  def reverseBits(data: UInt, width: Int): UInt = {
    VecInit((0 until width).map(i => data(i))).asUInt
  }

  // Hamming weight (count of 1s)
  def popCount(data: UInt): UInt = {
    PopCount(data)
  }

  // Leading zero count
  def leadingZeros(data: UInt, width: Int): UInt = {
    PriorityEncoder(Reverse(data))
  }
}
```

### Interface Conversion Tools

**Main functionality**: Provides common interface conversion and adaptation

**Protocol converters**:
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

**Clock domain crossing**:
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

  // Async FIFO implementation
  // Uses Gray code pointers to avoid metastability
}
```

### Debug and Monitoring Tools

**Main functionality**: Provides general debugging and performance monitoring tools

**Performance counter**:
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

  // Optional debug output
  when(io.inc) {
    printf(s"[PerfCounter] $name: %d\n", counter + 1.U)
  }
}
```

**Debug signal output**:
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

## Usage

### Usage Examples

**Using math tools**:
```scala
import util.MathUtils._

class MyModule extends Module {
  val addrBits = log2Ceil(entries)
  val bankSize = nextPow2(requestedSize)

  require(isPow2(bankSize), "Bank size must be power of 2")
}
```

**Using interface conversion**:
```scala
val converter = Module(new DecoupledToValid(UInt(32.W)))
converter.io.in <> some_decoupled_signal
val valid_signal = converter.io.out
```

**Using performance monitoring**:
```scala
val hit_counter = Module(new PerfCounter("cache_hits"))
hit_counter.io.inc := cache_hit

val miss_counter = Module(new PerfCounter("cache_misses"))
miss_counter.io.inc := cache_miss
```

### Extension Development

**Adding new tools**:
1. Determine utility's generality and reuse value
2. Implement standard Chisel module interfaces
3. Add sufficient parameterization support
4. Provide usage examples and test cases

**Tool design principles**:
- Keep interfaces concise and clear
- Support parameterized configuration
- Provide good error checking
- Consider hardware implementation efficiency

### Notes

1. **Hardware overhead**: Utility functions should consider hardware implementation costs
2. **Timing impact**: Avoid using complex tools on critical paths
3. **Parameter validation**: Perform sufficient parameter checking at compile time
4. **Complete documentation**: Provide clear usage instructions for each tool
5. **Test coverage**: Ensure correctness and boundary case handling of utility functions
