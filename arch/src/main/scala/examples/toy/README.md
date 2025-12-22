# Toy Buckyball Example Implementation

## Overview

This directory contains a complete example implementation of the Buckyball framework, demonstrating how to build a custom coprocessor based on the RoCC interface. Located in `arch/src/main/scala/examples/toy`, it serves as a reference implementation for the Buckyball system, integrating global decoder, Ball domain, and memory domain.

Core components:
- **ToyBuckyball.scala**: Main RoCC coprocessor implementation
- **CustomConfigs.scala**: System configuration and RoCC integration
- **CSR.scala**: Custom control and status registers
- **balldomain/**: Ball domain related components

## Code Structure

```
toy/
├── ToyBuckyball.scala    - Main coprocessor implementation
├── CustomConfigs.scala   - Configuration definitions
├── CSR.scala            - CSR implementation
└── balldomain/          - Ball domain components
```

### File Dependencies

**ToyBuckyball.scala** (Core implementation layer)
- Extends LazyRoCCBB, implements RoCC coprocessor interface
- Integrates GlobalDecoder, BallDomain, MemDomain
- Manages TileLink connections and DMA components

**CustomConfigs.scala** (Configuration layer)
- Defines BuckyballCustomConfig and BuckyballToyConfig
- Configures RoCC integration and system parameters
- Provides multi-core configuration support

**CSR.scala** (Register layer)
- Implements FenceCSR control register
- Provides simple 64-bit register interface

## Module Description

### ToyBuckyball.scala

**Main functionality**: Implements complete Buckyball RoCC coprocessor

**Key components**:

```scala
class ToyBuckyball(val b: CustomBuckyballConfig)(implicit p: Parameters)
  extends LazyRoCCBB (opcodes = b.opcodes, nPTWPorts = 2) {

  val reader = LazyModule(new BBStreamReader(...))
  val writer = LazyModule(new BBStreamWriter(...))
  val xbar_node = TLXbar()
}
```

**System architecture**:
```scala
// Frontend: global decoder
val gDecoder = Module(new GlobalDecoder)

// Backend: Ball domain and memory domain
val ballDomain = Module(new BallDomain)
val memDomain = Module(new MemDomain)

// Response arbitration
val respArb = Module(new Arbiter(new RoCCResponseBB()(p), 2))
```

**TileLink connections**:
```scala
xbar_node := TLBuffer() := reader.node
xbar_node := TLBuffer() := writer.node
id_node := TLWidthWidget(b.dma_buswidth/8) := TLBuffer() := xbar_node
```

**Inputs/Outputs**:
- Input: RoCC command interface, PTW interface
- Output: RoCC response, TileLink memory access
- Edge cases: Busy-wait handling during Fence operations

### CustomConfigs.scala

**Main functionality**: Defines system configuration and RoCC integration

**Configuration class definition**:
```scala
class BuckyballCustomConfig(
  buckyballConfig: CustomBuckyballConfig = CustomBuckyballConfig()
) extends Config((site, here, up) => {
  case BuildRoCCBB => up(BuildRoCCBB) ++ Seq(
    (p: Parameters) => {
      val buckyball = LazyModule(new ToyBuckyball(buckyballConfig))
      buckyball
    }
  )
})
```

**System configuration**:
```scala
class BuckyballToyConfig extends Config(
  new framework.core.rocket.WithNBuckyballCores(1) ++
  new BuckyballCustomConfig(CustomBuckyballConfig()) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new WithCustomBootROM ++
  new chipyard.config.AbstractConfig
)
```

**Multi-core support**:
```scala
class WithMultiRoCCToyBuckyball(harts: Int*) extends Config(...)
```

### CSR.scala

**Main functionality**: Provides custom control and status registers

**Implementation**:
```scala
object FenceCSR {
  def apply(): UInt = RegInit(0.U(64.W))
}
```

**Fence handling logic**:
```scala
val fenceCSR = FenceCSR()
val fenceSet = ballDomain.io.fence_o
val allDomainsIdle = !ballDomain.io.busy && !memDomain.io.busy

when (fenceSet) {
  fenceCSR := 1.U
  io.cmd.ready := allDomainsIdle
}
```

## Usage

### System Integration

**RoCC interface integration**:
- Register coprocessor through BuildRoCCBB configuration key
- Support multi-core configuration
- Provide 2 PTW ports for address translation

**Inter-domain communication**:
```scala
// BallDomain -> MemDomain bridge
ballDomain.io.sramRead <> memDomain.io.ballDomain.sramRead
ballDomain.io.sramWrite <> memDomain.io.ballDomain.sramWrite
```

**DMA connections**:
```scala
memDomain.io.dma.read.req <> outer.reader.module.io.req
memDomain.io.dma.write.req <> outer.writer.module.io.req
```

### Notes

1. **Fence semantics**: Use CSR to implement Fence operation synchronization
2. **Busy-wait detection**: Assertion checks to prevent long simulation stalls
3. **TLB integration**: TLB functionality integrated in MemDomain
4. **Response arbitration**: BallDomain has higher priority than MemDomain
5. **Configuration dependencies**: Correctly configure CustomBuckyballConfig parameters
