# BuckyBall Example Configurations

## Overview

This directory contains example configurations and reference implementations of the BuckyBall framework, demonstrating how to configure and extend BuckyBall systems. Located at `arch/src/main/scala/examples`, it serves as the configuration layer, providing configuration templates and system instances for developers.

Main components include:
- **BuckyBallConfig.scala**: Global configuration parameter definitions
- **toy/**: Complete example system implementation with custom coprocessor and CSR extensions

## Code Structure

```
examples/
├── BuckyBallConfig.scala     - Global configuration definitions
└── toy/                      - Complete example system
    ├── balldomain/           - Ball domain component implementation
    │   ├── BallDomain.scala  - Ball domain top-level
    │   ├── bbus/             - Ball bus registration
    │   │   └── busRegister.scala
    │   ├── rs/               - Ball RS registration
    │   │   └── rsRegister.scala
    │   └── decoder/          - Ball decoder (if exists)
    ├── CustomConfigs.scala   - System configuration composition
    └── ToyBuckyBall.scala    - System top-level module
```

### File Dependencies

**BuckyBallConfig.scala** (Base Configuration Layer)
- Defines global configuration parameters and defaults
- Inherited and extended by all other configuration files
- Provides system-level configuration interface

**toy/CustomConfigs.scala** (Configuration Composition Layer)
- Inherits from BuckyBallConfig and adds custom parameters
- Composes multiple configuration fragments into complete configuration
- Provides configuration support for ToyBuckyBall

**toy/ToyBuckyBall.scala** (System Instantiation Layer)
- Uses CustomConfigs to instantiate complete system
- Serves as entry point for Mill build
- Generates final Verilog code

## Module Details

### BuckyBallConfig.scala

**Main Function**: Define global configuration parameters for the BuckyBall framework

**Key Components**:

```scala
object BuckyBallConfigs {
  val defaultConfig = BaseConfig
  val toyConfig = BuckyBallToyConfig.defaultConfig
  
  // Actually used configuration
  val customConfig = toyConfig
  
  type CustomBuckyBallConfig = BaseConfig
}
```

**Configuration Selection**:
The framework uses `customConfig` to select the active configuration. This allows easy switching between different system configurations.

**Input/Output**:
- Input: No direct input, parameters passed through configuration system
- Output: Configuration parameters for use by other modules
- Edge cases: Configuration conflicts resolved by priority-based overriding

### toy/ - Example System

The toy system demonstrates a complete BuckyBall implementation with various Ball devices.

#### toy/ToyBuckyBall.scala

**Main Function**: System top-level module, instantiates complete toy system

**Key Components**:

```scala
class ToyBuckyBall(implicit b: CustomBuckyBallConfig, p: Parameters) extends LazyRoCC {
  override lazy val module = new ToyBuckyBallModuleImp(this)
}

class ToyBuckyBallModuleImp(outer: ToyBuckyBall) extends LazyRoCCModuleImp(outer) {
  // Global Decoder
  val globalDecoder = Module(new GlobalDecoder)
  
  // Global Reservation Station (with ROB)
  val globalRS = Module(new GlobalReservationStation)
  
  // Ball Domain (regular Module, not LazyModule)
  val ballDomain = Module(new BallDomain)
  
  // Memory Domain (complete domain with DMA+TLB+SRAM)
  val memDomain = LazyModule(new MemDomain)
  
  // Connect components
  globalDecoder.io.rocc <> io.cmd
  globalRS.io.decode <> globalDecoder.io.issue
  ballDomain.io.issue <> globalRS.io.ballIssue
  memDomain.module.io.issue <> globalRS.io.memIssue
  // ... more connections
}
```

**Build Flow**:
1. Load configuration from BuckyBallConfig
2. Instantiate ToyBuckyBall LazyRoCC module
3. Generate Verilog through ChiselStage
4. Output to generated-src directory

**Input/Output**:
- Input: RoCC interface commands from Rocket core
- Output: RoCC interface responses, busy signals
- Edge cases: Configuration errors cause build failure

#### toy/balldomain/ - Ball Domain Components

**BallDomain.scala**: Ball domain top-level module
- Integrates Ball Decoder, local Ball RS, and BBus
- Provides single-channel interface to Global RS
- Routes commands to appropriate Ball devices

**bbus/busRegister.scala**: Ball bus registration
```scala
class BBusModule extends BBus {
  // Register Ball device generators
  registerBall(() => new VecBall, ballId = 0.U)
  registerBall(() => new MatrixBall, ballId = 1.U)
  registerBall(() => new TransposeBall, ballId = 2.U)
  registerBall(() => new Im2colBall, ballId = 3.U)
  registerBall(() => new ReluBall, ballId = 4.U)
}
```

**rs/rsRegister.scala**: Ball RS device registration
```scala
class BallRSModule extends BallReservationStation {
  // Register Ball device information
  registerBallInfo(name = "VecBall", bid = 0, latency = 10)
  registerBallInfo(name = "MatrixBall", bid = 1, latency = 20)
  registerBallInfo(name = "TransposeBall", bid = 2, latency = 15)
  registerBallInfo(name = "Im2colBall", bid = 3, latency = 15)
  registerBallInfo(name = "ReluBall", bid = 4, latency = 10)
}
```

#### toy/CustomConfigs.scala

**Main Function**: Compose multiple configuration fragments for the toy system

**Configuration Composition**:

```scala
object BuckyBallToyConfig {
  val defaultConfig = BaseConfig(
    opcodes = OpcodeSet.custom3,
    inputType = UInt(8.W),        // INT8 input
    accType = UInt(32.W),         // INT32 accumulator
    veclane = 16,                 // 16-element vectors
    accveclane = 4,               // 4-element accumulator vectors
    rob_entries = 16,             // 16 ROB entries
    sp_banks = 4,                 // 4 scratchpad banks
    acc_banks = 8,                // 8 accumulator banks
    sp_capacity = CapacityInKilobytes(256),   // 256KB scratchpad
    acc_capacity = CapacityInKilobytes(64),   // 64KB accumulator
    numVecPE = 16,                // 16 vector PEs
    numVecThread = 16             // 16 vector threads
  )
}
```

**Configuration Parameters**:
- **opcodes**: Custom instruction opcode set (custom3 = 0x7b)
- **inputType**: Data type for input operands
- **accType**: Data type for accumulator
- **veclane**: Number of elements per vector lane
- **rob_entries**: Reorder buffer depth
- **Memory configuration**: Bank counts and capacities
- **Vector configuration**: PE count and thread count

## Usage Guide

### Building the Toy System

**Generate Verilog**:
```bash
cd arch
mill arch.runMain examples.toy.ToyBuckyBall
```

**Generated Files**:
- Location: `arch/generated-src/toy/`
- Files: Verilog (.v), FIRRTL (.fir), annotation (.anno.json)

### Custom Configuration Development

**Steps**:
1. Copy `CustomConfigs.scala` as template
2. Modify configuration parameters to meet requirements
3. Implement necessary custom components
4. Update top-level module to reference new configuration
5. Register Ball devices in BBus and Ball RS

**Example: Adding New Ball Device**:

1. Implement Ball device:
```scala
class MyCustomBall(implicit b: CustomBuckyBallConfig, p: Parameters) 
  extends Module with BallRegist {
  // Implement Ball interfaces
  val io = IO(new BlinkIO)
  def ballId = 6.U  // Assign unique Ball ID
  // ... implementation
}
```

2. Register in BBusModule:
```scala
registerBall(() => new MyCustomBall, ballId = 6.U)
```

3. Register in BallRSModule:
```scala
registerBallInfo(name = "MyCustomBall", bid = 6, latency = 12)
```

### Configuration Best Practices

**Parameter Selection**:
1. **Memory Sizes**: Balance capacity vs. area
   - Scratchpad: Main working memory for data
   - Accumulator: Smaller, used for accumulation results

2. **ROB Depth**: Impacts instruction-level parallelism
   - Larger ROB: More in-flight instructions, higher parallelism
   - Smaller ROB: Lower area, simpler control logic

3. **Bank Counts**: Affects memory bandwidth
   - More banks: Higher parallel access bandwidth
   - Fewer banks: Simpler arbitration, lower area

4. **Vector Configuration**: Depends on workload
   - Vector lane width: Match data parallelism
   - PE/Thread count: Balance performance vs. area

**Common Configurations**:

```scala
// High-performance configuration
val highPerfConfig = BaseConfig(
  veclane = 32,                 // Wider vectors
  rob_entries = 32,             // Deeper ROB
  sp_banks = 8,                 // More banks
  sp_capacity = CapacityInKilobytes(512)
)

// Area-optimized configuration
val smallConfig = BaseConfig(
  veclane = 8,
  rob_entries = 8,
  sp_banks = 2,
  sp_capacity = CapacityInKilobytes(64)
)
```

### Important Notes

1. **Configuration Priority**: Later configurations in the chain override earlier ones with same parameter names
2. **Dependency Management**: Ensure custom component dependencies are correctly declared in configuration
3. **Build Path**: Generated file paths specified by TargetDirAnnotation
4. **Parameter Validation**: Configuration parameters validated during instantiation; invalid configurations cause build failure
5. **Ball ID Uniqueness**: Each Ball device must have unique ID across the system
6. **Bank Access Rules**: Remember op1 and op2 cannot access same bank simultaneously

## System Architecture

The toy system implements the complete BuckyBall architecture:

```
┌─────────────────────────────────────────────────────────┐
│                  Rocket Core (via RoCC)                 │
└────────────────────┬────────────────────────────────────┘
                     │
            ┌────────▼────────┐
            │ Global Decoder  │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │   Global RS     │
            │  (with ROB)     │
            └────┬──────┬─────┘
                 │      │
         ┌───────▼──┐ ┌▼──────────┐
         │  Ball    │ │   Mem     │
         │  Domain  │ │  Domain   │
         │          │ │           │
         │  ┌─────┐ │ │ ┌──────┐ │
         │  │BBus │ │ │ │ DMA  │ │
         │  └──┬──┘ │ │ │+TLB  │ │
         │     │    │ │ └───┬──┘ │
         │  ┌──▼───┐│ │     │    │
         │  │Balls ││ │  ┌──▼──┐ │
         │  └──────┘│ │  │Mem  │ │
         └──────┬───┘ │  │Ctrl │ │
                │     │  └─────┘ │
                │     └─────┬────┘
                │           │
            ┌───▼───────────▼───┐
            │  Memory Controller│
            │ (Scratchpad+Acc)  │
            └───────────────────┘
```

**Supported Ball Devices**:
- **VecBall** (ID=0): Vector operations
- **MatrixBall** (ID=1): Matrix multiplication (various formats)
- **TransposeBall** (ID=2): Matrix transpose
- **Im2colBall** (ID=3): Im2col transformation for convolution
- **ReluBall** (ID=4): ReLU activation function

## Related Documentation

- [Framework Overview](../framework/README.md) - Core framework architecture
- [Ball Domain Details](toy/balldomain/README.md) - Ball domain implementation
- [Prototype Ball Devices](../prototype/README.md) - Ball device implementations
- [Memory Domain](../framework/builtin/memdomain/README.md) - Memory subsystem
- [Simulation Guide](../sims/README.md) - Running simulations

## Troubleshooting

**Issue**: Build fails with "Ball ID conflict"
- **Solution**: Ensure each Ball device has unique ID in both BBus and RS registration

**Issue**: Generated Verilog has timing violations
- **Solution**: Reduce clock frequency or optimize critical paths

**Issue**: Simulation shows incorrect results
- **Solution**: Verify Ball device implementation and memory access patterns

**Issue**: Configuration parameter not taking effect
- **Solution**: Check configuration priority and ensure parameter is in correct config fragment
