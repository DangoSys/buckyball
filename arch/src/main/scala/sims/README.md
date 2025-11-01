# Simulation Configurations

This directory contains simulation configurations and interfaces for various simulators, providing unified configuration management for different simulation environments.

## Directory Structure

```
sims/
├── firesim/
│   └── TargetConfigs.scala    - FireSim FPGA simulation configuration
├── verilator/
│   └── Elaborate.scala        - Verilator simulation top-level generation
└── verify/
    └── TargetConfig.scala     - Verification configurations
```

## Verilator Simulation (verilator/)

### Elaborate.scala

Top-level generator for Verilator simulation:

```scala
object Elaborate extends App {
  // Select Ball type from command line arguments
  val ballType = args.headOption.getOrElse("toy")
  
  val config = ballType match {
    case "toy" => new ToyBuckyBallConfig
    case "vec" => new WithBlink(TargetBall.VecBall)
    case "matrix" => new WithBlink(TargetBall.MatrixBall)
    case "transpose" => new WithBlink(TargetBall.TransposeBall)
    case "im2col" => new WithBlink(TargetBall.Im2colBall)
    case "relu" => new WithBlink(TargetBall.ReluBall)
    case _ => new ToyBuckyBallConfig
  }
  
  val gen = () => LazyModule(new TestHarness()(config)).module
  
  (new ChiselStage).execute(
    args.tail,  // Remaining args passed to firtool
    Seq(
      ChiselGeneratorAnnotation(gen),
      TargetDirAnnotation("generated-src/verilator")
    )
  )
}
```

**Generation Flow**:
1. Parse command line arguments and configuration
2. Instantiate BuckyBall system module
3. Generate Verilog RTL code
4. Output auxiliary files for simulation

**Output Files**:
- `*.v` - Verilog files
- `*.anno.json` - FIRRTL annotation files
- `*.fir` - FIRRTL intermediate representation

## FireSim Simulation (firesim/)

### TargetConfigs.scala

Configurations for running on FireSim FPGA platform:

```scala
class FireSimBuckyBallConfig extends Config(
  new WithDefaultFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new BuckyBallConfig
)
```

**Key Configuration Items**:
- **Bridge Configuration**: UART, BlockDevice, NIC I/O bridges
- **Memory Model**: DDR3/DDR4 memory controller configuration
- **Clock Domains**: Multi-clock domain management
- **Debug Interface**: JTAG and Debug Module configuration

**Use Cases**:
- Large-scale system simulation
- Long-running workload testing
- Multi-core system performance evaluation
- I/O-intensive application verification

## Verification Configurations (verify/)

### TargetConfig.scala

Configurations for single Ball device verification:

```scala
sealed trait TargetBall
object TargetBall {
  case object VecBall extends TargetBall
  case object MatrixBall extends TargetBall
  case object TransposeBall extends TargetBall
  case object Im2colBall extends TargetBall
  case object ReluBall extends TargetBall
}
```

**WithBlink Configuration**: Empty configuration class for composing with Ball-specific configs

**Usage**:
```bash
# Verify specific Ball device
mill arch.runMain sims.verilator.Elaborate matrix
mill arch.runMain sims.verilator.Elaborate transpose
```

## Build and Usage

### Verilator Simulation Build

```bash
# Generate Verilog
cd arch
mill arch.runMain sims.verilator.Elaborate [ball_type]

# Build simulator (in sims/verilator directory)
cd ../../sims/verilator
make CONFIG=ToyBuckyBall
```

**Available Ball Types**:
- `toy`: Complete toy system (default)
- `vec`: Vector Ball only
- `matrix`: Matrix Ball only
- `transpose`: Transpose Ball only
- `im2col`: Im2col Ball only
- `relu`: ReLU Ball only

### FireSim Deployment

```bash
# Set up FireSim environment
cd firesim
source sourceme-f1-manager.sh

# Build FPGA bitstream
firesim buildbitstream

# Run simulation
firesim runworkload
```

## Debug and Optimization

### Verilator Debug

- **Waveform Generation**: Use `--trace` option to generate VCD files
- **Performance Profiling**: Use `--prof-cfuncs` for profiling
- **Coverage**: Use `--coverage` to generate coverage reports

### FireSim Debug

- **Printf Debugging**: Use `printf` statements for debug output
- **Assertion Checking**: Enable runtime assertion verification
- **Performance Counters**: Integrated HPM counters for monitoring

## Configuration Parameters

### Common Parameters

```scala
// Processor core configuration
case object RocketTilesKey extends Field[Seq[RocketTileParams]]

// Memory system configuration
case object MemoryBusKey extends Field[MemoryBusParams]

// Peripheral configuration
case object PeripheryBusKey extends Field[PeripheryBusParams]
```

### Simulation-Specific Parameters

```scala
// Verilator simulation parameters
case object VerilatorDRAMKey extends Field[Boolean](false)

// FireSim simulation parameters
case object FireSimBridgesKey extends Field[Seq[BridgeIOAnnotation]]
```

## Extension Development

### Adding New Simulator Support

1. Create new configuration directory (e.g., `vcs/`)
2. Implement simulator-specific configuration classes
3. Add build scripts and Makefiles
4. Update documentation and test cases

### Custom Configuration

```scala
class MyCustomConfig extends Config(
  new WithMyCustomParameters ++
  new BuckyBallConfig
)
```

## Related Documentation

- [Architecture Overview](../README.md)
- [Verilator Workflow](../../../../workflow/steps/verilator/README.md)
- [Test Framework](../../../../bb-tests/README.md)
