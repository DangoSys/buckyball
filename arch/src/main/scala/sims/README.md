# Simulation Configurations

This directory contains simulation configurations and interfaces for various simulators, providing unified configuration management for different simulation environments.

## Directory Structure

```
sims/
├── firesim/
│   └── TargetConfigs.scala    - FireSim FPGA simulation configuration
└── verilator/
    └── TargetConfig.scala     - Verilator simulation top-level generation
```

## Verilator Simulation (verilator/)

### TargetConfig.scala

Top-level generator for Verilator simulation:

```scala
object Elaborate extends App {
  if (args.isEmpty) {
    println("Usage: Elaborate <full.config.ClassName> [firtool-opts...]")
    sys.exit(1)
  }

  val configClass = Class.forName(args(0))
  val config = configClass.getDeclaredConstructor().newInstance().asInstanceOf[Config]

  ChiselStage.emitSystemVerilogFile(
    new BBSimHarness()(config.toInstance),
    firtoolOpts = args.drop(1),
    args = Array.empty
  )
}
```

**Generation Flow**:
1. Parse command line arguments and configuration
2. Instantiate Buckyball system module
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
class FireSimBuckyballConfig extends Config(
  new WithDefaultFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new BuckyballConfig
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

## Build and Usage

### Verilator Simulation Build

```bash
# Generate Verilog
cd arch
mill arch.runMain sims.verilator.Elaborate sims.verilator.BuckyballToyVerilatorConfig

# Build simulator (in sims/verilator directory)
cd ../../sims/verilator
make CONFIG=ToyBuckyball
```

**Available Verilator Configurations**:
- `sims.verilator.BuckyballToyVerilatorConfig`
- `sims.verilator.BuckyballGobanVerilatorConfig`
- `sims.verilator.BuckyballKonbiVerilatorConfig`
- `sims.verilator.BuckyballPolyVerilatorConfig`

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
  new BuckyballConfig
)
```

## Related Documentation

- [Architecture Overview](../README.md)
- [Verilator Workflow](../../../../workflow/steps/verilator/README.md)
- [Test Framework](../../../../bb-tests/README.md)
