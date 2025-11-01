# BuckyBall Scala Source Code

This directory contains all Scala/Chisel hardware description language source code for the BuckyBall project, implementing hardware architecture design and simulation environments.

## Overview

BuckyBall uses Scala/Chisel as the hardware description language, built on Berkeley's Rocket-chip and Chipyard frameworks. This directory contains implementations from low-level hardware components to system-level integration.

Main functional modules include:
- **framework**: Core framework implementation, including processor core, memory subsystem, bus interconnect, etc.
- **prototype**: Prototype implementation of dedicated accelerators
- **examples**: Example configurations and reference designs
- **sims**: Simulation environment configurations and interfaces
- **Util**: General utility classes and helper functions

## Code Structure

```
scala/
├── framework/          - BuckyBall core framework
│   ├── blink/          - Blink communication components
│   ├── builtin/        - Built-in hardware components
│   │   ├── frontend/   - Frontend processing components
│   │   ├── memdomain/  - Memory domain implementation
│   │   └── util/       - Framework utility classes
│   └── rocket/         - Rocket core extensions
├── prototype/          - Dedicated accelerator prototypes
│   ├── format/         - Data format processing
│   ├── im2col/         - Image processing acceleration
│   ├── matrix/         - Matrix computation engine
│   ├── transpose/      - Matrix transpose acceleration
│   └── vector/         - Vector processing unit
├── examples/           - Examples and configurations
│   └── toy/            - Toy example system
├── sims/               - Simulation configurations
│   ├── firesim/        - FireSim FPGA simulation
│   └── verilator/      - Verilator simulation
└── Util/               - General utility classes
```

## Module Description

### framework/ - Core Framework
Implements BuckyBall's core architecture components, including:
- Processor core and extensions
- Memory subsystem and cache hierarchy
- Bus interconnect and communication protocols
- System configuration and parameterization mechanisms

### prototype/ - Accelerator Prototypes
Contains hardware implementations of dedicated computation accelerators:
- Machine learning accelerators (matrix operations, convolution, etc.)
- Data processing accelerators (format conversion, transpose, etc.)
- Vector processing units (SIMD, multi-threading, etc.)

### examples/ - Example Configurations
Provides system configuration examples and reference designs:
- Basic configuration templates
- Custom extension examples
- Integration test cases

### sims/ - Simulation Environment
Supports multiple simulators and verification environments:
- Verilator simulation
- FireSim FPGA simulation
- Performance analysis and debugging tools

## Development Guide

### Build System
BuckyBall uses Mill as the build tool:
```bash
# Compile all modules
mill arch.compile

# Generate Verilog
mill arch.runMain examples.toy.ToyBuckyBall

# Run tests
mill arch.test
```

### Code Standards
- Follow Scala and Chisel coding conventions
- Use ScalaFmt for code formatting
- Each module includes documentation and tests
- Configuration parameterization uses Chipyard Config system

### Extension Development
1. **Add new accelerator**: Create new module in prototype/ directory
2. **Modify framework**: Extend existing components in framework/ directory
3. **Add configuration**: Create new configuration files in examples/ directory
4. **Integration testing**: Use simulation environments in sims/ directory for verification

## Related Documentation

- [Core Framework Documentation](framework/README.md)
- [Accelerator Prototype Documentation](prototype/README.md)
- [Example Configuration Documentation](examples/README.md)
- [Simulation Environment Documentation](sims/README.md)
- [Utility Classes Documentation](Util/README.md)
