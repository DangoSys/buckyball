# Buckyball Architecture Design Overview

The Buckyball architecture module contains complete hardware design implementations, based on the RISC-V instruction set architecture, developed using the Scala/Chisel hardware description language. The architecture design follows modular and extensible principles, supporting various configurations and custom extensions.

## Architecture Hierarchy

### System-Level Architecture
Buckyball adopts a layered design, including from top to bottom:
- **SoC Subsystem**: Integrates multi-core processors, cache hierarchy, interconnect networks
- **Processor Core**: Custom implementation based on Rocket core
- **Coprocessor**: Dedicated accelerators supporting RoCC interface
- **Memory Subsystem**: High-performance memory controllers and DMA engines

### Core Features
- **Configurability**: Supports parameter configuration for core count, cache size, bus width, etc.
- **Extensibility**: Provides standardized coprocessor interfaces and extension mechanisms
- **Compatibility**: Maintains compatibility with the standard RISC-V ecosystem
- **Performance Optimization**: Performance-optimized design for specific application scenarios

## Directory Structure

```
arch/
├── src/main/scala/
│   └── framework/          - Buckyball framework core
│       ├── rocket/         - Rocket core custom implementation
│       └── builtin/        - Built-in component library
│           └── memdomain/  - Memory domain implementation
│               ├── mem/    - Memory components
│               └── dma/    - DMA engine
└── thirdparty/            - Third-party dependencies
    └── chipyard/          - Chipyard framework
```

## Design Principles

### Modular Design
Each functional module has clear interface definitions and independent implementations, facilitating testing, verification, and reuse. Modules communicate through standardized interfaces, reducing coupling.

### Parameterized Configuration
All hardware modules support parameterized configuration, achieving flexible hardware generation through Scala's type system and configuration framework. Configuration parameters include:
- Data path width
- Cache size and organization
- Parallelism and pipeline depth
- Coprocessor types and quantities

### Performance Optimization
Specialized performance optimizations for target application scenarios:
- Memory access pattern optimization
- Data pipeline design
- Parallel computing support
- Low-latency communication mechanisms

## Development Workflow

1. **Requirement Analysis**: Determine performance and functional requirements for target applications
2. **Architecture Design**: Select appropriate configuration parameters and extension modules
3. **RTL Implementation**: Use Chisel for hardware description and implementation
4. **Functional Verification**: Verify functional correctness through unit tests and integration tests
5. **Performance Evaluation**: Use simulators and FPGA for performance analysis and optimization

## Toolchain Support

- **Chisel/FIRRTL**: Hardware description and synthesis toolchain
- **Verilator**: Fast simulation and verification
- **VCS**: Commercial-grade simulation tools
- **FireSim**: FPGA accelerated simulation platform
- **Chipyard**: Integrated development environment and toolchain
