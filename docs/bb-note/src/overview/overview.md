# Buckyball Project Structure Overview

Buckyball is a scalable framework for domain-specific architectures. The project adopts a modular design with clear directory responsibilities, supporting a complete toolchain from hardware design to software development.

## Main Directory Structure

### Core Architecture Module
- **`arch/`** - Hardware architecture implementation, containing RTL code written in Scala/Chisel
  - Based on Rocket-chip and Chipyard framework
  - Implements custom RoCC coprocessors and memory subsystems
  - Supports various configuration and extension options

### Test Verification Module
- **`bb-tests/`** - Unified test framework
  - `workloads/` - Application workload tests
  - `customext/` - Custom extension verification
  - `sardine/` - Sardine test framework
  - `uvbb/` - Unit test suite

### Simulation Environment Module
- **`sims/`** - Simulators and verification environments
  - Supports Verilator, VCS and other simulators
  - Integrates FireSim FPGA accelerated simulation
  - Provides performance analysis and debugging tools

### Development Tools Module
- **`scripts/`** - Build and deployment scripts
  - Environment initialization scripts
  - Automated build tools
  - Dependency management and configuration

- **`workflow/`** - Development workflows and automation
  - CI/CD pipeline configuration
  - Documentation generation tools
  - Code quality checks

### Documentation System
- **`docs/`** - Project documentation
  - `bb-note/` - Technical documentation based on mdBook
  - `img/` - Documentation image resources
  - Supports automatic generation and updates

### Third-party Dependencies
- **`thirdparty/`** - External dependency modules (**submodules**)
  - `chipyard/` - Berkeley Chipyard SoC design framework
  - `circt/` - CIRCT circuit compiler toolchain

## Development Workflow

1. **Environment Setup**: Use `scripts/init.sh` to initialize the development environment
2. **Architecture Development**: Perform hardware design and modifications in the `arch/` directory
3. **Test Verification**: Use test suites in `bb-tests/` for functional verification
4. **Simulation Debugging**: Perform performance analysis through simulation environments in the `sims/` directory
5. **Documentation Updates**: Automatically generate or manually update technical documentation in `docs/`

## Build System

The project supports multiple build methods:
- **Make**: Traditional Makefile builds
- **SBT**: Scala project build tool
- **CMake**: Test framework build system
- **Conda**: Python environment and dependency management

## Version Management Notes

- **Submodules**: Modules under `thirdparty/` need independent updates
- **Main Repository**: Core code and configuration update synchronously with the main branch
- **Documentation**: Supports automatic generation, keeping in sync with code changes
