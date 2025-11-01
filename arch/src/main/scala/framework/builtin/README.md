# BuckyBall Built-in Component Library

## Overview

This directory contains the built-in hardware component implementations of the BuckyBall framework, providing standardized and reusable hardware modules. Located at `arch/src/main/scala/framework/builtin`, it serves as the component library layer, offering verified hardware building blocks for upper-level systems.

Main component modules:
- **memdomain**: Memory domain components, including storage and DMA engines
- **frontend**: Frontend processing components for instruction decode and scheduling
- **util**: Framework-level utility functions
- **BaseConfigs.scala**: Base configuration definitions

## Code Structure

```
builtin/
├── BaseConfigs.scala - Base configuration parameter definitions
├── memdomain/        - Memory domain implementation
│   ├── dma/          - DMA engines (BBStreamReader/Writer)
│   ├── mem/          - Memory components (Scratchpad, Accumulator)
│   ├── rs/           - Memory domain reservation station
│   ├── tlb/          - TLB implementation
│   ├── MemController.scala  - Memory controller
│   ├── MemDomain.scala      - Memory domain top-level
│   ├── MemLoader.scala      - Load instruction handler
│   ├── MemStorer.scala      - Store instruction handler
│   └── DomainDecoder.scala  - Memory domain decoder
├── frontend/         - Frontend components
│   ├── GobalDecoder.scala   - Global instruction decoder
│   ├── globalrs/            - Global reservation station
│   │   ├── GlobalReservationStation.scala
│   │   └── GlobalROB.scala  - Global reorder buffer
│   └── rs/                  - Ball domain reservation station
│       ├── reservationStation.scala
│       └── rob.scala
└── util/             - Utility function library
```

### Module Dependencies

```
Configuration Layer (BaseConfigs.scala)
    ↓
Component Layer (memdomain, frontend, util)
    ↓
Application Layer (examples, prototypes)
```

**BaseConfigs.scala** (Configuration Base Layer)
- Defines base configuration parameters for all built-in components
- Provides default configuration and parameter validation
- Referenced by all sub-modules as configuration source

**memdomain/** (Memory Subsystem)
- Depends on BaseConfigs for memory-related configuration
- Implements storage, DMA, address management, etc.
- Provides memory access services for other components

**frontend/** (Frontend Processing)
- Uses frontend configuration parameters from BaseConfigs
- Implements instruction fetch, decode, and scheduling
- Tightly integrated with processor core

**util/** (Utility Library)
- Provides common hardware design utility functions
- Widely used by other components
- Independent of specific configuration parameters

## Module Details

### BaseConfigs.scala

**Main Function**: Define base configuration parameters and defaults for built-in components

**Key Components**:

```scala
case class BaseConfig(
  opcodes: OpcodeSet = OpcodeSet.custom3,
  
  inputType: Data,                      // Input data type
  accType: Data,                        // Accumulator data type
  
  veclane: Int = 16,                    // Vector lane width
  accveclane: Int = 4,                  // Accumulator vector lane
  
  tlb_size: Int = 4,                    // TLB size
  rob_entries: Int = 16,                // Number of ROB entries
  rs_out_of_order_response: Boolean = true,  // Out-of-order response support
  
  dma_maxbytes: Int = 64,               // Unused
  dma_buswidth: Int = 128,              // DMA bus width
  
  sp_banks: Int = 4,                    // Scratchpad bank count
  acc_banks: Int = 8,                   // Accumulator bank count
  
  sp_capacity: BuckyBallMemCapacity = CapacityInKilobytes(256),
  acc_capacity: BuckyBallMemCapacity = CapacityInKilobytes(64),
  
  spAddrLen: Int = 15,                  // SPAD address length
  memAddrLen: Int = 32,                 // Memory address length
  
  numVecPE: Int = 16,                   // Vector PEs per thread
  numVecThread: Int = 16,               // Vector threads
  
  emptyBallid: Int = 5                  // Empty ball ID
)
```

**Configuration Parameters**:
- **Memory Domain**: Bank counts, capacities, address lengths
- **Frontend**: ROB entries, out-of-order response
- **Vector Unit**: PE count, thread count, lane width
- **Data Types**: Input and accumulator data types

**Parameter Validation**:
```scala
require(sp_banks > 0, "SP banks must be positive")
require(acc_banks > 0, "ACC banks must be positive")
require(rob_entries > 0 && isPow2(rob_entries), "ROB entries must be power of 2")
```

**Input/Output**:
- Input: User-defined configuration overrides
- Output: Validated complete configuration parameters
- Edge cases: Parameter conflict detection and error reporting

### memdomain/ Submodule

**Main Function**: Implement complete memory domain functionality

**Key Components**:
- **MemDomain.scala**: Memory domain top-level module
  - Integrates all memory subsystem components
  - Provides unified external interface
  - Manages DMA and Ball domain access coordination

- **MemController.scala**: Memory controller
  - Encapsulates Scratchpad and Accumulator
  - Dual-port design for DMA and Ball domain
  - Bank arbitration and routing

- **MemLoader.scala**: Load instruction handler
  - Receives load instructions from reservation station
  - Issues DMA read requests
  - Writes data to Scratchpad/Accumulator

- **MemStorer.scala**: Store instruction handler
  - Reads data from Scratchpad/Accumulator
  - Issues DMA write requests with alignment
  - Handles byte masking

- **dma/**: DMA engines
  - **BBStreamReader**: Streaming read with TLB support
  - **BBStreamWriter**: Streaming write with alignment
  - Transaction ID management

- **mem/**: Memory components
  - **Scratchpad**: Multi-bank scratchpad memory
  - **AccBank**: Accumulator bank with accumulation pipeline
  - **SramBank**: Generic SRAM bank

- **tlb/**: Translation Lookaside Buffer
  - Virtual to physical address translation
  - Integrated with DMA engines

- **rs/**: Memory domain reservation station
  - FIFO-based instruction scheduler
  - Local ROB for memory instructions

**Interface Definition**:
```scala
class MemDomainIO(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  // From Global RS
  val issue = Flipped(Decoupled(new MemRsIssue))
  
  // To Global RS
  val complete = Decoupled(new MemRsComplete)
  
  // Ball domain SRAM interface
  val sramRead = Vec(sp_banks, Flipped(new SramReadIO))
  val sramWrite = Vec(sp_banks, Flipped(new SramWriteIO))
  val accRead = Vec(acc_banks, Flipped(new SramReadIO))
  val accWrite = Vec(acc_banks, Flipped(new SramWriteIO))
  
  // DMA interface
  val dma = new Bundle {
    val read = Decoupled(new BBReadRequest)
    val write = Decoupled(new BBWriteRequest)
  }
  
  // TLB interface
  val tlb = Vec(2, new BBTLBIO)
  val ptw = Vec(2, Flipped(new TLBPTWIO))
  val tlbExp = Output(Vec(2, new BBTLBExceptionIO))
}
```

### frontend/ Submodule

**Main Function**: Implement processor frontend functionality for instruction decode and scheduling

**Core Components**:
- **GobalDecoder.scala**: Global instruction decoder
  - Classifies instructions into Ball/Memory/Fence types
  - Constructs PostGDCmd for domain-specific decoders
  - Interfaces with Global RS

- **globalrs/**: Global reservation station
  - **GlobalReservationStation.scala**: Central instruction manager
    - Allocates ROB entries
    - Issues to Ball and Memory domains
    - Handles completion from both domains
    - Manages Fence synchronization
  - **GlobalROB.scala**: Global reorder buffer
    - Tracks instruction state across domains
    - Supports out-of-order completion
    - Sequential commit

- **rs/**: Ball domain reservation station
  - **reservationStation.scala**: Ball-specific scheduler
  - **rob.scala**: Local ROB for Ball instructions

**Data Flow**:
```
RoCC → Global Decoder → Global RS → Ball Domain / Mem Domain
                          ↓                ↓            ↓
                      Global ROB    Ball Decoder  Mem Decoder
                   (tracks state)       ↓            ↓
                                   Ball Devices  Loader/Storer
```

### util/ Submodule

**Main Function**: Provide common utility functions

**Utility Categories**:
- Mathematical operation tools
- Interface conversion tools
- Debug and monitoring tools
- Common hardware patterns

## Usage Guide

### Configuration Usage

**Basic Configuration Inheritance**:
```scala
class MySystemConfig extends Config(
  new BaseConfig ++
  new WithCustomMemDomain(spBanks = 8) ++
  new WithCustomFrontend(robEntries = 32)
)
```

**Parameter Access**:
```scala
class MyModule(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val spBanks = b.sp_banks
  val accBanks = b.acc_banks
  val robEntries = b.rob_entries
}
```

### Extension Development

**Adding New Components**:
1. Create new module in corresponding subdirectory
2. Add configuration parameters in BaseConfigs.scala
3. Implement standard Module interface
4. Add corresponding test cases
5. Update documentation

**Custom Configuration**:
```scala
case class MyComponentParams(
  param1: Int = 16,
  param2: Boolean = true
)

class WithMyComponent(param1: Int, param2: Boolean) extends Config((site, here, up) => {
  case MyComponentKey => MyComponentParams(param1, param2)
})
```

### Important Notes

1. **Configuration Consistency**: Ensure related component configurations are compatible
2. **Resource Constraints**: Pay attention to reasonable hardware resource allocation
3. **Timing Optimization**: Focus on timing paths across components
4. **Interface Standards**: Follow unified interface design specifications
5. **Test Coverage**: Provide sufficient test cases for each component
6. **Memory Access**: Respect bank access constraints (op1 and op2 cannot access same bank)
7. **ROB Management**: Coordinate between Global ROB and local ROBs

## Architecture Highlights

### Instruction Pipeline
```
RoCC Interface
    ↓
Global Decoder (classify instruction type)
    ↓
Global RS (with ROB) ← tracks all in-flight instructions
    ↓           ↓
Ball Domain  Mem Domain
    ↓           ↓
Ball Devices  Loader/Storer
    ↓           ↓
    MemController
    ↓           ↓
Scratchpad  Accumulator
```

### Memory Architecture
```
Main Memory
    ↓ (DMA + TLB)
MemController
├─→ Scratchpad (4 banks × 64KB = 256KB)
└─→ Accumulator (8 banks × 8KB = 64KB)
    ↑
Ball Devices (read/write access)
```

## Performance Considerations

1. **ROB Depth**: 16 entries support up to 16 in-flight instructions
2. **Memory Banks**: 4 scratchpad + 8 accumulator banks enable parallel access
3. **Out-of-Order Execution**: Global RS supports OOO when enabled
4. **DMA Bandwidth**: 128-bit bus provides high memory throughput
5. **Pipeline Depth**: Multi-stage pipeline for high clock frequency

## Related Documentation

- [Memory Domain Details](memdomain/README.md) - Memory subsystem implementation
- [Frontend Components](frontend/README.md) - Instruction decode and scheduling
- [DMA Engines](memdomain/dma/README.md) - DMA implementation
- [TLB Management](memdomain/tlb/README.md) - Address translation
- [Framework Overview](../README.md) - Upper-level architecture
