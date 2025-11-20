# BuckyBall Framework Core

## Overview

This directory contains the core implementation of the BuckyBall framework, serving as the foundation layer for the entire hardware architecture. Located at `arch/src/main/scala/framework`, it provides a complete implementation of processor cores, built-in components, and system interconnects.

Main functional modules include:
- **builtin**: Built-in hardware component library, including memory domain and frontend modules
- **blink**: System interconnect and communication framework

## Code Structure

```
framework/
├── builtin/          - Built-in component library
│   ├── memdomain/    - Memory domain implementation
│   │   ├── dma/      - DMA engines (BBStreamReader/Writer)
│   │   ├── mem/      - Memory components (Scratchpad, Accumulator, SRAM banks)
│   │   ├── rs/       - Memory domain reservation station
│   │   ├── tlb/      - TLB implementation
│   │   ├── MemController.scala  - Memory controller
│   │   ├── MemDomain.scala      - Memory domain top-level
│   │   ├── MemLoader.scala      - Load instruction handler
│   │   └── MemStorer.scala      - Store instruction handler
│   ├── frontend/     - Frontend components
│   │   ├── GobalDecoder.scala   - Global instruction decoder
│   │   ├── globalrs/            - Global reservation station
│   │   │   ├── GlobalReservationStation.scala
│   │   │   └── GlobalROB.scala  - Global reorder buffer
│   │   └── rs/                  - Ball domain reservation station
│   ├── util/         - Framework utility functions
│   └── BaseConfigs.scala - Base configuration parameters
└── blink/            - System interconnect framework
    ├── baseball.scala    - Ball device base trait
    ├── blink.scala       - Blink protocol definitions
    └── bbus.scala        - Ball bus implementation
```

### Module Dependencies

```
Application Layer → builtin components → blink interconnect → Physical interface
                        ↓                    ↓
                   Memory domain        Ball protocol
                   Frontend             System bus
```

## Module Details

### builtin/ - Built-in Component Library

**Main Function**: Provides standardized hardware component implementations

**Component Categories**:

#### memdomain/ - Memory Domain
The memory domain encapsulates all memory-related functionality:

**Key Components**:
- **MemDomain.scala**: Top-level memory domain module
  - Integrates MemController, MemLoader, MemStorer, and TLB
  - Provides unified interface to Global RS
  - Handles both load and store operations

- **MemController.scala**: Memory controller
  - Encapsulates Scratchpad and Accumulator
  - Provides DMA and Ball Domain interfaces
  - Handles bank arbitration and routing

- **MemLoader.scala**: Load instruction handler
  - Receives load instructions from reservation station
  - Issues DMA read requests
  - Writes data to Scratchpad/Accumulator

- **MemStorer.scala**: Store instruction handler
  - Receives store instructions from reservation station
  - Reads data from Scratchpad/Accumulator
  - Issues DMA write requests with data alignment and masking

- **dma/**: DMA engines
  - **BBStreamReader**: Streaming DMA read with TLB support
  - **BBStreamWriter**: Streaming DMA write with alignment handling
  - Transaction ID management for multiple outstanding requests

- **mem/**: Memory components
  - **Scratchpad.scala**: 4-bank scratchpad memory (256KB total)
  - **AccBank.scala**: Accumulator bank with accumulation pipeline
  - **SramBank.scala**: Generic single-port SRAM bank implementation

- **rs/**: Memory domain reservation station
  - **reservationStation.scala**: Local FIFO-based scheduler
  - **rob.scala**: Local reorder buffer for memory instructions
  - **ringFifo.scala**: Circular FIFO implementation

- **tlb/**: Translation Lookaside Buffer
  - Virtual to physical address translation
  - Integrated with DMA engines

#### frontend/ - Frontend Components
The frontend handles global instruction management:

**Key Components**:
- **GobalDecoder.scala**: Global instruction decoder
  - Classifies instructions into Ball/Memory/Fence types
  - Constructs PostGDCmd for domain-specific decoders
  - Interfaces with Global RS

- **globalrs/**: Global reservation station
  - **GlobalReservationStation.scala**: Central instruction manager
    - Allocates ROB entries
    - Issues instructions to Ball and Memory domains
    - Handles instruction completion from both domains
    - Manages Fence instruction synchronization
  - **GlobalROB.scala**: Global reorder buffer
    - Tracks instruction state across domains
    - Supports out-of-order completion
    - Sequential commit of completed instructions

- **rs/**: Ball domain reservation station
  - **reservationStation.scala**: Ball-specific scheduler
  - **rob.scala**: Local ROB for Ball instructions

#### util/ - Framework Utilities
Common utility functions and helper modules

#### BaseConfigs.scala
**Configuration Parameters**:
```scala
case class BaseConfig(
  veclane: Int = 16,              // Vector lane width
  accveclane: Int = 4,            // Accumulator vector lane width
  rob_entries: Int = 16,          // Number of ROB entries
  rs_out_of_order_response: Boolean = true,  // Out-of-order response support
  sp_banks: Int = 4,              // Scratchpad bank count
  acc_banks: Int = 8,             // Accumulator bank count
  sp_capacity: BuckyBallMemCapacity = CapacityInKilobytes(256),
  acc_capacity: BuckyBallMemCapacity = CapacityInKilobytes(64),
  spAddrLen: Int = 15,            // SPAD address length
  memAddrLen: Int = 32,           // Memory address length
  numVecPE: Int = 16,             // Vector PEs per thread
  numVecThread: Int = 16,         // Vector threads
  emptyBallid: Int = 5            // Empty ball ID
)
```

### blink/ - System Interconnect

**Main Function**: Implements system-level interconnect and Ball protocol

**Key Components**:
- **baseball.scala**: Ball device base trait
  - Defines `BallRegist` trait for Ball device registration
  - Provides common interface for all Ball devices

- **blink.scala**: Blink protocol definitions
  - Command/response interfaces
  - Status and control signals
  - SRAM read/write interfaces

- **bbus.scala**: Ball bus implementation (BBus)
  - Manages multiple Ball device connections
  - Command router: Routes commands to appropriate Ball devices
  - Bus router: Arbitrates Ball device responses
  - Memory router: Handles memory access arbitration
  - Performance monitoring counters

**Interconnect Features**:
- Support for multiple bus protocols
- Arbitration and routing functionality
- Latency and bandwidth management
- Dynamic Ball device registration

## Usage Guide

### Framework Integration

**Configuration System**:
```scala
class BuckyBallConfig extends Config(
  new WithBuiltinComponents ++
  new WithBlinkInterconnect ++
  new BaseConfig
)
```

**Module Instantiation**:
```scala
class BuckyBallSystem(implicit p: Parameters) extends LazyModule {
  // Memory domain
  val memdomain = Module(new MemDomain)

  // Ball domain
  val balldomain = Module(new BallDomain)

  // Global RS
  val globalRS = Module(new GlobalReservationStation)

  // Connect modules
  balldomain.io.issue <> globalRS.io.ballIssue
  memdomain.io.issue <> globalRS.io.memIssue
  globalRS.io.ballComplete <> balldomain.io.complete
  globalRS.io.memComplete <> memdomain.io.complete
}
```

### Extension Development

**Adding New Components**:
1. Create new component module in builtin directory
2. Implement standard Module interface
3. Register in configuration system
4. Update interconnect and routing logic

**Custom Ball Device**:
1. Extend `BallRegist` trait
2. Implement Blink protocol interfaces
3. Register in BBus
4. Add to Ball RS device list

### Design Principles

1. **Parameter Passing**: Use Chipyard's Parameters system for configuration
2. **Clock Domains**: Pay attention to clock domain crossing between modules
3. **Reset Strategy**: Ensure proper reset sequencing and dependencies
4. **Performance Optimization**: Focus on critical paths and timing constraints
5. **Debug Support**: Integrate necessary debug and monitoring interfaces
6. **Memory Access**: Respect bank access constraints (op1 and op2 cannot access same bank)
7. **Handshake Protocols**: Use ready/valid handshake for all data transfers

## Architecture Highlights

### Instruction Flow
```
RoCC → Global Decoder → Global RS → Ball Domain / Mem Domain
                          ↓                ↓            ↓
                      Global ROB    Ball Decoder  Mem Decoder
                   (tracks state)       ↓            ↓
                                   Ball Devices  Loader/Storer
                                        ↓            ↓
                                   MemController ← → MemController
```

### Memory Access Flow
```
Ball Devices ──→ MemController ──→ Scratchpad (4 banks)
                      │           └→ Accumulator (8 banks)
                      │
Mem Domain    ──→ MemController
  (Loader/Storer)     │
                      ↓
                  DMA + TLB
                      ↓
                 Main Memory
```

## Related Documentation

- [Blink Interconnect System](blink/README.md) - System interconnect implementation
- [Built-in Components](builtin/README.md) - Standard hardware components
- [Memory Domain](builtin/memdomain/README.md) - Memory subsystem details
- [Frontend Components](builtin/frontend/README.md) - Instruction management
- [BuckyBall Source Overview](../README.md) - Upper-level architecture

## Performance Considerations

1. **ROB Size**: 16 entries support up to 16 in-flight instructions
2. **Bank Parallelism**: 4 scratchpad + 8 accumulator banks enable parallel access
3. **Out-of-Order Execution**: Global RS supports out-of-order completion when enabled
4. **DMA Bandwidth**: 128-bit bus width provides high memory bandwidth
5. **Pipeline Depth**: Multi-stage pipeline allows high clock frequency

## Common Issues and Solutions

**Issue**: Instructions stall in Global RS
- **Solution**: Check ROB capacity and completion signals from domains

**Issue**: Memory access conflicts
- **Solution**: Ensure op1 and op2 don't access same bank, respect bank boundaries

**Issue**: DMA timeout
- **Solution**: Verify TLB configuration and page table walker connectivity

**Issue**: Ball device not responding
- **Solution**: Check Ball device registration in BBus and RS device list
