# Memory Domain (MemDomain)

## Overview

MemDomain is the memory subsystem of BuckyBall, located at `framework/builtin/memdomain`. It handles memory management and data transfer between main memory and on-chip storage (scratchpad and accumulator).

## File Structure

```
memdomain/
├── MemDomain.scala         - Top-level memory domain module
├── MemController.scala     - Memory controller
├── MemLoader.scala         - Load instruction handler
├── MemStorer.scala         - Store instruction handler
├── DomainDecoder.scala     - Memory instruction decoder
├── DISA.scala             - Address space definitions
├── dma/                   - DMA engines
├── mem/                   - Memory components
├── rs/                    - Reservation station
└── tlb/                   - TLB implementation
```

## Core Components

### MemDomain

Top-level module that integrates all memory subsystem components.

**Interfaces**:
```scala
class MemDomain extends LazyModule {
  val io = IO(new Bundle {
    // From Global RS
    val issue = Flipped(Decoupled(new MemRsIssue))
    
    // To Global RS  
    val complete = Decoupled(new MemRsComplete)
    
    // Ball domain SRAM access
    val sramRead = Vec(sp_banks, Flipped(new SramReadIO))
    val sramWrite = Vec(sp_banks, Flipped(new SramWriteIO))
    val accRead = Vec(acc_banks, Flipped(new SramReadIO))
    val accWrite = Vec(acc_banks, Flipped(new SramWriteIO))
    
    // DMA
    val dma = new DMABundle
    
    // TLB
    val tlb = Vec(2, new BBTLBIO)
    val ptw = Vec(2, Flipped(new TLBPTWIO))
    val tlbExp = Output(Vec(2, new BBTLBExceptionIO))
  })
}
```

### MemController

Encapsulates Scratchpad and Accumulator, provides interfaces for DMA and Ball domain.

**Features**:
- Dual-port design: DMA port and Ball domain port
- Bank arbitration for parallel access
- Unified interface for Scratchpad and Accumulator

**Arbitration**: 
- Read: Ball domain (execution) has higher priority than DMA (main)
- Write: Ball domain has higher priority than DMA

### MemLoader

Handles load instructions (bb_mvin).

**Operation Flow**:
1. Receive load instruction from reservation station
2. Issue DMA read request to main memory
3. Write received data to Scratchpad/Accumulator
4. Report completion to reservation station

**Key Logic**:
- Caches instruction parameters (mem_addr, spad_addr, iter, bank info)
- Tracks response count (supports up to 16 responses)
- Streams data to SRAM as responses arrive

### MemStorer

Handles store instructions (bb_mvout).

**Operation Flow**:
1. Receive store instruction from reservation station
2. Read data from Scratchpad/Accumulator
3. Issue DMA write requests (with alignment handling)
4. Report completion after all data sent

**Key Logic**:
- 16-byte buffer for data alignment
- Handles unaligned addresses by merging data
- Generates byte masks for partial writes

### Memory Architecture

**Scratchpad**:
- Purpose: Store input data and intermediate results
- Configuration: `b.sp_banks` banks, `b.spad_bank_entries` entries per bank
- Width: `b.spad_w` bits
- Total capacity: Default 256KB (4 banks × 64KB)

**Accumulator**:
- Purpose: Store accumulation results and final outputs
- Configuration: `b.acc_banks` banks, `b.acc_bank_entries` entries per bank
- Width: `b.acc_w` bits
- Total capacity: Default 64KB (8 banks × 8KB)

## Data Flow

```
Main Memory ←→ DMA Engine ←→ MemController ←→ Scratchpad/Accumulator
                                    ↕
                              Ball Devices
```

**Dual-Port Access**:
1. DMA port: For data transfer with main memory
2. Ball domain port: For computation access by accelerators

## Submodules

### dma/ - DMA Engines

**BBStreamReader**: Streaming data reader
- TileLink interface for memory access
- TLB support for virtual addressing
- Transaction ID management for multiple outstanding requests

**BBStreamWriter**: Streaming data writer  
- Handles data alignment (16-byte aligned)
- Generates byte masks for partial writes
- TLB support

### mem/ - Memory Components

**Scratchpad**: Multi-bank on-chip memory
- Single-port per bank (read or write, not simultaneous)
- Bank arbitration between DMA and Ball domain

**AccBank**: Accumulator bank
- Includes AccPipe for accumulation operations
- AccReadRouter for read request arbitration

**SramBank**: Generic SRAM bank implementation

### tlb/ - TLB

**BBTLBCluster**: TLB cluster manager
- Virtual to physical address translation
- TLB miss handling
- Exception signaling

### rs/ - Reservation Station

**MemReservationStation**: Local instruction scheduler for memory operations
- FIFO-based scheduling
- Separate issue paths for load and store
- Forwards completion signals to Global RS

**RingFifo**: Circular FIFO buffer implementation

## Configuration

```scala
case class BaseConfig(
  sp_banks: Int = 4,              // Scratchpad banks
  sp_capacity: CapacityInKilobytes(256),
  acc_banks: Int = 8,             // Accumulator banks  
  acc_capacity: CapacityInKilobytes(64),
  dma_buswidth: Int = 128,        // DMA bus width
  spAddrLen: Int = 15,            // Address length
  memAddrLen: Int = 32
)
```

## Usage

```scala
// Instantiate memory domain
implicit val config = new CustomBuckyBallConfig
val memDomain = LazyModule(new MemDomain)

// Connect to Global RS
memDomain.module.io.issue <> globalRS.io.memIssue
globalRS.io.memComplete <> memDomain.module.io.complete

// Connect to Ball domain
ballDevices.io.sramRead <> memDomain.module.io.sramRead
ballDevices.io.sramWrite <> memDomain.module.io.sramWrite
```

## Important Notes

1. **Bank Access Constraint**: op1 and op2 of Ball instructions cannot access the same bank simultaneously
2. **Address Alignment**: DMA writes handle 16-byte alignment automatically
3. **Single Port**: Each SRAM bank has only one port, simultaneous read/write to same address not supported
4. **Response Ordering**: DMA responses may arrive out of order, use addrcounter for ordering

## Related Documentation

- [DMA Implementation](dma/README.md)
- [Memory Components](mem/README.md)
- [TLB](tlb/README.md)
- [Reservation Station](rs/README.md)
