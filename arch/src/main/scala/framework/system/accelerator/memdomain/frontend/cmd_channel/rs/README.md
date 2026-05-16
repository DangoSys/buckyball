# Reservation Station (RS) Module

## Overview

Reservation station module for memory domain instruction scheduling and out-of-order execution management, located at `framework/builtin/memdomain/rs`. Includes reservation station and reorder buffer (ROB) implementation.

## File Structure

```
rs/
├── reservationStation.scala  - Memory reservation station
├── rob.scala                 - Reorder buffer
└── ringFifo.scala           - Ring FIFO (unused)
```

## MemReservationStation

### Interface

```scala
class MemReservationStation(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val mem_decode_cmd_i = Flipped(Decoupled(new MemDecodeCmd))
    val rs_rocc_o = new Bundle {
      val resp  = Decoupled(new RoCCResponseBB)
      val busy  = Output(Bool())
    }
    val issue_o     = new MemIssueInterface
    val commit_i    = new MemCommitInterface
  })
}
```

### Issue Interface

```scala
class MemIssueInterface(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  val ld = Decoupled(new MemRsIssue)    // Load instruction issue
  val st = Decoupled(new MemRsIssue)    // Store instruction issue
}
```

### Commit Interface

```scala
class MemCommitInterface(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  val ld = Flipped(Decoupled(new MemRsComplete))    // Load completion
  val st = Flipped(Decoupled(new MemRsComplete))    // Store completion
}
```

## ROB - Reorder Buffer

### Interface

```scala
class ROB (implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val alloc = Flipped(Decoupled(new MemDecodeCmd))
    val issue = Decoupled(new RobEntry)
    val complete = Flipped(Decoupled(UInt(log2Up(b.rob_entries).W)))
    val empty = Output(Bool())
    val full  = Output(Bool())
  })
}
```

### ROB Entry

```scala
class RobEntry(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  val cmd    = new MemDecodeCmd                    // Memory instruction
  val rob_id = UInt(log2Up(b.rob_entries).W)      // ROB ID
}
```

### Core Data Structures

```scala
val robFifo = Module(new Queue(new RobEntry, b.rob_entries))
val robIdCounter = RegInit(0.U(log2Up(b.rob_entries).W))
val robTable = Reg(Vec(b.rob_entries, Bool()))
```

- `robFifo`: FIFO queue maintaining instruction order
- `robIdCounter`: ROB ID counter
- `robTable`: Completion status table tracking instruction completion

## Workflow

### Instruction Allocation

1. Receive `MemDecodeCmd` from memory domain decoder
2. Allocate unique ROB ID
3. Enqueue instruction with ROB ID into ROB FIFO
4. Mark as incomplete in completion status table

```scala
robFifo.io.enq.valid       := io.alloc.valid
robFifo.io.enq.bits.cmd    := io.alloc.bits
robFifo.io.enq.bits.rob_id := robIdCounter

when(io.alloc.fire) {
  robIdCounter := robIdCounter + 1.U
  robTable(robIdCounter) := false.B
}
```

### Instruction Issue

1. Check head instruction in ROB
2. Separate by type (load/store)
3. Issue only when execution unit ready

```scala
io.issue_o.ld.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.is_load
io.issue_o.st.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.is_store

rob.io.issue.ready  := (rob.io.issue.bits.cmd.is_load && io.issue_o.ld.ready) ||
                       (rob.io.issue.bits.cmd.is_store && io.issue_o.st.ready)
```

### Instruction Completion

1. Arbiter handles multiple completion signals
2. Update completion status table
3. Support out-of-order completion

```scala
val completeArb = Module(new Arbiter(UInt(log2Up(b.rob_entries).W), 2))
completeArb.io.in(0).valid  := io.commit_i.ld.valid
completeArb.io.in(1).valid  := io.commit_i.st.valid

when(io.complete.fire) {
  robTable(io.complete.bits) := true.B
}
```

## Configuration

ROB configuration through `CustomBuckyballConfig`:

```scala
class CustomBuckyballConfig extends Config((site, here, up) => {
  case "rob_entries" => 16    // Number of ROB entries
})
```

## Execution Model

### In-Order Issue, Out-of-Order Completion

- **In-Order Issue**: Instructions issued from ROB head in program order
- **Out-of-Order Completion**: Instructions can complete out-of-order, tracked by ROB ID
- **In-Order Commit**: Instructions commit in program order (simplified in current implementation)

### Memory Consistency

- **Load/Store Separation**: Load and store instructions handled separately
- **Dependency Checking**: ROB maintains memory access ordering
- **Exception Handling**: Supports memory access exception handling

## Related Modules

- [Memory Domain](../README.md)
- [Memory Controller](../mem/README.md)
- [DMA Engines](../dma/README.md)
