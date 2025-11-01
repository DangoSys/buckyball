# Reservation Station & ROB

## Overview

This module implements the Reservation Station and Reorder Buffer (ROB) in the BuckyBall system for out-of-order execution and instruction scheduling support. The reservation station manages instruction issue and completion, while ROB ensures instructions commit in program order, maintaining precise exception semantics.

## File Structure

```
rs/
├── reservationStation.scala  - Reservation station implementation
└── rob.scala                - Reorder buffer implementation
```

## Core Components

### BallReservationStation - Ball Domain Reservation Station

The reservation station is a key component connecting the instruction decoder and execution units, responsible for:

**Main functionality**:
- Receives instructions from Ball domain decoder
- Dispatches to different execution units based on instruction type
- Manages instruction issue and completion status
- Generates RoCC responses

**Supported execution units**:
- **ball1**: VecUnit (vector processing unit)
- **ball2**: BBFP (floating-point processing unit)
- **ball3**: im2col (image processing accelerator)
- **ball4**: transpose (matrix transpose accelerator)

**Interface design**:
```scala
class BallReservationStation extends Module {
  val io = IO(new Bundle {
    // Instruction input
    val ball_decode_cmd_i = Flipped(DecoupledIO(new BallDecodeCmd))

    // RoCC response output
    val rs_rocc_o = new Bundle {
      val resp = DecoupledIO(new RoCCResponseBB)
      val busy = Output(Bool())
    }

    // Execution unit interfaces
    val issue_o = new BallIssueInterface    // Issue interface
    val commit_i = new BallCommitInterface  // Commit interface
  })
}
```

**Instruction dispatch logic**:
```scala
// Dispatch instructions based on bid (Ball ID)
io.issue_o.ball1.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.bid === 1.U  // VecUnit
io.issue_o.ball2.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.bid === 2.U  // BBFP
io.issue_o.ball3.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.bid === 3.U  // im2col
io.issue_o.ball4.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.bid === 4.U  // transpose
```

### ROB - Reorder Buffer

ROB implements sequential instruction management and out-of-order completion support:

**Design features**:
- Uses FIFO queue to maintain instruction order
- Uses completion status table to track instruction execution status
- Supports out-of-order completion but in-order issue
- Provides ROB ID for instruction identification

**Core data structures**:
```scala
class RobEntry extends Bundle {
  val cmd = new BallDecodeCmd           // Instruction content
  val rob_id = UInt(log2Up(rob_entries).W)  // ROB identifier
}
```

**State management**:
```scala
val robFifo = Module(new Queue(new RobEntry, rob_entries))  // Instruction queue
val robTable = Reg(Vec(rob_entries, Bool()))               // Completion status table
val robIdCounter = RegInit(0.U(log2Up(rob_entries).W))     // ID counter
```

## Workflow

### Instruction Allocation Flow
1. **Instruction enqueue**: Instructions from decoder enter ROB
2. **Assign ROB ID**: Allocate unique ROB ID to each instruction
3. **State initialization**: Mark as incomplete in completion status table

```scala
when(io.alloc.fire) {
  robIdCounter := robIdCounter + 1.U
  robTable(robIdCounter) := false.B  // Mark as incomplete
}
```

### Instruction Issue Flow
1. **Head check**: Check if ROB head instruction is incomplete
2. **Type dispatch**: Dispatch instruction to corresponding execution unit based on bid
3. **Ready control**: Only issue when target execution unit is ready

```scala
val headEntry = robFifo.io.deq.bits
val headCompleted = robTable(headEntry.rob_id)
io.issue.valid := robFifo.io.deq.valid && !headCompleted
```

### Instruction Completion Flow
1. **Completion arbitration**: Multiple execution unit completion signals handled by arbiter
2. **State update**: Update completion status table based on ROB ID
3. **Queue dequeue**: Remove completed head instruction from ROB

```scala
val completeArb = Module(new Arbiter(UInt(log2Up(rob_entries).W), 4))
when(io.complete.fire) {
  robTable(io.complete.bits) := true.B  // Mark as completed
}
```

## Configuration Parameters

### Key Configuration Items
- **rob_entries**: ROB entry count, affects out-of-order execution window size
- **Execution unit count**: Currently supports 4 Ball execution units
- **Arbitration strategy**: Uses round-robin arbitration for multiple completion signals

### Performance Considerations
- **ROB size**: Larger ROB supports more out-of-order execution but increases hardware overhead
- **Issue bandwidth**: Currently maximum one instruction issued per cycle
- **Completion bandwidth**: Supports multiple instruction completions per cycle

## Interface Protocol

### BallIssueInterface - Issue Interface
```scala
class BallIssueInterface extends Bundle {
  val ball1 = Decoupled(new BallRsIssue)  // VecUnit issue
  val ball2 = Decoupled(new BallRsIssue)  // BBFP issue
  val ball3 = Decoupled(new BallRsIssue)  // im2col issue
  val ball4 = Decoupled(new BallRsIssue)  // transpose issue
}
```

### BallCommitInterface - Commit Interface
```scala
class BallCommitInterface extends Bundle {
  val ball1 = Flipped(Decoupled(new BallRsComplete))  // VecUnit commit
  val ball2 = Flipped(Decoupled(new BallRsComplete))  // BBFP commit
  val ball3 = Flipped(Decoupled(new BallRsComplete))  // im2col commit
  val ball4 = Flipped(Decoupled(new BallRsComplete))  // transpose commit
}
```

## Usage Examples

### Basic Configuration
```scala
// Configure ROB size in CustomBuckyBallConfig
class MyBuckyBallConfig extends CustomBuckyBallConfig {
  override val rob_entries = 16  // 16-entry ROB
}

// Instantiate reservation station
val reservationStation = Module(new BallReservationStation)
```

### Connecting Execution Units
```scala
// Connect VecUnit
vecUnit.io.cmd <> reservationStation.io.issue_o.ball1
reservationStation.io.commit_i.ball1 <> vecUnit.io.resp

// Connect BBFP
bbfp.io.cmd <> reservationStation.io.issue_o.ball2
reservationStation.io.commit_i.ball2 <> bbfp.io.resp
```

## Debug and Monitoring

### Status Signals
- **io.rs_rocc_o.busy**: Reservation station busy status
- **rob.io.empty**: ROB empty status
- **rob.io.full**: ROB full status

### Performance Counters
The following performance counters can be added for monitoring:
- Instruction issue count
- Instruction completion count
- ROB utilization
- Load distribution across execution units

## Extension Guide

### Adding New Execution Units
1. Add new issue port in `BallIssueInterface`
2. Add corresponding commit port in `BallCommitInterface`
3. Add corresponding dispatch and arbitration logic in reservation station
4. Update completion signal arbiter port count

### Optimization Suggestions
- **Multi-issue support**: Can be extended to issue multiple instructions per cycle
- **Dynamic scheduling**: Implement more complex scheduling algorithms
- **Load balancing**: Perform load balancing across multiple execution units of the same type

## Related Documentation

- [Ball Domain Overview](../README.md)
- [Ball Domain Bus](../bbus/README.md)
- [Image Processing Accelerator](../im2col/README.md)
- [Vector Processing Unit](../../../prototype/vector/README.md)
