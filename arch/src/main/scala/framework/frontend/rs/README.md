# Ball Domain Reservation Station

## Core Features

General Ball domain reservation station implementation supporting:
- ✅ Circular ROB (ring buffer)
- ✅ In-order issue, max half of ROB depth instructions
- ✅ Out-of-order completion and commit
- ✅ Configurable in-order/out-of-order response mode
- ✅ Dynamic Ball device count support

## File Structure

```
rs/
├── reservationStation.scala  - Main RS module, connects ROB and Ball devices
├── rob.scala                 - Circular ROB implementation
└── README.md                 - This document
```

## Architecture Design

### Overall Pipeline

```
Ball Decoder → ROB Alloc → ROB Issue → Ball Execute → Complete → ROB Commit
     ↓                                                      ↑
  Immediate resp (configurable)                  In-order/OoO filter (configurable)
```

### Module Responsibilities

**Reservation Station (BallReservationStation)**:
- Receives Ball domain decoded instructions
- Manages multiple Ball devices' issue and completion
- Decides response strategy based on configuration (in-order/out-of-order)
- Filters non-head completion signals in in-order mode

**ROB (Reorder Buffer)**:
- Circular queue management (circular rob_id)
- In-order issue, limits inflight count
- Out-of-order commit (commits all completed instructions per cycle)
- Exposes internal state for reservation station decisions

## ROB Implementation Details

### Circular Queue Structure

```scala
// Core state
val robEntries   = Reg(Vec(b.rob_entries, new RobEntry))  // Instruction storage
val robValid     = Reg(Vec(b.rob_entries, Bool()))         // Entry valid
val robIssued    = Reg(Vec(b.rob_entries, Bool()))         // Issued
val robComplete  = Reg(Vec(b.rob_entries, Bool()))         // Completed

// Circular queue pointers
val headPtr      = Reg(UInt())  // Oldest uncommitted instruction
val tailPtr      = Reg(UInt())  // Next allocation position
val robIdCounter = Reg(UInt())  // ROB ID circular counter (0 ~ rob_entries-1)

// Issue limit
val issuedCount  = Reg(UInt())  // Issued but not completed instruction count
val maxIssueLimit = (b.rob_entries / 2).U  // Max issue half
```

### ROB ID Circulation

```scala
// Circular increment on allocation
when(io.alloc.fire) {
  robIdCounter := Mux(robIdCounter === (b.rob_entries - 1).U,
                      0.U, robIdCounter + 1.U)
}
```

### In-order Issue Logic

Scan from head pointer to find the first non-issued instruction:

```scala
// Scan all positions
for (i <- 0 until b.rob_entries) {
  val ptr = (headPtr + i.U) % b.rob_entries.U
  scanValid(i) := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
}

// Priority encoder finds the first
val issuePtr = PriorityEncoder(scanValid)

// Check issue limit
val canIssue = scanValid.orR && (issuedCount < maxIssueLimit)
```

### Out-of-order Commit Logic

Commit all completed instructions per cycle, then update head pointer:

```scala
// Commit all completed instructions
for (i <- 0 until b.rob_entries) {
  when(robValid(i.U) && robComplete(i.U)) {
    robValid(i.U) := false.B
    robIssued(i.U) := false.B
    robComplete(i.U) := false.B
  }
}

// Head pointer skips all committed positions, moves to first valid uncompleted instruction
```

### Exposed State Signals

```scala
io.empty          // ROB empty
io.full           // ROB full
io.head_ptr       // Head pointer position
io.issued_count   // Inflight instruction count
io.entry_valid    // Each entry valid
io.entry_complete // Each entry completed
```

## Reservation Station Implementation Details

### Issue Logic

Dispatch to corresponding Ball device based on instruction `bid` (Ball ID):

```scala
for (i <- 0 until numBalls) {
  val ballId = BallRsRegists(i).ballId.U
  io.issue_o.balls(i).valid := rob.io.issue.valid &&
                               rob.io.issue.bits.cmd.bid === ballId
  io.issue_o.balls(i).bits  := rob.io.issue.bits
}

// ROB ready: only issue when target Ball device is ready
rob.io.issue.ready := VecInit(
  BallRsRegists.zipWithIndex.map { case (info, idx) =>
    (rob.io.issue.bits.cmd.bid === info.ballId.U) &&
    io.issue_o.balls(idx).ready
  }
).asUInt.orR
```

### Completion Signal Handling (Critical)

**Out-of-order mode**: Accept all completion signals

```scala
if (b.rs_out_of_order_response) {
  rob.io.complete <> completeArb.io.out
}
```

**In-order mode**: Only accept completion signals where `rob_id == head_ptr`

```scala
else {
  val isHeadComplete = completeArb.io.out.bits === rob.io.head_ptr
  rob.io.complete.valid := completeArb.io.out.valid && isHeadComplete
  rob.io.complete.bits  := completeArb.io.out.bits
  // Non-head instructions are blocked and wait
  completeArb.io.out.ready := rob.io.complete.ready && isHeadComplete
}
```

## Configuration Parameters

### BaseConfigs.scala

```scala
case class CustomBuckyballConfig(
  rob_entries: Int = 16,                      // ROB entry count
  rs_out_of_order_response: Boolean = true,   // Out-of-order response mode
  ...
)
```

### Parameter Description

| Parameter | Default | Description |
|------|--------|------|
| `rob_entries` | 16 | ROB depth, affects out-of-order window size |
| `rs_out_of_order_response` | true | true=out-of-order response, false=in-order response |

## Usage Examples

### Registering Ball Devices

```scala
val ballDevices = Seq(
  BallRsRegist(ballId = 0, ballName = "VectorUnit"),
  BallRsRegist(ballId = 1, ballName = "MatrixUnit"),
  BallRsRegist(ballId = 2, ballName = "LoadUnit"),
  BallRsRegist(ballId = 3, ballName = "StoreUnit")
)

val rs = Module(new BallReservationStation(ballDevices))
```

### Connecting Interfaces

```scala
// Input: from Ball domain decoder
rs.io.ball_decode_cmd_i <> decoder.io.ball_cmd_o

// Output: to each Ball device
rs.io.issue_o.balls(0) <> vectorUnit.io.cmd_i
rs.io.issue_o.balls(1) <> matrixUnit.io.cmd_i
rs.io.issue_o.balls(2) <> loadUnit.io.cmd_i
rs.io.issue_o.balls(3) <> storeUnit.io.cmd_i

// Completion signals: from each Ball device
rs.io.commit_i.balls(0) <> vectorUnit.io.complete_o
rs.io.commit_i.balls(1) <> matrixUnit.io.complete_o
rs.io.commit_i.balls(2) <> loadUnit.io.complete_o
rs.io.commit_i.balls(3) <> storeUnit.io.complete_o

// RoCC response
rocc.resp <> rs.io.rs_rocc_o.resp
rocc.busy := rs.io.rs_rocc_o.busy
```

## Performance Characteristics

### Out-of-order Mode (rs_out_of_order_response = true)

**Advantages**:
- ✅ High throughput
- ✅ Ball devices not blocked
- ✅ Full utilization of ROB capacity
- ✅ Suitable for high-performance scenarios

**Disadvantages**:
- ❌ No strict instruction ordering guarantee
- ❌ Difficult to debug

**Use cases**:
- Independent Ball computation tasks
- Batch operations without data dependencies
- Maximum throughput required

### In-order Mode (rs_out_of_order_response = false)

**Advantages**:
- ✅ Strict program order commit
- ✅ Predictable behavior
- ✅ Easy to debug
- ✅ Supports precise exceptions

**Disadvantages**:
- ❌ Lower throughput
- ❌ Ball devices may be blocked (waiting for head completion)
- ❌ ROB utilization may be low

**Use cases**:
- Operation sequences with data dependencies
- Debug and verification
- Strict ordering requirements

### Impact of Issue Limit

```
Max inflight count = rob_entries / 2
```

**With ROB=16**:
- Max 8 instructions issued simultaneously
- Remaining 8 positions for buffering new instructions
- Balance issue pressure and buffering capacity

## Performance Tuning Recommendations

### Increase ROB Depth

```scala
override val rob_entries = 32  // Increase to 32
```
- ✅ Larger out-of-order window
- ✅ More instructions can execute in parallel
- ❌ Increased area and power

### Adjust Issue Limit

To modify issue ratio, edit `rob.scala`:

```scala
val maxIssueLimit = (b.rob_entries * 3 / 4).U  // Change to 3/4
```

### Hybrid Mode (Future Extension)

Reservation station can use signals like `rob.io.entry_complete` for more complex strategies:

```scala
// Example: dynamically adjust based on completion status
val completedRatio = PopCount(rob.io.entry_complete) / rob_entries.U
val allowResponse = (completedRatio > threshold.U) || rob.io.empty
```

## Timing Diagrams

### Out-of-order Mode Execution Flow

```
Cycle | Action               | headPtr | issuedCount | ROB State
------|----------------------|---------|-------------|------------------
1     | Alloc instr 0        | 0       | 0           | [0:not issued]
2     | Issue instr 0        | 0       | 1           | [0:issued]
3     | Alloc+issue instr 1  | 0       | 2           | [0:issued, 1:issued]
4     | Instr 1 completes    | 0       | 1           | [0:issued, 1:completed]
5     | Instr 1 commits      | 0       | 1           | [0:issued, 1:empty]
6     | Instr 0 completes    | 0       | 0           | [0:completed, 1:empty]
7     | Instr 0 commits      | 2       | 0           | [0:empty, 1:empty]
```

### In-order Mode Execution Flow

```
Cycle | Action               | headPtr | Complete signal  | Commit
------|----------------------|---------|------------------|----------
1     | Alloc instr 0        | 0       | -                | -
2     | Issue instr 0        | 0       | -                | -
3     | Alloc+issue instr 1  | 0       | -                | -
4     | Instr 1 completes    | 0       | rob_id=1 ❌block | -
5     | Instr 1 waits        | 0       | rob_id=1 ❌block | -
6     | Instr 0 completes    | 0       | rob_id=0 ✅accept| Instr 0
7     | Head moves           | 1       | -                | -
8     | Instr 1 retries      | 1       | rob_id=1 ✅accept| Instr 1
```

## Debug Techniques

### View ROB State

```scala
when(rob.io.alloc.fire) {
  printf("Alloc: rob_id=%d, bid=%d\n",
    rob.io.alloc.bits.rob_id, rob.io.alloc.bits.cmd.bid)
}

when(rob.io.issue.fire) {
  printf("Issue: rob_id=%d, head=%d, issued_count=%d\n",
    rob.io.issue.bits.rob_id, rob.io.head_ptr, rob.io.issued_count)
}

when(rob.io.complete.fire) {
  printf("Complete: rob_id=%d, head=%d\n",
    rob.io.complete.bits, rob.io.head_ptr)
}
```

### Common Issue Troubleshooting

**Issue 1: ROB always full**
- Check if Ball devices complete normally
- Check if completion signals are connected correctly
- In in-order mode, check if head instruction is stuck

**Issue 2: Instructions not issued**
- Check if `issued_count` reaches limit
- Check Ball device ready signals
- Check if bid matches registered Ball devices

**Issue 3: Low performance in in-order mode**
- Consider switching to out-of-order mode
- Increase ROB depth
- Optimize Ball device execution latency

## Related Documentation

- [Framework Overview](../../../README.md)
- [Ball Domain Implementation Example](../../../../examples/toy/balldomain/)
- [BaseConfigs Configuration Guide](../../BaseConfigs.scala)

## Design Tradeoffs

| Design Choice | Reason |
|---------|------|
| Fixed out-of-order ROB commit | Simplifies ROB logic, improves performance |
| RS controls in-order/out-of-order | Flexible strategy, easy to extend |
| Issue limit = depth/2 | Balance parallelism and buffering |
| Expose ROB internal state | Support complex scheduling strategies |
| Circular queue | ROB ID recyclable, supports long-running |
