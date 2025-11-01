# Frontend Components

## Overview

Frontend components handle instruction decode and scheduling in BuckyBall. Located at `arch/src/main/scala/framework/builtin/frontend`.

## File Structure

```
frontend/
├── GobalDecoder.scala    - Global instruction decoder
└── globalrs/             - Global reservation station
    ├── GlobalReservationStation.scala
    └── GlobalROB.scala   - Global reorder buffer
```

Note: Ball domain reservation station (rs/) is separate and located alongside the frontend.

## Core Components

### GlobalDecoder

Decodes RoCC instructions and classifies them into different types.

**Interface**:
```scala
class GlobalDecoder(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val rocc = Flipped(Decoupled(new RoCCCommand))
    val issue = Decoupled(new PostGDCmd)
  })
}
```

**Instruction Classification**:
- `is_ball`: Ball instructions (computation operations)
- `is_mem`: Memory instructions (bb_mvin, bb_mvout)
- `is_fence`: Fence instructions

**Decode Logic**:
```scala
val func7 = io.rocc.bits.inst.funct
val is_mem_instr = (func7 === MVIN_BITPAT) || (func7 === MVOUT_BITPAT)
val is_fence_instr = (func7 === FENCE_BITPAT)
val is_ball_instr = !is_mem_instr && !is_fence_instr
```

### GlobalReservationStation

Central instruction manager between GlobalDecoder and execution domains.

**Interface**:
```scala
class GlobalReservationStation(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // From GlobalDecoder
    val decode = Flipped(Decoupled(new PostGDCmd))
    
    // To Ball Domain
    val ballIssue = Decoupled(new BallRsIssue)
    val ballComplete = Flipped(Decoupled(new BallRsComplete))
    
    // To Mem Domain
    val memIssue = Decoupled(new MemRsIssue)
    val memComplete = Flipped(Decoupled(new MemRsComplete))
    
    // RoCC response
    val resp = Decoupled(new RoCCResponse)
  })
}
```

**Key Features**:
- ROB-based instruction tracking
- Issues to Ball and Memory domains
- Handles completion from both domains
- Fence instruction synchronization (waits for ROB to be empty)
- Configurable out-of-order response support

**Instruction Flow**:
1. Allocate ROB entry for incoming instruction
2. Issue to appropriate domain (Ball or Memory)
3. Receive completion signal from domain
4. Update ROB entry status
5. Generate RoCC response (immediately or after ROB commit)

### GlobalROB

Reorder buffer for tracking instruction state.

**Structure**:
```scala
class ROBEntry extends Bundle {
  val valid = Bool()       // Entry is valid
  val issued = Bool()      // Instruction issued
  val completed = Bool()   // Instruction completed
  val rob_id = UInt()      // ROB ID
  val is_ball = Bool()     // Ball instruction
  val is_mem = Bool()      // Memory instruction
}
```

**Circular Buffer**:
- `head`: Points to oldest uncommitted instruction
- `tail`: Points to next allocation position
- Size: Configurable via `b.rob_entries` (default 16)

**Operation Modes**:
1. **Out-of-order mode** (`rs_out_of_order_response = true`):
   - Accept all completion signals
   - Commit completed instructions out-of-order
   - Respond immediately upon completion

2. **Sequential mode** (`rs_out_of_order_response = false`):
   - Only accept completion when `rob_id == head_ptr`
   - Commit in order
   - Respond after commit

**Issue Limit**: Maximum issue depth is half of ROB size to prevent excessive in-flight instructions.

## Instruction Pipeline

```
RoCC Interface
    ↓
GlobalDecoder (classify: Ball/Mem/Fence)
    ↓
Global RS (allocate ROB entry)
    ↓           ↓
Ball Domain  Mem Domain
    ↓           ↓
(execution)  (execution)
    ↓           ↓
Complete → Global RS → Update ROB → Respond
```

## Fence Handling

Fence instructions ensure all previous instructions complete before proceeding:

1. Fence instruction arrives at Global RS
2. Wait for ROB to become empty (all previous instructions completed)
3. Accept fence instruction (fire)
4. Immediately respond (fence has no execution phase)

**Note**: Fence instructions do not enter the ROB.

## Configuration

```scala
case class BaseConfig(
  rob_entries: Int = 16,                      // ROB depth
  rs_out_of_order_response: Boolean = true    // Out-of-order response
)
```

## Usage

```scala
// Instantiate components
val globalDecoder = Module(new GlobalDecoder)
val globalRS = Module(new GlobalReservationStation)

// Connect decoder to RS
globalRS.io.decode <> globalDecoder.io.issue

// Connect to RoCC interface
globalDecoder.io.rocc <> roccInterface.cmd
roccInterface.resp <> globalRS.io.resp

// Connect to domains
ballDomain.io.issue <> globalRS.io.ballIssue
memDomain.io.issue <> globalRS.io.memIssue
globalRS.io.ballComplete <> ballDomain.io.complete
globalRS.io.memComplete <> memDomain.io.complete
```

## Important Notes

1. **ROB ID Tracking**: Each instruction gets a unique rob_id that must be forwarded through the entire pipeline
2. **Backpressure**: If ROB is full, GlobalDecoder is blocked
3. **Fence Synchronization**: Fence blocks new instructions until ROB is empty
4. **Issue Limit**: At most rob_entries/2 instructions can be in-flight

## Related Documentation

- [Global RS Implementation](globalrs/README.md)
- [ROB Details](globalrs/GlobalROB.scala)
- [Ball RS](../rs/README.md)
