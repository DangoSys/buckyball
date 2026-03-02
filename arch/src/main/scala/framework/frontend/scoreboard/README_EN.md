# Buckyball ILP Proposal: Bank Scoreboard for Dependency Tracking

## 1. Background & Problem

### 1.1 Current State

Buckyball's Global ROB supports out-of-order issue and out-of-order completion. However, software inserts `bb_fence()` between every pair of instructions. A fence requires the ROB to **drain completely** before accepting the next instruction, effectively reducing ILP to 1 — making the out-of-order machinery **dead code**.

Typical pattern:
```
MVIN(bank0) → fence → RELU(bank0→bank1) → fence → MVOUT(bank1)
```

The fundamental problem fences solve is **bank-level RAW/WAR/WAW data hazards**. But fences use the coarsest possible granularity — a global barrier. Even instructions operating on completely independent banks are forced to serialize.

### 1.2 Goals

- Implement an **instruction-agnostic Bank Scoreboard** in `framework/frontend/scoreboard/`
- The scoreboard only cares about each instruction's **read bank set** and **write bank set**, not the instruction type
- Retain fence instruction semantics, but redefine fence as a **barrier point within the ROB** — the ROB scan for issue candidates stops at the fence boundary
- Unify all instructions' bank information encoding into designated bits of rs1
- Enable out-of-order skip-issue: instructions on different banks can leapfrog hazard-blocked instructions

---

## 2. Unified rs1 Bank Encoding

### 2.1 Design Principle

All instructions encode bank operands in rs1:

```
rs1[7:0]   = bank_0  (1st operand bank / MVIN write bank / MVOUT read bank)
rs1[15:8]  = bank_1  (2nd operand bank, dual-operand instructions only)
rs1[23:16] = bank_2  (write/result bank)
```

### 2.2 Per-Instruction New Encoding

#### Mem Domain Instructions

| Instruction | rs1 | rs2 | Change Summary |
|-------------|-----|-----|----------------|
| **MVIN** (24) | `[7:0]=wr_bank`, `[63:8]=mem_addr` | `[9:0]=depth, [28:10]=stride, [63:29]=special` | bank moved from rs2 to rs1[7:0]; mem_addr shifted to upper rs1 |
| **MVOUT** (25) | `[7:0]=rd_bank`, `[63:8]=mem_addr` | `[9:0]=depth, [28:10]=stride, [63:29]=special` | Same as MVIN |
| **MSET** (23) | `[7:0]=bank_id` | `[4:0]=row, [9:5]=col, [10]=alloc, ...` | bank moved from rs2 to rs1[7:0] |

#### Ball Domain Instructions

| Instruction | Current Encoding | New Change |
|-------------|-----------------|------------|
| **MATMUL_WARP16** (32) | op1=rs1[7:0], op2=rs1[15:8], wr=**rs2[7:0]** | wr moves to **rs1[23:16]**; rs2 repacked (iter starts at bit 0) |
| **RELU** (38) | op1=rs1[7:0], wr=**rs2[7:0]** | wr moves to **rs1[23:16]** |
| **TRANSPOSE** (34) | op1=rs1[7:0], wr=**rs1[15:8]** | wr moves to **rs1[23:16]** |
| **IM2COL** (33) | op1=rs1[7:0], wr=**rs1[15:8]** | wr moves to **rs1[23:16]** |
| **CONCAT** (39) | op1=rs1[7:0], wr=**rs2[7:0]** | wr moves to **rs1[23:16]** |
| **TRANSFER** (45) | op1=rs1[7:0], wr=**rs2[7:0]** | wr moves to **rs1[23:16]** |

> BBFP_MUL(26), MATMUL_WS(27), ABFT_SYSTOLIC(42), CONV(43), CIM(44) and other dual-operand instructions follow the MATMUL_WARP16 pattern.

### 2.3 Unified Extraction

After unification, the GlobalDecoder only needs:
```
rd_bank_0 = rs1[7:0]       // always
rd_bank_1 = rs1[15:8]      // always (valid flag controls usage)
wr_bank   = rs1[23:16]     // for Ball instructions
          or rs1[7:0]      // for MVIN/MSET (write bank == bank_0)
```

Plus a simple valid lookup table indexed by func7, producing a fully instruction-agnostic `BankAccessInfo`.

---

## 3. Bank Scoreboard Design

### 3.1 Location

`framework/frontend/scoreboard/BankScoreboard.scala`

### 3.2 Core Interface

```scala
class BankAccessInfo(bankIdLen: Int) extends Bundle {
  val rd_bank_0_valid = Bool()
  val rd_bank_0_id    = UInt(bankIdLen.W)
  val rd_bank_1_valid = Bool()
  val rd_bank_1_id    = UInt(bankIdLen.W)
  val wr_bank_valid   = Bool()
  val wr_bank_id      = UInt(bankIdLen.W)
}
```

The scoreboard **only receives `BankAccessInfo`** — it has no knowledge of instruction types, domains, or func7 codes.

### 3.3 Internal Data Structure

```scala
// Read counter: multi-bit, allows multiple instructions to read the same bank concurrently (RR is not a conflict)
val bankRdCount = RegInit(VecInit(Seq.fill(bankNum)(0.U(cntWidth.W))))
// cntWidth = log2Ceil(rob_entries + 1)

// Write flag: 1-bit, WAW rule guarantees at most 1 writer in-flight per bank
val bankWrBusy = RegInit(VecInit(Seq.fill(bankNum)(false.B)))
```

**Why is 1-bit sufficient for wrCount?**
The hazard detection rule requires `bankWrBusy[X]==false` before issuing a write to bank X (WAW blocking). Therefore, it is impossible for two write instructions targeting the same bank to be in-flight simultaneously.

**Why does rdCount need multiple bits?**
Multiple instructions reading the same bank concurrently is allowed (RR is not a conflict). A counter is needed to track how many concurrent readers exist, so write operations can detect WAR hazards.

### 3.4 Hazard Detection Rules

```
New instruction reads bank X  → requires bankWrBusy[X] == false   (RAW hazard: bank is being written)
New instruction writes bank X → requires bankRdCount[X] == 0       (WAR hazard: bank is being read)
                                 AND     bankWrBusy[X] == false    (WAW hazard: bank is being written)
```

```scala
def hasHazard(info: BankAccessInfo): Bool = {
  val rd0 = info.rd_bank_0_valid && bankWrBusy(info.rd_bank_0_id)
  val rd1 = info.rd_bank_1_valid && bankWrBusy(info.rd_bank_1_id)
  val wr  = info.wr_bank_valid && (
    bankRdCount(info.wr_bank_id) =/= 0.U ||
    bankWrBusy(info.wr_bank_id)
  )
  rd0 || rd1 || wr
}
```

### 3.5 Counter/Flag Updates

**On issue** (issue.fire):
```scala
when(info.rd_bank_0_valid) { bankRdCount(info.rd_bank_0_id) += 1.U }
when(info.rd_bank_1_valid) { bankRdCount(info.rd_bank_1_id) += 1.U }
when(info.wr_bank_valid)   { bankWrBusy(info.wr_bank_id)   := true.B }
```

**On complete** (complete.fire):
```scala
when(info.rd_bank_0_valid) { bankRdCount(info.rd_bank_0_id) -= 1.U }
when(info.rd_bank_1_valid) { bankRdCount(info.rd_bank_1_id) -= 1.U }
when(info.wr_bank_valid)   { bankWrBusy(info.wr_bank_id)   := false.B }
```

Completion requires retrieving `BankAccessInfo` from the ROB entry, so **GlobalRobEntry must store `BankAccessInfo`**.

### 3.6 MSET Precise Tracking

MSET is marked as `wr_bank_valid=true, wr_bank_id=rs1[7:0]`. It serializes only with instructions accessing the same bank, without blocking other banks.

### 3.7 GP Domain (RVV) Instructions

GP instructions don't access scratchpad banks. Their `BankAccessInfo` has all valid flags set to false — they are never blocked by the scoreboard and don't block others.

---

## 4. Fence Instruction — New Semantics

### 4.1 Design

Fence **enters the ROB** (currently it does not). But it is not dispatched to any execution domain.

Fence acts as an **issue barrier** within the ROB. When the ROB scans from head to find issuable instructions, it **stops scanning at the first unresolved fence** — all instructions after the fence are invisible to the issue logic.

### 4.2 Fence Lifecycle

1. Fence enters ROB, receives an entry, marked with a fence flag
2. ROB issue scan uses fence as boundary — only instructions before fence are candidates
3. When fence reaches head (all prior instructions committed), fence auto-completes
4. Head advances past fence, subsequent instructions become visible for issue

### 4.3 Relationship with Scoreboard

Instructions before the fence are still managed by the scoreboard — different-bank instructions can issue out of order. The fence only limits the scan window boundary, ensuring post-fence instructions cannot leapfrog the fence.

### 4.4 Practical Effect

- **Without fence**: All instructions in ROB are candidates based on bank dependencies — maximum parallelism
- **With fence**: Instructions before and after fence are strictly ordered; instructions within each group can still reorder freely
- Software can selectively use fence for forced ordering (debugging, special semantics, etc.)

---

## 5. ROB Issue Logic Modification

### 5.1 Current Logic

```scala
// Scan from head, find first valid && !issued && !complete instruction
scanValid(i) := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
```

### 5.2 New Logic

```scala
// Add two conditions: (1) no bank hazard (2) not behind a fence
scanValid(i) := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
                && !hasHazard(robEntries(ptr).bankAccess)
                && !isBehindFence(i)
```

**isBehindFence computation:**
Scan from head; once a valid, uncompleted fence entry is encountered, all subsequent positions are marked `isBehindFence = true`.

```scala
val fenceBarrier = Wire(Vec(rob_entries, Bool()))
var seenFence = false.B
for (i <- 0 until rob_entries) {
  val ptr = (headPtr + i.U) % rob_entries.U
  val isFence = robValid(ptr) && isFenceEntry(ptr) && !robComplete(ptr)
  seenFence = seenFence || isFence
  fenceBarrier(i) := seenFence  // fence itself and everything after are masked
}
```

### 5.3 Fence Auto-Completion

When fence becomes head (all prior instructions committed), it auto-marks as issued + complete:

```scala
when (robValid(headPtr) && isFenceEntry(headPtr)) {
  robIssued(headPtr)   := true.B
  robComplete(headPtr) := true.B
}
```

---

## 6. Files to Modify

### Hardware (Scala)

| File | Change |
|------|--------|
| **NEW** `framework/frontend/scoreboard/BankScoreboard.scala` | BankAccessInfo definition, BankScoreboard module |
| `framework/frontend/decoder/GobalDecoder.scala` | Add BankAccessInfo to PostGDCmd; add bank extraction logic with valid lookup table |
| `framework/frontend/globalrs/GlobalROB.scala` | Add BankAccessInfo + isFence to GlobalRobEntry; integrate BankScoreboard; modify issue logic (hazard + fence barrier); fence auto-completion |
| `framework/frontend/globalrs/GlobalReservationStation.scala` | Remove old fence logic (fenceActive etc.); fence now enters ROB instead of stalling until drain |
| `framework/memdomain/frontend/cmd_channel/decoder/DomainDecoder.scala` | Extract bank_id from rs1[bankIdLen-1:0]; mem_addr from rs1 upper bits; iter/stride from new rs2 positions |
| `examples/toy/balldomain/DomainDecoder.scala` | wr_bank from rs1[23:16]; update decode table |

### Software (C instruction macros)

| File | Change |
|------|--------|
| `bb-tests/.../isa/24_mvin.c` | bank_id to rs1[7:0], mem_addr to rs1 upper bits |
| `bb-tests/.../isa/25_mvout.c` | Same as MVIN |
| `bb-tests/.../isa/23_mset.c` (or equivalent) | bank_id to rs1[7:0] |
| `bb-tests/.../isa/32_mul_warp16.c` | wr_bank from rs2[7:0] to rs1[23:16] |
| `bb-tests/.../isa/38_relu.c` | wr_bank from rs2[7:0] to rs1[23:16] |
| `bb-tests/.../isa/34_transpose.c` | wr_bank from rs1[15:8] to rs1[23:16] |
| `bb-tests/.../isa/33_im2col.c` | wr_bank from rs1[15:8] to rs1[23:16] |
| `bb-tests/.../isa/45_transfer.c` | wr_bank from rs2[7:0] to rs1[23:16] |
| Other Ball instruction macro files | Similar treatment |
| All test .c files | **Remove `bb_fence()` calls** |

---

## 7. Implementation Steps

1. Define `BankAccessInfo` Bundle; create `BankScoreboard` module in `framework/frontend/scoreboard/`
2. Modify software instruction macros (MVIN/MVOUT/MSET/Ball instructions) — unify bank encoding to rs1
3. Modify `MemDomainDecoder` to adapt to new rs1/rs2 field positions
4. Modify `BallDomainDecoder` — wr_bank from rs1[23:16]
5. Modify `GlobalDecoder` — add `BankAccessInfo` extraction logic
6. Modify `GlobalROB` — integrate BankScoreboard, implement hazard detection + fence barrier scan
7. Modify `GlobalReservationStation` — remove old fence logic, fence enters ROB
8. Remove `bb_fence()` calls from tests
9. Compile + run tests

---

## 8. Verification

1. Chisel compilation passes with no type/syntax errors
2. Run all CTests (relu_test, transpose_test, mvin_mvout_test, transfer_test, tiled_matmul, etc.)
3. All tests produce correct results without fence
4. Optional: waveform inspection confirming different-bank instructions issue in parallel
5. Optional: test with fence re-inserted, confirming fence barrier semantics work correctly
