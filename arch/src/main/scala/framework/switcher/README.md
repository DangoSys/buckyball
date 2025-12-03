# Switcher Module

This directory contains two small but critical adapters that translate between Ball devices' physical memory ports and a unified "virtual line" representation.

- `ToVirtualLine`: Merges SPAD/ACC physical ports into a uniform virtual interface carrying metadata (`is_acc`, `bank_id`, `rob_id`).
- `ToPhysicalLine`: Restores the virtual interface back into physical SPAD/ACC ports, preserving data and `rob_id`.

## Goals
- **Unified view**: Provide a single, consistent interface for memory requests and responses regardless of origin (SPAD or ACC).
- **Explicit metadata**: Attach `is_acc`, `bank_id`, and `rob_id` so downstream routing and accounting are straightforward.
- **Clear behavior**: Support either bank-sharing with arbitration or segmented direct mapping, depending on system choice.
- **Type correctness**: Keep widths and masks consistent; no implicit format conversions.

## Interfaces
- Physical ports:
  - `SramReadWithRobId(n, w)`: wraps `SramReadIO(n, w)` plus `rob_id`.
  - `SramWriteWithRobId(n, w, mask_len)`: wraps `SramWriteIO(n, w, mask_len)` plus `rob_id`.
- Virtual ports:
  - `SramReadWithInfo(n, w)`: `SramReadIO` plus `rob_id`, `is_acc` (Bool), `bank_id`.
  - `SramWriteWithInfo(n, w, mask_len)`: `SramWriteIO` plus the same metadata.
- Widths:
  - `rob_id` depends on `b.rob_entries`.
  - `bank_id` wide enough for `b.sp_banks + b.acc_banks`.
  - `is_acc` is a `Bool`.

## ToVirtualLine
- Inputs:
  - `sramRead_i/sramWrite_i`: size `b.sp_banks`.
  - `accRead_i/accWrite_i`: size `b.acc_banks`.
- Outputs:
  - `sramRead_o/sramWrite_o`: virtual lines.
- Port count (choose one design and keep consistent):
  - **Max** design: `numBanks = max(b.sp_banks, b.acc_banks)`. Shared banks arbitrate (typically SPAD priority); tail banks connect to whichever exists.
  - **Sum** design: `numBanks = b.sp_banks + b.acc_banks`. Lower range maps SPAD; upper range maps ACC. No arbitration; direct mapping with metadata.
- Read path:
  - Drive `req` from physical to virtual; broadcast `resp` from virtual to the selected physical endpoint, using `ready` to consume.
  - Set `rob_id/is_acc/bank_id` based on origin and index. Use `false.B/true.B` for `is_acc`.
- Write path:
  - Map write `addr/data/mask` and metadata to virtual lines.
  - No `resp` channel on writes; only handshake and field mapping.

## ToPhysicalLine
- Inputs:
  - `sramRead_i/sramWrite_i`: virtual lines (same size as ToVirtualLine outputs).
- Outputs:
  - `sramRead_o/sramWrite_o`: size `b.sp_banks`.
  - `accRead_o/accWrite_o`: size `b.acc_banks`.
- Routing:
  - **Max** design: Use `is_acc` and `bank_id` to select the ACC bank; SPAD uses the virtual index `i`.
  - **Sum** design: Lower (SPAD) and upper (ACC) segments map 1:1 back to physical ports; `bank_id` marks the internal index.
- Timing notes:
  - If read `resp` and meta signals form a combinational loop, consider `RegNext` on meta to break it.

## Integration
- In `bbus`, each Ball connects physical ports → `ToVirtualLine` → virtual interface.
- Then `ToPhysicalLine` restores to physical ports that connect to `MemRouter`.
- If a strict direct-connect behavior is desired, use the **Sum** design and ensure both sides agree on vector sizes.

## Common Pitfalls
- Assigning `0.U/1.U` to `is_acc` (must be `false.B/true.B`).
- Using `io.sramRead_i(j)` instead of `io.accRead_i(i)` for ACC reads.
- Typos like `b.acc_Banks` (should be `b.acc_banks`).
- Dangling logical operators (e.g., trailing `||`) causing compile errors.
- Missing write-path mapping, leaving `ready` low and blocking writes.

## Recommendations
- Decide upfront: **Max + arbitration** vs **Sum + segmented direct mapping** and keep both modules and top-level wiring consistent.
- Align priority policy (SPAD vs ACC) with system requirements if using shared-bank arbitration.
- Keep `rob_id` and `bank_id` consistent to avoid accounting and replay issues.
