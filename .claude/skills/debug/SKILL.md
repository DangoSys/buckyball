---
name: debug
description: Systematically debug Buckyball simulation failures. Use this skill when simulation returns FAILED, CTest fails, Chisel compilation errors appear, or users report abnormal Ball behavior. It covers log analysis, waveform analysis, and common failure pattern matching.
---

**Important: build and simulation operations must be called through MCP tools. Do not use bbdev CLI or nix develop directly.**

## Step 1 - Locate Logs

1. Find the log directory:
   - MCP tool JSON output includes `log_dir`
   - If absent, use `ls -t arch/log/ | head -5` to find recent log dirs
2. Confirm key log files exist:
   - `stdout.log` - program stdout (`PASSED`/`FAILED`, `printf`)
   - `disasm.log` - disassembled instruction stream
   - `bdb.log` - Buckyball hardware debug log (most important)
   - `bbdev/server.log` - bbdev server log (compilation errors usually appear here)

## Step 2 - Layered Analysis

Analyze from high level to low level, ruling out simple issues first.

### Level 1: Compilation errors (`bbdev/server.log`)

If MCP tools return HTTP 500 or `returncode=1`, inspect `server.log` first:
- Chisel compile errors (type mismatch, missing registration, etc.)
- mill build errors (dependency issues)
- CTest compile errors (C syntax/link errors)

### Level 2: Program output (`stdout.log`)

- Search for `PASSED` / `FAILED` to confirm test results
- Search for `printf` outputs and compare actual vs expected values
- Search for `panic` / `abort` / `trap` to detect exceptions

### Level 3: Instruction stream (`disasm.log`)

- Confirm custom Ball instructions were executed (search `custom3`)
- Check instruction order (`mvin -> ball_op -> mvout -> fence`)
- Check for traps or exceptions

### Level 4: Hardware tracing (`bdb.log`)

This is the most important log for Ball logic bugs. It contains three traces:

**[ITRACE] instruction trace:**
- `ISSUE rob_id=X domain=Y funct=0xZZ` - instruction issued
- `COMPLETE rob_id=X` - instruction completed
- Check: issued? completed? completion order correct?

**[MTRACE] memory trace:**
- `READ ch=X vbank=Y group=Z addr=0xAA` - SRAM read
- `WRITE ch=X vbank=Y group=Z addr=0xAA data=0x...` - SRAM write
- Check: addresses correct? data correct? `bank_id` matches?

**[PMCTRACE] performance trace:**
- `BALL ball_id=X rob_id=Y elapsed=Z` - Ball operation latency
- `LOAD/STORE rob_id=X elapsed=Y` - memory operation latency
- Check: elapsed reasonable? any unusually long operations?

### Level 5: Waveform analysis (`waveform-mcp`)

If logs are not enough, use waveform-mcp for cycle-level analysis. See `/waveform` skill.

## Step 3 - Common Failure Patterns

### 1. Ball not responding (`cmdResp` never fires)
**Symptoms:** timeout or deadlock; `bdb.log` has `ISSUE` but no `COMPLETE`
**Causes:**
- FSM stuck in a state (check transition conditions)
- `SRAM resp.valid` not consumed (for example, missing `resp.ready := true.B`)
- `cmdResp.valid` never asserted

### 2. All-zero output data
**Symptoms:** CTest FAILED; output matrix is all zeros
**Causes:**
- write address bug (`waddr` not incremented)
- write mask all zero (for example, forgot `mask := 1`)
- wrong `bank_id` (writes into wrong bank)

### 3. Output unchanged (`output == input`)
**Symptoms:** CTest FAILED; output equals input
**Causes:**
- compute logic not executed (compute state skipped)
- read data written back without processing

### 4. Partial data errors
**Symptoms:** CTest FAILED; some rows correct, others wrong
**Causes:**
- `iter` count miscalculated (missing reads/writes for some rows)
- address offset bug (row stride error)
- boundary condition bug

### 5. SRAM timing error
**Symptoms:** data appears shifted by one row
**Causes:**
- SRAM read latency is 1 cycle, but code reads `resp.bits.data` in the same cycle as `req.fire`
- Correct behavior: wait one cycle and read when `resp.valid` is high

### 6. `bank_id` conflict
**Symptoms:** assertion failure or corrupted data
**Causes:**
- `op1_bank` and `wr_bank` use the same bank (read/write conflict)
- multiple Balls access the same bank concurrently

### 7. `rob_id` mismatch
**Symptoms:** completion order looks incorrect
**Causes:**
- `cmdResp.bits.rob_id` is wrong
- `rob_id` was not latched on `cmdReq.fire`

## Step 4 - Fix and Verify

1. Modify Chisel source code based on issues found in logs/waveforms
2. Rebuild and rerun simulation through MCP tools to verify fixes
3. If new failures appear, return to Step 2 and reanalyze
