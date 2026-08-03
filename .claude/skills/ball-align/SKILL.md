---
name: ball-align
description: >
  Align a Buckyball Ball across the full stack (ctest → bemu → compiler → MLIR → RTL → UVM)
  to one contract. Use when the user asks for Ball alignment / contract sync, when bemu and
  Verilator/RTL disagree, when renaming ISA fields, or when extending tests after a Ball
  semantic change. Source of truth: docs/superpowers/ball-dev-guide.md.
---

**Important:** build/sim only via MCP from `.mcp.json` (`validate`, `bbdev_workload_*`, `bbdev_bemu_*`, `bbdev_bebop_verilator_*` / `bbdev_verilator_*`, `bbdev_uvm_*`). Do not call `bbdev` CLI or `nix develop -c bbdev` directly. If `buckyball-dev` is missing, stop and report it.

Gold = **ctest semantics**. bemu / compiler / RTL / UVM must share the same contract. Test env is **fail-hard**: illegal inputs panic/assert at every layer; no soft defaults or fallbacks.

Canonical guide: `docs/superpowers/ball-dev-guide.md`. This skill is the executable alignment workflow.

## Phase 0 - Lock the contract (before any code)

Write down and confirm with the user (stop if any item is missing):

1. ISA fields and shape sources (mset / instruction)
2. Element width and per-`iter` read/write footprint
3. Illegal-input table (same checks at every layer)
4. Output layout (tile / dense-pack / zero-fill)
5. Names match semantics (e.g. col vs group); renames must update ISA macros, emu, compiler, ctest, and regression tomls together

Hard rules:

- Delete dead fields or give them one meaning immediately; do not leave "for later"
- Addressing follows **mvin/mvout / bank row width**; no invented strides (common bug: treating a 16B row as 64B)
- Capacity (`max*`, bank depth): use measurable bounds from the current config; do not raise limits before small tests are green

## Phase 1 - Inventory by layer

Against `examples/balls/<name>/` and the target chip, list gaps:

| Kind | Content | Where to run |
|------|---------|--------------|
| Small tests | Short shapes, hand vectors, edge/illegal | bemu + Verilator |
| Bank tests | `iter≈BANK_LINES`, `srand`/`rand`; soft numeric may report LOSS | **bemu only** |
| Integration tests | Real consumers; must update on contract changes | bemu; small paths may use Verilator |
| MLIR | Under `examples/balls/<name>/workloads/mlir_tests/`; **not** under chip | FileCheck + optest ELF → bemu regression |
| UVM | Stimulus/scoreboard on the same contract | **After RTL is green** |

- MLIR: keep `bank` Op and lowered `ball` Op separate (see transpose: contract + optest)
- New ctest / optest → chip **bemu** `workloads-elf.toml` + `workloads-pk.toml`
- Verilator lists take **small tests only**, not bank tests

## Phase 2 - One semantics across the stack

Check each layer. Any fork is a **bug** — do not change semantics for one layer alone:

1. C expected / bemu / RTL / scoreboard share output layout
2. Same legal/illegal checks at every layer
3. Compiler: bank Op → assign-physical-banks → ball Op → intr; keep lit files per layer, do not mix

Pass `validate(chip=..., balldomain?=...)` before continuing.

## Phase 3 - RTL alignment checklist

- Params from toml / BallParam; `require` ties to `bankWidth` / `inBW` / `outBW`
- Latch fields on `cmdReq.fire`; SRAM read latency is 1 cycle; `status.idle/running` match the FSM
- Explicit Chisel widths (`+&`, wide enough `UInt`); hard-block in-place same-bank writes that destroy source data

## Phase 4 - Verification order (do not skip)

1. `bbdev_workload_build(chip=...)`
2. Small suite + bank tests via `bbdev_bemu_sim` / `bbdev_bemu_batch` — if bemu is red, fix **contract / tests / emu only**; do not touch RTL
3. Verilator **small tests** via `bbdev_bebop_verilator_sim` (or non-bebop `bbdev_verilator_sim`) — bemu green + Verilator red → RTL/timing; use `/waveform` + `/debug`
4. After fixes, re-run the full small suite plus previously failing edges; PMC from `bdb.ndjson` / `bdb.log` `pmctrace` (`elapsed`)
5. Sync integration tests; UVM only after RTL small tests are green

Full Verilator builds often MCP-timeout: check whether the server is still compiling; if binary/marker is ready, call `*_sim` directly.

## Phase 5 - Minimum delivery checklist

Copy and tick:

```
- [ ] Contract (fields, shapes, illegal table) + validate pass
- [ ] Small suite bemu green; bank tests bemu green
- [ ] bank/ball MLIR OpTest + listed in bemu regression
- [ ] RTL aligned + Verilator small tests green
- [ ] Integration tests synced; UVM (after RTL)
```

## Engineering constraints

- Comments and prints in English; KISS; no fallbacks that hide bugs
- On baremetal, avoid builtins that pull missing symbols
- Keep chip configs and regression tomls consistent
