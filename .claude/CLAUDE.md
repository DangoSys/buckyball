# Buckyball

A RISC-V based DSA (Domain Specific Architecture) framework. Built with Chisel 6.5.0 and Nix Flake.

## Project Structure

- `arch/src/main/scala/framework/` — framework core
  - `balldomain/blink/` — Blink protocol definitions (`BlinkIO`, `BankRead/Write`, `BallStatus`)
  - `balldomain/configs/` — `BallDomainParam` + TOML loaders
  - `balldomain/bbus/` — BBus interconnect
  - `balldomain/rs/` — `BallRsIssue` / `BallRsComplete` (issue/complete interfaces)
  - `memdomain/backend/banks/` — `SramReadIO` / `SramWriteIO`
  - `core/bbtile/` — BBTile integration (Rocket core + Buckyball)
  - `top/` — `GlobalConfig` (top-level parameter aggregation)
- `examples/balls/<name>/` — Ball implementations (arch / compiler / emu / workloads)
- `examples/chips/<chip>/configs/` — chip TOML (tile/core/balldomain/memdomain)
  - ball registration: `tiles/cores/balldomains/*.toml` (`ballNum`, `ballIdMappings`, `ballISA`)
- `arch/src/main/scala/sims/` — simulation configs
  - `verilator/` — Verilator config
- `bb-tests/` — tests
  - `workloads/lib/bbhw/isa/` — ISA C macros (one `.c` file per instruction)
  - `workloads/src/CTest/` — C test cases
- `bbdev/` — developer toolchain (Motia workflow backend)

## Blink Protocol

Balls connect to BBus through the Blink protocol. Every Ball implements the `HasBlink` trait.

```
BlinkIO(b: GlobalConfig, inBW: Int, outBW: Int):
  cmdReq:    Flipped(Decoupled(BallRsIssue))     // command input (includes BallDecodeCmd + rob_id)
  cmdResp:   Decoupled(BallRsComplete)           // completion output (includes rob_id)
  bankRead:  Vec(inBW, Flipped(BankRead))        // SRAM read ports
  bankWrite: Vec(outBW, Flipped(BankWrite))      // SRAM write ports
  status:    BallStatus { idle, running }        // status signals

BankRead/BankWrite metadata fields (all Input):
  bank_id, rob_id, ball_id, group_id

SramReadIO:  req.valid/ready + req.bits.addr  ->  resp.valid + resp.bits.data
SramWriteIO: req.valid/ready + req.bits(addr, data, mask, wmode)  ->  resp.valid + resp.bits.ok
```

Key timing rule: SRAM read latency is 1 cycle (`resp.valid` is asserted in the next cycle after `req.fire`).

## Registration Invariants

Ball registration is TOML under `examples/chips/<chip>/configs/tiles/cores/balldomains/`.
`cores/default.toml` selects which balldomain file is active (e.g. toy uses `balldomains/full.toml`).

When adding or modifying a balldomain TOML, these must hold:

1. `ballNum` equals `ballIdMappings` length
2. `ballId` is strictly `0, 1, 2, ...` with no gaps/duplicates
3. `ballName` / `funct7` / `mnemonic` have no duplicates
4. every `ballISA.bid` exists in `ballIdMappings`, and every ball has at least one ISA entry
5. relative `config=` paths resolve to existing files (when present)
6. `inBW` / `outBW` are positive integers

Use MCP `validate(chip=..., balldomain=...)` or `/check`.

## MCP Tools

Project MCP config lives in root `.mcp.json` (Claude Code / Codex / Cursor / any stdio MCP host).
Entry point: `scripts/claude/run_mcp_server.sh` → `bbdev/mcp/__main__.py` (tools in `bbdev/mcp/tools/`).
It cds to the repo root, sets `NIX_QUIET=1`, and keeps stdout MCP-clean.

**Important: agents must invoke build/sim/synth/test via MCP tools. Do not call `bbdev` CLI or `nix develop -c bbdev ...` directly.**
Humans use `bbdev` CLI (see `docs/zh/设计文档/主线架构/0.0.1/工具链/`). MCP auto-starts bbdev HTTP and each `bbdev_*` tool returns a `trace_id`; agents must query `bbdev_task_status(trace_id)` until the task reaches a terminal state.
If `buckyball-dev` is missing from the host tool list, stop and tell the human to reload project MCP from `.mcp.json`; do not invent a parallel workflow.

Daily agent path:
`bbdev_compiler_build` → `bbdev_workload_build` → `bbdev_bemu_sim` → `bbdev_bebop_verilator_run` (or uvm when needed)

### Validation
- `validate(chip="toy", balldomain?=)` — check balldomain TOML invariants (default: file selected by `cores/default.toml`)

### bbdev API wrappers (automatic server lifecycle + task status)
All bbdev POST endpoints are exposed as `bbdev_*` tools. Daily path prefers:
- `bbdev_compiler_build` / `bbdev_workload_{clean,build,tohex}`
- `bbdev_bemu_{sim,batch}`
- `bbdev_bebop_verilator_{clean,verilog,build,sim,run,batch}` — bebop-accelerated RTL
- `bbdev_verilator_{clean,verilog,build,sim,run}` — non-bebop Verilator RTL path
- `bbdev_uvm_{build,run}` / `bbdev_yosys_{run,verilog,synth}`
Also: `bbdev_bebop_p2e_*`, `bbdev_firesim_*`, `bbdev_dc_verilog`, `bbdev_kernel_build`

Every `bbdev_*` task submission returns immediately with `accepted=true`,
`processing=true`, and `trace_id`. Poll `bbdev_task_status(trace_id)`; only
`success=true` and `returncode=0` permits the next workflow stage.

Default Verilator config: `sims.verilator.BuckyballToyVerilatorConfig`
Pebble: `sims.verilator.BuckyballPebbleVerilatorConfig`
Workload binary names are `{chip}_{stem}-{platform}`, e.g. `toy_vecunit_matmul_ones-singlecore-baremetal`

### Analysis report paths
- Area reports: `bbdev/api/steps/yosys/log/hierarchy_report.txt` (submodule breakdown), `area_report.txt` (top-level)
- Timing report: `bbdev/api/steps/yosys/log/timing_report.txt`
- Simulation logs: `arch/log/<timestamp>/stdout.log`, `disasm.log`
- bdb debug log: `arch/log/<timestamp>/bdb.log`, with three DPI-C traces:
  - `[ITRACE]` — instruction issue/complete
  - `[MTRACE]` — SRAM reads/writes
  - `[PMCTRACE]` — Ball/Mem performance counters (elapsed cycles)

## Skills

Project skills live under `.claude/skills/` (canonical). Discovery links:
- Claude Code / Cursor: `.claude/skills/` (Cursor also has `.cursor/skills/ball-align` → same tree)
- Codex: `.codex/skills/<name>/SKILL.md` → same tree

- `/ball` — create a new Ball operator (full flow: implementation -> registration -> ISA -> CTest -> simulation)
- `/ball-align` — align Ball across ctest/bemu/compiler/MLIR/RTL/UVM to one contract (`docs/superpowers/ball-dev-guide.md`)
- `/check` — registration consistency check + auto-fix
- `/verify` — Ball functional verification (build -> simulation -> PMC analysis)
- `/optimize` — RTL area/latency optimization (applies to any module, not only Balls)
- `/debug` — simulation debugging (log analysis -> waveform -> failure pattern matching)
- `/waveform` — waveform analysis (`waveform-mcp` usage guide)

## Conventions

- Do not edit registration files while changing Ball implementation; do not edit implementation files while changing registration
- Chisel version is 6.5.0; do not use 6.6+ APIs
- Register CTests in CMakeLists via `add_cross_platform_test_target`
- **Do not call `bbdev` CLI or `nix develop -c bbdev ...` directly**; use MCP tools. If MCP is not loaded, report that and stop.
- Ball wrapper class names must match `ballName` in the chip balldomain TOML
- MCP starts bbdev via `bbdev start --server` (`iii` + `bbdev/api/.venv` motia). Missing venv/iii → hard error. Do not use Node `pnpm/motia`.
