# Buckyball

A RISC-V based DSA (Domain Specific Architecture) framework. Built with Chisel 6.5.0 and Nix Flake.

## Project Structure

- `arch/src/main/scala/framework/` — framework core
  - `balldomain/prototype/` — Ball operator implementations (one subdirectory per Ball)
  - `balldomain/blink/` — Blink protocol definitions (`BlinkIO`, `BankRead/Write`, `BallStatus`)
  - `balldomain/configs/` — `BallDomainParam` + `default.json` (`ballIdMappings`)
  - `balldomain/bbus/` — BBus interconnect
  - `balldomain/rs/` — `BallRsIssue` / `BallRsComplete` (issue/complete interfaces)
  - `memdomain/backend/banks/` — `SramReadIO` / `SramWriteIO`
  - `core/bbtile/` — BBTile integration (Rocket core + Buckyball)
  - `top/` — `GlobalConfig` (top-level parameter aggregation)
- `arch/src/main/scala/examples/toy/balldomain/` — toy config
  - `DISA.scala` — instruction opcodes (`funct7` BitPat)
  - `DomainDecoder.scala` — instruction decode table (`ListLookup`)
  - `bbus/busRegister.scala` — Ball generator registration (`match case`)
- `arch/src/main/scala/sims/` — simulation configs
  - `verilator/` — Verilator config
- `bb-tests/` — tests
  - `workloads/lib/bbhw/isa/` — ISA C macros (one `.c` file per instruction)
  - `workloads/src/CTest/toy/` — C test cases
  - `sardine/` — pytest test framework
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

When adding or modifying Ball registrations, all six conditions below must hold:

1. `ballNum` in `default.json` equals the length of `ballIdMappings`
2. `ballId` is strictly increasing (`0, 1, 2, ...`) with no gaps
3. No duplicated `ballId`
4. No duplicated `funct7` values in `DISA.scala`
5. Case names in `busRegister.scala` equal `ballName` set in `default.json`
6. BID values in `DomainDecoder.scala` equal `ballId` set in `default.json`

Use the `/check` skill to validate all invariants and optionally auto-fix issues.

## MCP Tools

The project configures the `buckyball-dev` MCP server with the following tools.

**Important: build, simulation, synthesis, and tests must be invoked via MCP tools. Do not call `bbdev` CLI or `nix develop` directly.**
`bbdev` CLI is for human developers, while MCP tools are for agents. MCP tools manage bbdev server lifecycle and call it through HTTP APIs.

### Validation
- `validate` — check all 6 registration invariants

### bbdev API wrappers (with automatic server lifecycle management)
- `bbdev_workload_build` — build CTests
- `bbdev_verilator_run(binary, config?, batch?, coverage?)` — full flow: clean -> verilog -> build -> sim
- `bbdev_verilator_verilog(config)` — generate Verilog; `config` is required
- `bbdev_verilator_build(jobs?, coverage?)` — build Verilator simulator
- `bbdev_verilator_sim(binary, batch?, coverage?)` — run simulation (requires prior build)
- `bbdev_sardine_run(workload?, coverage?)` — run batch tests
- `bbdev_yosys_synth(top?, config?)` — Yosys synthesis + OpenSTA timing analysis

Default config value: `sims.verilator.BuckyballToyVerilatorConfig`
Simulation binary naming format: `ctest_<name>_test_singlecore-baremetal`

### Analysis report paths
- Area reports: `bbdev/api/steps/yosys/log/hierarchy_report.txt` (submodule breakdown), `area_report.txt` (top-level)
- Timing report: `bbdev/api/steps/yosys/log/timing_report.txt`
- Coverage report: `bb-tests/sardine/reports/coverage/html/`
- Simulation logs: `arch/log/<timestamp>/stdout.log`, `disasm.log`
- bdb debug log: `arch/log/<timestamp>/bdb.log`, with three DPI-C traces:
  - `[ITRACE]` — instruction issue/complete
  - `[MTRACE]` — SRAM reads/writes
  - `[PMCTRACE]` — Ball/Mem performance counters (elapsed cycles)

## Skills

Project skills are under `.claude/skills/`:
- `/ball` — create a new Ball operator (full flow: implementation -> registration -> ISA -> CTest -> simulation)
- `/check` — registration consistency check + auto-fix
- `/verify` — Ball functional verification (build -> simulation -> coverage -> PMC analysis)
- `/optimize` — RTL area/latency optimization (applies to any module, not only Balls)
- `/debug` — simulation debugging (log analysis -> waveform -> failure pattern matching)
- `/waveform` — waveform analysis (`waveform-mcp` usage guide)

## Conventions

- Do not edit registration files while changing Ball implementation; do not edit implementation files while changing registration
- Chisel version is 6.5.0; do not use 6.6+ APIs
- Register CTests in CMakeLists via `add_cross_platform_test_target`
- **Do not call `bbdev` CLI or `nix develop -c bbdev ...` directly**; use MCP tools
- Ball wrapper class names must match `ballName` in `default.json`
