# Buckyball Claude Code Workflow

Claude Code is the interactive frontend, and bbdev is the execution backend. Claude calls bbdev HTTP APIs through the MCP server (server mode with automatic lifecycle management).

## Three Workflows

| # | Trigger | Function |
|---|------|------|
| 1 | `/ball <Name>` | Create a new Ball: implementation -> registration -> ISA macro -> CTest -> build -> simulation verification |
| 2 | `/verify <Name>` | Verify a Ball: completeness check -> fill missing parts -> build -> simulation -> coverage analysis |
| 3 | `/optimize <Name>` | Optimize a Ball: area (Yosys) + timing (OpenSTA) + latency (simulation cycles) -> optimize -> regression verification |

## Architecture

```
User ──→ Claude Code (slash commands + CLAUDE.md)
              │
              ├── Code read/write: Read/Edit/Write
              ├── Static validation: MCP validate
              └── Build/sim/synth/test: MCP bbdev_* -> bbdev HTTP API
                    │
                    └── bbdev server (Motia workflow backend, lifecycle managed by MCP)
                          ├── POST /verilator/run      Full flow: clean->verilog->build->sim
                          ├── POST /verilator/verilog  Generate Verilog (supports --balltype)
                          ├── POST /verilator/build    Build Verilator (supports --coverage)
                          ├── POST /verilator/sim      Run simulation (supports --coverage)
                          ├── POST /workload/build     Build CTest
                          ├── POST /sardine/run        Batch tests (supports --coverage -> coverage report)
                          └── POST /yosys/synth        Yosys synthesis + OpenSTA timing analysis
```

## File List

| File | Description |
|------|------|
| `scripts/claude/mcp_server.py` | MCP server: validate + bbdev API wrappers + server lifecycle management |
| `.claude/settings.json` | MCP configuration |
| `CLAUDE.md` | Global instructions: project structure, Blink protocol, registration invariants, tool usage |
| `.claude/commands/ball.md` | Full flow for creating a Ball via `/ball <Name>` |
| `.claude/commands/verify.md` | Ball verification via `/verify <Name>` |
| `.claude/commands/optimize.md` | Ball optimization via `/optimize <Name>` |
| `.claude/commands/check.md` | Static validation via `/check` |

## MCP Server Tool List

### Validation
| Tool | Function |
|------|------|
| `validate` | Check 6 registration invariants (`ballId` increasing, unique `funct7`, BID alignment, etc.) |

### bbdev API Wrappers
| Tool | API | Description |
|------|-----|------|
| `bbdev_workload_build` | `/workload/build` | Build CTest |
| `bbdev_verilator_run` | `/verilator/run` | Full flow: clean->verilog->build->sim |
| `bbdev_verilator_verilog` | `/verilator/verilog` | Generate Verilog |
| `bbdev_verilator_build` | `/verilator/build` | Build Verilator |
| `bbdev_verilator_sim` | `/verilator/sim` | Run simulation |
| `bbdev_sardine_run` | `/sardine/run` | Batch tests |
| `bbdev_yosys_synth` | `/yosys/synth` | Yosys synthesis + OpenSTA |

## bbdev Server Lifecycle

MCP server manages bbdev server automatically:
- Auto-start on first `bbdev_*` call (`pnpm dev --port <port>`)
- Clean BullMQ AOF before startup to avoid replaying stale events
- Auto-select port from 5100-5500
- Return only after health check passes
- Check liveness before each call, auto-restart if down
- Clean up automatically when MCP server exits

## Detailed Workflow

### `/ball <Name>` - Create Ball

1. **Requirement collection**: read `default.json`/`DISA.scala` to determine `ballId`/`funct7`, then confirm function/inBW/outBW/op2 with user
2. **Implement Ball**: reference existing Ball code and create wrapper/core/config under `prototype/`
3. **Register**: update `default.json` + `busRegister` + `DISA` + `DomainDecoder`
4. **ISA macro**: create C macro file and update `isa.h`
5. **CTest**: create test `.c`, register in `CMakeLists.txt`, append sardine list
6. **Verification**: `validate` -> `bbdev_workload_build` -> `bbdev_verilator_run` -> PASS/FAIL

### `/verify <Name>` - Verify Ball

1. **Completeness check**: verify registration/ISA macro/CTest/sardine entries, fill missing parts
2. **Build + simulation**: `bbdev_workload_build` -> `bbdev_verilator_run`
3. **Coverage analysis**: `bbdev_sardine_run(coverage=true)` -> inspect report -> propose extra tests
4. **Failure analysis**: read simulation logs -> analyze Chisel code -> propose fixes

### `/optimize <Name>` - Optimize Ball

1. **Baseline measurement**: `bbdev_yosys_synth` (area + timing) + `bbdev_verilator_run` (cycle count)
2. **Area analysis**: extract submodule area from `hierarchy_report`, identify large contributors
3. **Timing/latency analysis**: critical paths in `timing_report` + simulation cycle count + FSM source analysis
4. **Optimization plan**: quantified options (method/area delta/latency delta/frequency impact/trade-off)
5. **Implementation**: modify Chisel code
6. **Post-opt measurement**: rerun Yosys + Verilator and output before/after report
