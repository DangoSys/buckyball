# Buckyball Agent Workflow (MCP + bbdev)

Agents talk to bbdev through the project MCP server. Humans use `bbdev` CLI.
Config is the repo-root `.mcp.json` — shared by Claude Code, Codex, Cursor, and any stdio MCP host.

## Layout

```
User / Agent host
  └── .mcp.json → bash scripts/claude/run_mcp_server.sh
                    └── nix develop -c python3 bbdev/mcp/__main__.py
                          ├── validate
                          └── bbdev_*  → bbdev HTTP (submit + trace_id)
```

| File | Role |
|------|------|
| `.mcp.json` | Host-agnostic MCP registration |
| `scripts/claude/run_mcp_server.sh` | cd to repo root, `NIX_QUIET=1`, clean stdout |
| `bbdev/mcp/` | MCP package: `common.py`, `server.py`, `tools/*.py` |
| `.claude/CLAUDE.md` | Agent rules |
| `docs/zh/设计文档/主线架构/0.0.1/工具链/` | Human CLI docs |

## Daily path

```text
bbdev_compiler_build(chip)
  → bbdev_task_status(trace_id) until success
  → bbdev_workload_build(chip)
  → bbdev_task_status(trace_id) until success
  → bbdev_bemu_sim(chip, binary)
  → bbdev_task_status(trace_id) until success
  → bbdev_bebop_verilator_run(binary, config)
  → bbdev_task_status(trace_id) until success
```

UVM when needed: `bbdev_uvm_build` / `bbdev_uvm_run`.

## MCP tools

### Validation
| Tool | Purpose |
|------|---------|
| `validate(chip, balldomain?)` | Chip balldomain TOML invariants (`ballIdMappings` / `ballISA`) |

### bbdev wrappers (all POST APIs)
| Tool | API |
|------|-----|
| `bbdev_config_install` | `/config/install` |
| `bbdev_compiler_build` | `/compiler/build` |
| `bbdev_task_status` | State for a submitted `trace_id` |
| `bbdev_workload_{clean,build,tohex}` | `/workload/{clean,build,tohex}` |
| `bbdev_bemu_{sim,batch}` | `/bebop/bemu/{sim,batch}` |
| `bbdev_bebop_verilator_*` | `/bebop/verilator/{clean,verilog,build,sim,run,batch}` |
| `bbdev_verilator_*` | `/verilator/{clean,verilog,build,sim,run}` (non-bebop) |
| `bbdev_vcs_*` | `/vcs/{clean,verilog,build,sim,run}` |
| `bbdev_bebop_p2e_*` | `/bebop/p2e/{clean,verilog,buildbitstream,runworkload,batch}` |
| `bbdev_uvm_{verilog,build,run}` | `/uvm/{verilog,build,run}` |
| `bbdev_yosys_{run,verilog,synth}` | `/yosys/{run,verilog,synth}` |
| `bbdev_dc_verilog` | `/dc/verilog` |
| `bbdev_firesim_*` | `/firesim/{enumeratefpgas,buildbitstream,infrasetup,runworkload}` |
| `bbdev_kernel_build` | `/kernel/build` |

Daily work: bebop + bemu + workload. Reload project MCP after changing `bbdev/mcp/`.

## Server lifecycle

On first `bbdev_*` call the MCP server (same path as human `bbdev start --server`):

1. Requires `iii` in PATH and `bbdev/api/.venv/bin/motia` (no auto-install; fail if missing)
2. Starts `bbdev start --server --port <auto>` (ports 5100–5500)
3. Waits until worker routes are registered
4. Submits HTTP and returns `accepted=true`, `processing=true`, and `trace_id`
5. Stops via `bbdev stop --server` on MCP exit

Port is dynamic. Do not use Node `pnpm/motia`.

Every `bbdev_*` tool is non-blocking. Submit a task once, then call
`bbdev_task_status(trace_id)` until it returns a terminal result. Only
`success=true` with `returncode=0` is a passing task. `failure=true` is terminal
and must stop the flow. A state-less trace is queued only when it was submitted by
the current MCP server; any other unknown trace fails.

## Slash commands

| Trigger | Skill |
|---------|-------|
| `/ball <Name>` | `.claude/skills/ball` |
| `/ball-align` | `.claude/skills/ball-align` (also `.codex/skills/`, `.cursor/skills/`) |
| `/verify <Name>` | `.claude/skills/verify` |
| `/optimize <Name>` | `.claude/skills/optimize` |
| `/check` | `.claude/skills/check` |

## Smoke test

```bash
# from any cwd; stdout must be JSON-only
bash /path/to/buckyball/scripts/claude/run_mcp_server.sh <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}
EOF
```
