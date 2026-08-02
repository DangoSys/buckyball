# Buckyball Agent Workflow (MCP + bbdev)

Agents talk to bbdev through the project MCP server. Humans use `bbdev` CLI.
Config is the repo-root `.mcp.json` — shared by Claude Code, Codex, Cursor, and any stdio MCP host.

## Layout

```
User / Agent host
  └── .mcp.json → bash scripts/claude/run_mcp_server.sh
                    └── nix develop -c python3 scripts/claude/mcp_server.py
                          ├── validate
                          └── bbdev_*  → bbdev HTTP (auto start + poll /result/{trace_id})
```

| File | Role |
|------|------|
| `.mcp.json` | Host-agnostic MCP registration |
| `scripts/claude/run_mcp_server.sh` | cd to repo root, `NIX_QUIET=1`, clean stdout |
| `scripts/claude/mcp_server.py` | MCP tools + bbdev lifecycle |
| `.claude/CLAUDE.md` | Agent rules |
| `docs/zh/设计文档/主线架构/0.0.1/工具链/` | Human CLI docs |

## Daily path

```text
bbdev_compiler_build(chip)
  → bbdev_workload_build(chip)
  → bbdev_bemu_sim(chip, binary)
  → bbdev_bebop_verilator_run(binary, config)
```

UVM when needed: `bbdev_uvm_build` / `bbdev_uvm_run`.

## MCP tools

### Validation
| Tool | Purpose |
|------|---------|
| `validate(chip, balldomain?)` | Chip balldomain TOML invariants (`ballIdMappings` / `ballISA`) |

### Preferred bbdev wrappers
| Tool | API |
|------|-----|
| `bbdev_compiler_build` | `/compiler/build` |
| `bbdev_workload_clean` | `/workload/clean` |
| `bbdev_workload_build` | `/workload/build` |
| `bbdev_bemu_sim` | `/bebop/bemu/sim` |
| `bbdev_bemu_batch` | `/bebop/bemu/batch` |
| `bbdev_bebop_verilator_*` | `/bebop/verilator/{clean,verilog,build,sim,run,batch}` |
| `bbdev_uvm_build` / `bbdev_uvm_run` | `/uvm/build`, `/uvm/run` |
| `bbdev_yosys_synth` | `/yosys/synth` |

`bbdev_verilator_*` still exist for the legacy non-bebop path; daily work should use bebop tools.

## Server lifecycle

On first `bbdev_*` call the MCP server (same path as human `bbdev start --server`):

1. Requires `iii` in PATH and `bbdev/api/.venv/bin/motia` (no auto-install; fail if missing)
2. Starts `bbdev start --server --port <auto>` (ports 5100–5500)
3. Waits until worker routes are registered
4. Submits HTTP + polls `bbdev/api/data/state_store.db/<trace_id>.bin` (same as CLI)
5. Stops via `bbdev stop --server` on MCP exit

Port is dynamic. Do not use Node `pnpm/motia`.

## Slash commands

| Trigger | Skill |
|---------|-------|
| `/ball <Name>` | `.claude/skills/ball` |
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
