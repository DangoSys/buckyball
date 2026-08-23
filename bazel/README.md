# Bazel layout

Bazel **organizes** chip/core/ball sources and repo metadata. **All compilation runs in bbdev** under `nix develop` (mill / cmake / ninja / cargo).

| Path | Role |
|------|------|
| `//examples/chips/<chip>` | Chip-scoped filegroup aliases (`ctests`, `mlir_tests`, `ballISA`) |
| `//examples/balls/<ball>` | Ball source filegroups |
| `//examples:{ctests,mlir_tests,ballISA,bank_params}` | Chip-selected via `--//bazel/config:chip` |
| `config/` | `--//bazel/config:chip=` + `bb_chips` / `bb_cores` repo rules |
| `configparse/` | Python scripts (`chip_bundle.py`, `workload_cmake_defs.py`, …) — invoked by bbdev, not Bazel actions |

## bbdev build entrypoints

```
bbdev compiler --build '--chip toy'
bbdev workload --build '--chip toy'
bbdev workload --build '--chip toy --model qwen3 --rushB bemu'
bbdev verilator --verilog '--chip toy'
python3 bazel/configparse/chip_bundle.py --repo . --chip toy --all
```

## Jurisdiction

- **bbdev** runs all compile/sim flows (mill, cmake, ninja, cargo).
- **Bazel** exposes chip-selected source sets and generates `bb_chips` index at fetch time.
- **chip bundle** (`examples/chips/<chip>/generated/`) is produced by `chip_bundle.py`, usually via bbdev `install_bundle()` before bemu/workload.

Optional workload flags (bbdev CLI, not Bazel):

```
--model qwen3
--rushB bemu|verilator
--stable
```

After editing a core `ballISA` table, regenerate with workload build or:

```
python3 bazel/configparse/toml2json.py examples/cores/<core>/configs/balldomains/default.toml /tmp/x.json --repo .
python3 bazel/configparse/json_to_ball_isa.py /tmp/x.json examples/cores/<core>/isa/ballISA.h
```
