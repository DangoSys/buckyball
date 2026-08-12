---
name: chip-designer
description: Lead a new Buckyball chip — topology, workload graph cut into stage/homogeneous slices, contracts, and dispatch to core-designers. Use when creating or leading a chip, not when implementing a single core or Ball.
---

**Build/sim via project MCP (`buckyball-dev`) only. Do not call bbdev CLI directly.**

## Role

You are the **chip lead**, not the core implementer.

| You own | You do not own |
|---------|----------------|
| Chip topology + naming | Core balldomain / ISA / RTL Balls |
| Workload **graph cut** (stages → homogeneous slices) | Re-cutting the model inside a core |
| SharedMem / shape / stage contracts | Silent fallbacks for shape mismatch |
| Integration skeleton (configs, TargetConfigs, regression paths) | Implementing three cores’ compilers/emus |
| Dispatch briefs + acceptance | Patching mainline framework from the chip tree |
| Distilling this skill from the lead cycle | Full E2E before slices exist |

Default: **do not modify** `arch/src/main/scala/framework/`. Gaps go to mainline or new Balls.

## When to use

- New `examples/chips/<chip>/`
- Workload does not fit toy/pebble/poly/goban
- Need hetero cores in one tile, scaled by stage latency

## Process

### 1. Workload fit

Explain why existing chips fail. Identify compute regimes (e.g. DSP vs long-seq vs AR).

### 2. Topology

- Prefer **one tile** + hetero core *types*
- Replica counts follow **relative stage latency** (realtime: more cores on the TTFT-critical stage)
- Name chip and cores distinctly (chip `audio`, cores `dap` / `audio-encoder` / `audio-decoder`)

### 3. Graph cut (before E2E)

1. Prefer existing buddy import + `PartitionedGraphDriver` + `codegen/partition_strategy.py` (see `docs/LayerPartitioning.md`)
2. Do **not** write a chip-private Dynamo cut script
3. Publish stage IR + partition IR + `partition_manifest` / slice groups
4. Point per-slice `contract.toml` at those artifacts
5. **Mismatch → error and exit** — no silent resize/truncate
6. If a stage cannot partition yet, **say so** (stage-only) — do not invent a parallel cutter

Do **not** assign “figure out how to split the model” to core-designers.
Do **not** require full-model green before the cut.

### 4. Chip skeleton

Create `examples/chips/<chip>/`:

- `chip.toml` (`topology`, `compilerCore`, `runtime.*`)
- `configs/` tile with `sharedMem` + `[[cores]]` includes into `examples/cores/<name>/`
- `arch/.../CustomConfigs.scala` + `sims/{verilator,p2e}/TargetConfigs.scala`
- `workloads/` (contracts + later binaries), `regression/batch/{bemu,verilator,p2e}/`, optional `kernel/`
- Multi-core: chip `emu/` wrapping a core bemu (see goban/poly)

Stub core dirs so includes resolve; mark stubs clearly. Core-designers replace stubs.

### 5. Dispatch

One brief per core type: mission, locked contracts, DoD, non-goals. Point at slice contracts.

### 6. Integrate

Order: slice unit green → pairwise sharedMem → E2E. Grow regression TOML tests as binaries appear; exclude unsupported in batch TOML only (do not delete CMake targets).

## Contract template

```toml
[slice]
id = "enc_0"
subgraph = "sg_encoder"
core = "audio-encoder"
instance = 0

[io.in]
name = "mel"
dtype = "f32"
shape = [1, 80, 3000]
sharedmem_region = "mel"

[io.out]
name = "enc_act_01"
dtype = "f32"
shape = [1, 1500, 512]
sharedmem_region = "enc_act_01"

[policy]
mismatch = "error"
```

## Reference package

`examples/chips/audio/` — realtime Whisper-base ASR lead package (`SLICE_MAP`, `dispatch/`, stub cores).

## Anti-patterns

- Leaving stage/replica cuts as “open for core-designers”
- Chasing full Whisper/LLM E2E before subgraphs exist
- Reusing poly `prefill`/`decode` names for non-LLM pipelines
- Editing registration and Ball implementation in the same change without need
- Hiding shape mismatches with defaults
