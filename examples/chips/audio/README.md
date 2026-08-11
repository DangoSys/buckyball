# Chip `audio` (lead package)

Realtime Whisper-base ASR tile. Chip lead owns topology + graph cut; core-designers own cores.

## Topology

`dap×1 + audio-encoder×3 + audio-decoder×2` on one tile with `sharedMem`.

## Start here

| Doc | Purpose |
|-----|---------|
| `docs/superpowers/specs/2026-08-10-audio-chip-design.md` | Design + ownership |
| `workloads/subgraphs/SLICE_MAP.md` | Stage + homogeneous cut |
| `dispatch/*.md` | Per-core designer briefs |
| `workloads/slices/*/contract.toml` | Hard I/O contracts |

## Configs

- Verilator: `sims.verilator.BuckyballAudioVerilatorConfig`
- P2E: `sims.p2e.P2EAudioConfig` / `P2EAudioLinuxConfig`
- `compilerCore`: `audio-encoder`

## Status

Lead scaffold landed. Core balldomains/compilers/emus are **stubs**. Regression test lists are empty until cores deliver slice binaries. Full Whisper E2E is an integration milestone after slices are green.
