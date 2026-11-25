# FireSim Workflow

FireSim FPGA simulation. All commands take `[--config <config TOML>] --device <device>` (or
`--hart <hart>`), a device without `--config` from its design's one `<Design>DefaultConfig.toml`; the mill class comes from the generated design.pb.

Manager YAMLs are generated per design under `scripts/yaml/<design>/`.

## Commands

```bash
bbdev firesim --enumeratefpgas '--device toy'
bbdev firesim --buildbitstream '--device toy'
bbdev firesim --infrasetup '--device toy'
bbdev firesim --runworkload '--device toy'
```

`buildbitstream` takes hours. `infrasetup` / `runworkload` require a prior
bitstream under `thirdparty/firesim/deploy/results-build` for that design recipe.
