# FireSim Workflow

FireSim FPGA simulation. All commands take `--chip`; mill class comes from
`examples/chips/<chip>/configs/chip.toml` `[sims].firesim`.

Manager YAMLs are generated per chip under `scripts/yaml/<chip>/`.

## Commands

```bash
bbdev firesim --enumeratefpgas '--chip toy'
bbdev firesim --buildbitstream '--chip toy'
bbdev firesim --infrasetup '--chip toy'
bbdev firesim --runworkload '--chip toy'
```

`buildbitstream` takes hours. `infrasetup` / `runworkload` require a prior
bitstream under `thirdparty/firesim/deploy/results-build` for that chip recipe.
