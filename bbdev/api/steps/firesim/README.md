# FireSim Workflow

FireSim FPGA simulation. All commands take `--design`; mill class comes from
`examples/designs/<design>/configs/design.toml` `[sims].firesim`.

Manager YAMLs are generated per design under `scripts/yaml/<design>/`.

## Commands

```bash
bbdev firesim --enumeratefpgas '--design toy'
bbdev firesim --buildbitstream '--design toy'
bbdev firesim --infrasetup '--design toy'
bbdev firesim --runworkload '--design toy'
```

`buildbitstream` takes hours. `infrasetup` / `runworkload` require a prior
bitstream under `thirdparty/firesim/deploy/results-build` for that design recipe.
