# Cores

Each directory in this tree describes one concrete single-core design. A Core
owns its Rocket/frontend parameters, BallDomain, private MemDomain, and the
compiler package for its ISA. It does not own a tile topology, device model,
kernel, or regression suite.

Every Core has a `configs/` tree and `isa/` (`ballISA.h` from balldomain).
Buckyball Cores also have `compiler/` (buddy-mlir dialect plugin; CMake is the
C++ recipe, not the product entry) and non-empty `ballISA`. Rocket-only
cores keep `configs/default.toml` without a `balldomain` key so tiles stay
Rocket-only; empty `configs/balldomains/` + `isa/` still exist for Bazel /
`compilerCore` consumers. BEMU lives in bebop; chip-level tile runners are
`examples/chips/<chip>/emu/src/main.rs`. Ball functional models are
`examples/balls/<ball>/emu/`.

Chip topology files instantiate a concrete Core TOML through relative
`include` paths; that include is the config entry (not a fixed
`configs/default.toml`). Chips own tile-shared memory, runtime, and multi-core
behavior.
