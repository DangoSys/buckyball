# Cores

Each directory in this tree describes one concrete single-core design. A Core
owns its Rocket/frontend parameters, BallDomain, private MemDomain, and the
compiler package for its ISA. It does not own a tile topology, device model,
kernel, or regression suite.

Every Core has a `manifest.toml` and a `core.toml`. A Buckyball Core also has a
`compiler/` directory. Chip topology files instantiate Cores through relative
`include` paths; Chips own all tile-shared memory, runtime, and multi-core
behavior.
