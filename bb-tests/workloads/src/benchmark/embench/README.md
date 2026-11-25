# Embench

## Directory Structure

```
bb-tests/workloads/src/benchmark/embench/
├── support/                 # Common support files
│   ├── top.c                # Runs every benchmark and checks its result: <design>-embench-baremetal
│   ├── beebsc.c/h           # Simplified libc (rand/malloc etc.)
│   ├── support.h            # Interface definitions (always includes boardsupport.h)
│   ├── boardsupport.c/h     # Buckyball adaptation (empty init + start/stop_trigger)
│   └── chipsupport.c/h      # Buckyball adaptation (empty init)
└── src/                     # registered benchmarks (see CMakeLists)
    ├── aha-mont64/  crc32/  edn/  huffbench/  matmult-int/
    ├── md5sum/  minver/  nbody/  nettle-aes/  nettle-sha256/  nsichneu/
    ├── picojpeg/  primecount/  qrduino/  sglib-combined/  slre/  st/
    └── statemate/  tarfind/  ud/  wikisort/
```
