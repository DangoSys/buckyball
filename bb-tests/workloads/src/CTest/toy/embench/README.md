# Embench

## Directory Structure

```
bb-tests/workloads/src/CTest/toy/embench/
├── support/                 # Common support files
│   ├── main.c               # Unified main (calls initialise_board → benchmark → verify)
│   ├── beebsc.c/h           # Simplified libc (rand/malloc etc.)
│   ├── support.h            # Interface definitions (always includes boardsupport.h)
│   ├── boardsupport.c/h     # Buckyball adaptation (empty init + start/stop_trigger)
│   └── chipsupport.c/h      # Buckyball adaptation (empty init)
└── src/                     # 19 benchmarks
    ├── aha-mont64/  crc32/  cubic/  edn/  huffbench/  matmult-int/
    ├── minver/  nbody/  nettle-aes/  nettle-sha256/  nsichneu/
    ├── picojpeg/  qrduino/  sglib-combined/  slre/  st/
    └── statemate/  ud/  wikisort/
```
