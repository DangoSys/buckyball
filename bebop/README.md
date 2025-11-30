# bebop
A buckyball emulator written in Rust


### Quick start

1. Activate the virtual environment
```
source $BUCKYBALL_PATH/env.sh
```

<!-- 2. Build the simulator
```
./scripts/install.sh
``` -->

3. Start the socket server
```
./scripts/bebop_setup.sh
```

4. Run the program
```
$BUCKYBALL_PATH/bebop/host/spike/riscv-isa-sim/install/bin/spike --extension=bebop --log-commits $BUCKYBALL_PATH/bb-tests/build/workloads/src/OpTest/gemmini/transpose-baremetal 2>/dev/null
```
