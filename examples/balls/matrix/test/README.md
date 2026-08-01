# MatrixUnit throughput test

Run from `arch` with:

```text
sbt "testOnly examples.balls.matrix.MatrixUnitThroughputSpec"
```

The test always runs ten WS tasks followed by ten OS tasks. The OS batch is not
issued until every WS task has completed. Configuration is controlled through:

```text
MATRIX_TEST_SEED=97
MATRIX_WS_SHAPES=129x1x31,145x7x17,...
MATRIX_OS_SHAPES=17x33x19,31x17x32,...
MATRIX_MAX_CYCLES=200000
```

Each shape list must contain exactly ten `MxNxK` entries. The seed changes the
signed int8 contents of the virtual A/B banks. A passing run prints only the
seed, WS/OS cycle counts and A-row throughput. Failure messages identify the
first incorrect completion or C write transaction.
