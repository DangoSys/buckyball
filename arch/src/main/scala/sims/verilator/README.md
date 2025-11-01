# Verilator Simulation Configuration

## Overview

This directory contains BuckyBall system simulation configuration for the Verilator platform. Verilator is an open-source Verilog/SystemVerilog simulator that compiles RTL code into high-performance C++ simulation models, providing a fast functional simulation and verification environment.

## File Structure

```
verilator/
└── Elaborate.scala  - Verilator elaboration configuration
```

## Core Implementation

### Elaborate.scala

This file implements the Verilog generation and elaboration process for the BuckyBall system:

```scala
object Elaborate extends App {
  val config = new examples.toy.BuckyBallToyConfig
  val params = config.toInstance

  ChiselStage.emitSystemVerilogFile(
    new chipyard.harness.TestHarness()(config.toInstance),
    firtoolOpts = args,
    args = Array.empty  // Pass command line arguments directly
  )
}
```
