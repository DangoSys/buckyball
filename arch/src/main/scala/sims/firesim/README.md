# FireSim Simulation Configuration

## Overview

This directory contains BuckyBall system simulation configuration for the FireSim platform. FireSim is an open-source FPGA-based simulation platform that provides hardware simulation environments, supporting system-level simulation and performance analysis.

## File Structure

```
firesim/
└── TargetConfigs.scala  - FireSim target configuration
```

## Configuration Description

### TargetConfigs.scala

This file defines BuckyBall system configuration for the FireSim platform:

**WithBootROM Configuration**:
```scala
class WithBootROM extends Config((site, here, up) => {
  case BootROMLocated(x) => {
    // Automatically select BootROM path
    val chipyardBootROM = new File("./thirdparty/chipyard/generators/testchipip/bootrom/bootrom.rv${MaxXLen}.img")
    val firesimBootROM = new File("./thirdparty/chipyard/target-rtl/chipyard/generators/testchipip/bootrom/bootrom.rv${MaxXLen}.img")

    // Prefer chipyard path, use firesim path if it doesn't exist
    val bootROMPath = if (chipyardBootROM.exists()) {
      chipyardBootROM.getAbsolutePath()
    } else {
      firesimBootROM.getAbsolutePath()
    }
  }
})
```

**FireSimBuckyballToyConfig Configuration**:
```scala
class FireSimBuckyballToyConfig extends Config(
  new WithBootROM ++                              // BootROM configuration
  new firechip.chip.WithDefaultFireSimBridges ++ // Default FireSim bridges
  new firechip.chip.WithFireSimConfigTweaks ++   // FireSim configuration tweaks
  new examples.toy.BuckyBallToyConfig            // BuckyBall toy configuration
)
```

### Advanced Configuration

**Custom BootROM**:
```scala
class MyFireSimConfig extends Config(
  new WithBootROM ++
  new MyCustomBuckyBallConfig ++
  // Other configurations...
)
```
