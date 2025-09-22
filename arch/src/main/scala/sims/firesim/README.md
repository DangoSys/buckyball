# FireSim 仿真配置

## 概述

该目录包含了 BuckyBall 系统在 FireSim 平台上的仿真配置。FireSim 是一个基于 FPGA 的开源仿真平台，提供硬件仿真环境，支持系统级仿真和性能分析。

## 二、文件结构

```
firesim/
└── TargetConfigs.scala  - FireSim 目标配置
```

## 三、配置说明

### TargetConfigs.scala

该文件定义了 BuckyBall 系统在 FireSim 平台上的配置：

**WithBootROM 配置**：
```scala
class WithBootROM extends Config((site, here, up) => {
  case BootROMLocated(x) => {
    // 自动选择 BootROM 路径
    val chipyardBootROM = new File("./thirdparty/chipyard/generators/testchipip/bootrom/bootrom.rv${MaxXLen}.img")
    val firesimBootROM = new File("./thirdparty/chipyard/target-rtl/chipyard/generators/testchipip/bootrom/bootrom.rv${MaxXLen}.img")

    // 优先使用 chipyard 路径，如果不存在则使用 firesim 路径
    val bootROMPath = if (chipyardBootROM.exists()) {
      chipyardBootROM.getAbsolutePath()
    } else {
      firesimBootROM.getAbsolutePath()
    }
  }
})
```

**FireSimBuckyballToyConfig 配置**：
```scala
class FireSimBuckyballToyConfig extends Config(
  new WithBootROM ++                              // BootROM 配置
  new firechip.chip.WithDefaultFireSimBridges ++ // 默认 FireSim 桥接
  new firechip.chip.WithFireSimConfigTweaks ++   // FireSim 配置调整
  new examples.toy.BuckyBallToyConfig            // BuckyBall 玩具配置
)
```

### 高级配置

**自定义 BootROM**：
```scala
class MyFireSimConfig extends Config(
  new WithBootROM ++
  new MyCustomBuckyBallConfig ++
  // 其他配置...
)
```
