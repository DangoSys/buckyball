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

## 四、功能特性

### BootROM 管理
- **自动路径检测**: 自动检测并选择正确的 BootROM 文件路径
- **多路径支持**: 支持 Chipyard 和 FireSim 两种部署路径
- **架构适配**: 根据目标架构（RV32/RV64）选择对应的 BootROM

### FireSim 集成
- **默认桥接**: 集成 FireSim 的默认桥接组件
- **配置优化**: 应用 FireSim 特定的配置调整
- **系统兼容**: 确保与 FireSim 仿真环境的兼容性

## 五、使用方法

### 基本仿真流程

1. **环境准备**：
```bash
# 确保 FireSim 环境已正确配置
cd firesim
source sourceme-f1-manager.sh
```

2. **配置选择**：
```scala
// 在 FireSim 配置文件中使用
TARGET_CONFIG = FireSimBuckyballToyConfig
```

3. **构建目标**：
```bash
# 构建 FireSim 目标
make -C sims/firesim
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

**性能调优**：
```scala
class OptimizedFireSimConfig extends Config(
  new WithBootROM ++
  new firechip.chip.WithDefaultFireSimBridges ++
  new firechip.chip.WithFireSimConfigTweaks ++
  new WithOptimizedBuckyBall ++
  new examples.toy.BuckyBallToyConfig
)
```

## 六、仿真特性

### FPGA 加速
- **高性能仿真**: 利用 FPGA 硬件加速仿真执行
- **周期精确**: 提供周期精确的仿真结果
- **大规模支持**: 支持大规模系统的仿真

### 调试支持
- **波形捕获**: 支持信号波形的捕获和分析
- **性能监控**: 提供详细的性能统计信息
- **断点调试**: 支持硬件断点和调试功能

### 系统级仿真
- **完整系统**: 仿真包括处理器、内存、I/O 的完整系统
- **操作系统支持**: 支持运行完整的操作系统
- **网络仿真**: 支持网络功能的仿真

## 七、配置参数

### 关键参数
- **MaxXLen**: 目标架构位宽（32 或 64）
- **BootROM 路径**: BootROM 文件的位置
- **桥接配置**: FireSim 桥接组件的配置

### 性能参数
- **时钟频率**: 目标时钟频率设置
- **内存配置**: 内存大小和延迟配置
- **I/O 配置**: 外设和 I/O 接口配置

## 八、故障排除

### 常见问题

**BootROM 文件未找到**：
```
解决方案：
1. 检查 Chipyard 是否正确安装
2. 确认 BootROM 文件路径是否正确
3. 手动指定 BootROM 文件路径
```

**FireSim 桥接错误**：
```
解决方案：
1. 确认 FireSim 版本兼容性
2. 检查桥接配置是否正确
3. 更新 FireSim 到最新版本
```

**仿真性能问题**：
```
解决方案：
1. 调整 FPGA 时钟频率
2. 优化设计复杂度
3. 使用性能分析工具定位瓶颈
```

## 九、性能优化

### 仿真加速
- **并行化**: 利用多个 FPGA 进行并行仿真
- **流水线优化**: 优化设计的流水线深度
- **资源平衡**: 平衡 FPGA 资源使用

### 调试效率
- **选择性监控**: 只监控关键信号减少开销
- **分层调试**: 采用分层的调试策略
- **自动化测试**: 使用自动化测试脚本

## 十、扩展开发

### 添加新配置
```scala
class MyBuckyBallFireSimConfig extends Config(
  new WithBootROM ++
  new WithMyCustomFeatures ++
  new firechip.chip.WithDefaultFireSimBridges ++
  new firechip.chip.WithFireSimConfigTweaks ++
  new MyBuckyBallConfig
)
```

### 自定义桥接
```scala
class WithCustomBridges extends Config((site, here, up) => {
  // 添加自定义桥接配置
})
```

## 十一、相关文档

- [仿真环境概览](../README.md)
- [Verilator 仿真配置](../verilator/README.md)
- [BuckyBall 玩具配置](../../examples/toy/README.md)
- [FireSim 官方文档](https://docs.fires.im/)
