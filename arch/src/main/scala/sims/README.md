# 仿真配置模块

该目录包含了 BuckyBall 项目中各种仿真器的配置和接口实现，为不同的仿真环境提供统一的配置管理。

## 目录结构

```
sims/
├── firesim/
│   └── TargetConfigs.scala    - FireSim FPGA仿真配置
└── verilator/
    └── Elaborate.scala        - Verilator仿真顶层生成
```

## FireSim 配置 (firesim/)

### TargetConfigs.scala

定义了在 FireSim FPGA 平台上运行的目标配置：

```scala
class FireSimBuckyBallConfig extends Config(
  new WithDefaultFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new BuckyBallConfig
)
```

**关键配置项**:
- **Bridge 配置**: UART、BlockDevice、NIC 等 I/O 桥接
- **内存模型**: DDR3/DDR4 内存控制器配置
- **时钟域**: 多时钟域管理和时钟生成
- **调试接口**: JTAG 和 Debug Module 配置

**使用场景**:
- 大规模系统仿真
- 长时间运行的工作负载测试
- 多核系统性能评估
- I/O 密集型应用验证

## Verilator 配置 (verilator/)

### Elaborate.scala

Verilator 仿真的顶层模块生成器：

```scala
object Elaborate extends App {
  val config = new BuckyBallConfig
  val gen = () => LazyModule(new BuckyBallSystem()(config)).module

  (new ChiselStage).execute(args, Seq(
    ChiselGeneratorAnnotation(gen),
    TargetDirAnnotation("generated-src")
  ))
}
```

**生成流程**:
1. 解析命令行参数和配置
2. 实例化 BuckyBall 系统模块
3. 生成 Verilog RTL 代码
4. 输出仿真所需的辅助文件

**输出文件**:
- `BuckyBallSystem.v` - 主 Verilog 文件
- `BuckyBallSystem.anno.json` - FIRRTL 注解文件
- `BuckyBallSystem.fir` - FIRRTL 中间表示

## 配置参数化

### 通用参数
```scala
// 处理器核心配置
case object RocketTilesKey extends Field[Seq[RocketTileParams]]

// 内存系统配置
case object MemoryBusKey extends Field[MemoryBusParams]

// 外设配置
case object PeripheryBusKey extends Field[PeripheryBusParams]
```

### 仿真特定参数
```scala
// Verilator 仿真参数
case object VerilatorDRAMKey extends Field[Boolean](false)

// FireSim 仿真参数
case object FireSimBridgesKey extends Field[Seq[BridgeIOAnnotation]]
```

## 构建和使用

### Verilator 仿真构建
```bash
# 生成 Verilog
cd arch
mill arch.runMain sims.verilator.Elaborate

# 编译仿真器
cd generated-src
verilator --cc BuckyBallSystem.v --exe sim_main.cpp
make -C obj_dir -f VBuckyBallSystem.mk
```

### FireSim 仿真部署
```bash
# 配置 FireSim 环境
cd firesim
source sourceme-f1-manager.sh

# 构建 FPGA 镜像
firesim buildbitstream

# 运行仿真
firesim runworkload
```

## 调试和优化

### Verilator 调试
- **波形生成**: 使用 `--trace` 选项生成 VCD 文件
- **性能分析**: 使用 `--prof-cfuncs` 进行性能剖析
- **覆盖率**: 使用 `--coverage` 生成覆盖率报告

### FireSim 调试
- **Printf 调试**: 使用 `printf` 语句输出调试信息
- **断言检查**: 启用运行时断言验证
- **性能计数器**: 集成 HPM 计数器监控

## 扩展开发

### 添加新仿真器支持
1. 创建新的配置目录 (如 `vcs/`)
2. 实现仿真器特定的配置类
3. 添加构建脚本和 Makefile
4. 更新文档和测试用例

### 自定义配置
```scala
class MyCustomConfig extends Config(
  new WithMyCustomParameters ++
  new BuckyBallConfig
)
```

## 相关文档
- [架构概览](../README.md)
- [Verilator 工作流](../../../../workflow/steps/verilator/README.md)
- [测试框架](../../../../bb-tests/README.md)
