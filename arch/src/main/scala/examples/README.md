# BuckyBall 示例配置实现

## 概述

该目录包含了 BuckyBall 框架的示例配置和参考实现，用于展示如何配置和扩展 BuckyBall 系统。目录位于 `arch/src/main/scala/examples` 下，在整个架构中作为配置层，为开发者提供配置模板和系统实例。

主要功能包括：
- **BuckyBallConfig**: 全局配置参数定义
- **toy 系统**: 示例系统实现，包含自定义协处理器和 CSR 扩展

## 代码结构

```
examples/
├── BuckyBallConfig.scala  - 全局配置定义
└── toy/                   - 完整示例系统
    ├── balldomain/        - 球域组件实现
    ├── CSR.scala          - 自定义CSR实现
    ├── CustomConfigs.scala - 系统配置组合
    └── ToyBuckyBall.scala - 系统顶层模块
```

### 文件依赖关系

**BuckyBallConfig.scala** (基础配置层)
- 定义全局配置参数和默认值
- 被所有其他配置文件继承和扩展
- 提供系统级的配置接口

**toy/CustomConfigs.scala** (配置组合层)
- 继承 BuckyBallConfig 并添加自定义参数
- 组合多个配置片段形成完整配置
- 为 ToyBuckyBall 提供配置支持

**toy/ToyBuckyBall.scala** (系统实例化层)
- 使用 CustomConfigs 实例化完整系统
- 作为 mill 构建的入口点
- 生成最终的 Verilog 代码

## 模块说明

### BuckyBallConfig.scala

**主要功能**: 定义 BuckyBall 框架的全局配置参数

**关键组件**:

```scala
class BuckyBallConfig extends Config(
  new WithNBigCores(1) ++
  new WithRV32 ++
  new WithBuckyBallRoCC ++
  new WithL1ICacheWays(4) ++
  new WithL1DCacheWays(4) ++
  new BaseConfig
)
```

**配置参数**:
- **WithNBigCores(1)**: 配置单核 Rocket 处理器
- **WithRV32**: 使用 32 位 RISC-V 指令集
- **WithBuckyBallRoCC**: 启用 BuckyBall 自定义协处理器
- **WithL1ICacheWays(4)**: L1 指令缓存 4 路组相联
- **WithL1DCacheWays(4)**: L1 数据缓存 4 路组相联

**输入输出**:
- 输入: 无直接输入，通过配置系统传递参数
- 输出: 配置参数供其他模块使用
- 边缘情况: 配置冲突时按优先级覆盖

### toy/CustomConfigs.scala

**主要功能**: 组合多个配置片段，为 toy 系统提供完整配置

**关键组件**:

```scala
class ToyBuckyBallConfig extends Config(
  new WithToyBallDomain ++
  new WithCustomCSR ++
  new BuckyBallConfig
)
```

**配置组合**:
- **WithToyBallDomain**: 添加球域组件配置
- **WithCustomCSR**: 启用自定义 CSR 支持
- **BuckyBallConfig**: 继承基础配置

### toy/ToyBuckyBall.scala

**主要功能**: 系统顶层模块，实例化完整的 toy 系统

**关键组件**:

```scala
object ToyBuckyBall extends App {
  implicit val config = new ToyBuckyBallConfig

  val gen = () => LazyModule(new BuckyBallSystem).module

  (new ChiselStage).execute(args, Seq(
    ChiselGeneratorAnnotation(gen),
    TargetDirAnnotation("generated-src/toy")
  ))
}
```

**构建流程**:
1. 加载 ToyBuckyBallConfig 配置
2. 实例化 BuckyBallSystem LazyModule
3. 通过 ChiselStage 生成 Verilog
4. 输出到 generated-src/toy 目录

**输入输出**:
- 输入: 命令行参数 (args)
- 输出: Verilog 文件和相关构建产物
- 边缘情况: 配置错误时构建失败

### toy/CSR.scala

**主要功能**: 实现自定义控制状态寄存器

**关键组件**:

```scala
class CustomCSR extends Module {
  val io = IO(new Bundle {
    val csr_req = Flipped(Valid(new CSRReq))
    val csr_resp = Valid(new CSRResp)
  })

  // CSR 地址映射
  val custom_csr_addr = 0x800.U

  when(io.csr_req.valid && io.csr_req.bits.addr === custom_csr_addr) {
    // 处理自定义 CSR 读写
  }
}
```

**CSR 功能**:
- 扩展标准 RISC-V CSR 空间
- 提供自定义控制和状态接口
- 支持读写操作和权限检查

## 使用方法

### 使用方法

**构建 toy 系统**:
```bash
cd arch
mill arch.runMain examples.toy.ToyBuckyBall
```

**自定义配置开发**:
1. 复制 CustomConfigs.scala 作为模板
2. 修改配置参数满足需求
3. 实现必要的自定义组件
4. 更新顶层模块引用新配置

### 注意事项

1. **配置优先级**: 配置链中后面的配置会覆盖前面的同名参数
2. **依赖管理**: 确保自定义组件的依赖在配置中正确声明
3. **构建路径**: 生成的文件路径由 TargetDirAnnotation 指定
4. **参数验证**: 配置参数在实例化时进行验证，错误配置会导致构建失败
