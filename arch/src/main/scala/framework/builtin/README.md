# BuckyBall 内置组件库

## 概述

该目录包含了 BuckyBall 框架的内置硬件组件实现，提供标准化的可复用硬件模块。位于 `arch/src/main/scala/framework/builtin` 下，作为组件库层，为上层系统提供经过验证的硬件构建块。

主要组件模块：
- **memdomain**: 内存域组件，包含存储器和DMA引擎
- **frontend**: 前端处理组件
- **util**: 框架级工具函数
- **BaseConfigs.scala**: 基础配置定义

## 代码结构

```
builtin/
├── BaseConfigs.scala - 基础配置参数定义
├── memdomain/        - 内存域实现
├── frontend/         - 前端组件
└── util/             - 工具函数库
```

### 文件依赖关系

**BaseConfigs.scala** (配置基础层)
- 定义所有内置组件的基础配置参数
- 提供默认配置和参数验证
- 被所有子模块引用作为配置源

**memdomain/** (内存子系统)
- 依赖 BaseConfigs 获取内存相关配置
- 实现存储器、DMA、地址管理等功能
- 为其他组件提供内存访问服务

**frontend/** (前端处理)
- 使用 BaseConfigs 中的前端配置参数
- 实现指令获取、解码等前端功能
- 与处理器核心紧密集成

**util/** (工具库)
- 提供通用的硬件设计工具函数
- 被其他组件广泛使用
- 独立于具体配置参数

## 模块说明

### BaseConfigs.scala

**主要功能**: 定义内置组件的基础配置参数和默认值

**关键组件**:

```scala
trait BaseConfig extends Config {
  // 内存域配置
  case object MemDomainKey extends Field[MemDomainParams](MemDomainParams())

  // 前端配置
  case object FrontendKey extends Field[FrontendParams](FrontendParams())

  // 工具配置
  case object UtilKey extends Field[UtilParams](UtilParams())
}

case class MemDomainParams(
  spBanks: Int = 16,           // 暂存器Bank数量
  spBankEntries: Int = 1024,   // 每个Bank条目数
  accBanks: Int = 4,           // 累加器Bank数量
  accBankEntries: Int = 256,   // 累加器Bank条目数
  dmaEngines: Int = 2          // DMA引擎数量
)
```

**配置参数**:
- **MemDomainParams**: 内存域相关参数
- **FrontendParams**: 前端处理参数
- **UtilParams**: 工具函数参数

**参数验证**:
```scala
require(spBanks > 0, "SP banks must be positive")
require(isPow2(spBankEntries), "SP bank entries must be power of 2")
require(accBanks > 0, "ACC banks must be positive")
```

**输入输出**:
- 输入: 用户自定义配置覆盖
- 输出: 验证后的完整配置参数
- 边缘情况: 参数冲突检测和错误报告

### memdomain/ 子模块

**主要功能**: 实现完整的内存域功能

**包含组件**:
- **mem/**: 存储器组件(SramBank, AccBank, Scratchpad)
- **dma/**: DMA引擎(BBStreamReader, BBStreamWriter, LocalAddr)

**接口定义**:
```scala
class MemDomainIO(implicit p: Parameters) extends Bundle {
  val exec = new Bundle {
    val cmd = Flipped(Decoupled(new MemCmd))
    val resp = Decoupled(new MemResp)
  }
  val dma = new Bundle {
    val read = Decoupled(new DMAReadReq)
    val write = Flipped(Decoupled(new DMAWriteReq))
  }
}
```

### frontend/ 子模块

**主要功能**: 实现处理器前端功能

**核心功能**:
- 指令获取和缓存
- 分支预测和跳转处理
- 指令解码预处理

### util/ 子模块

**主要功能**: 提供通用工具函数

**工具类别**:
- 数学运算工具
- 接口转换工具
- 调试和监控工具

## 使用方法

### 配置使用

**基础配置继承**:
```scala
class MySystemConfig extends Config(
  new BaseConfig ++
  new WithCustomMemDomain(spBanks = 32) ++
  new WithCustomFrontend(icacheWays = 8)
)
```

**参数访问**:
```scala
class MyModule(implicit p: Parameters) extends Module {
  val memParams = p(MemDomainKey)
  val spBanks = memParams.spBanks
  val accBanks = memParams.accBanks
}
```

### 扩展开发

**添加新组件**:
1. 在相应子目录创建新模块
2. 在 BaseConfigs.scala 中添加配置参数
3. 实现标准的 LazyModule 接口
4. 添加相应的测试用例

**自定义配置**:
```scala
case class MyComponentParams(
  param1: Int = 16,
  param2: Boolean = true
)

trait WithMyComponent extends Config {
  case object MyComponentKey extends Field[MyComponentParams](MyComponentParams())
}
```

### 注意事项

1. **配置一致性**: 确保相关组件的配置参数兼容
2. **资源约束**: 注意硬件资源的合理分配
3. **时序优化**: 关注跨组件的时序路径
4. **接口标准**: 遵循统一的接口设计规范
5. **测试覆盖**: 为每个组件提供充分的测试用例
