# BuckyBall 框架核心

## 概述

该目录包含了 BuckyBall 框架的核心实现，是整个硬件架构的基础层。目录位于 `arch/src/main/scala/framework` 下，在系统架构中作为核心框架层，提供处理器核心、内置组件和系统互连的完整实现。

主要功能模块包括：
- **rocket**: 基于Rocket-chip的处理器核心定制实现
- **builtin**: 内置硬件组件库，包含内存域、前端等模块
- **blink**: 系统互连和通信框架

## 代码结构

```
framework/
├── rocket/           - Rocket核心定制实现
├── builtin/          - 内置组件库
│   ├── memdomain/    - 内存域实现
│   ├── frontend/     - 前端组件
│   └── util/         - 框架工具函数
└── blink/            - 系统互连框架
```

### 文件依赖关系

**rocket/** (处理器核心层)
- 实现BuckyBall定制的Rocket处理器核心
- 扩展标准Rocket-chip功能
- 提供RoCC协处理器接口

**builtin/** (内置组件层)
- 提供标准化的硬件组件实现
- 包含内存子系统、前端处理等模块
- 为上层应用提供基础硬件抽象

**blink/** (互连层)
- 实现系统级互连和通信协议
- 提供总线仲裁和路由功能
- 支持多核和多域通信

### 数据流向

```
应用层 → rocket核心 → builtin组件 → blink互连 → 物理接口
         ↓           ↓            ↓
    RoCC协处理器  内存域组件   系统总线
```

## 模块说明

### rocket/ - Rocket核心实现

**主要功能**: 提供BuckyBall定制的RISC-V处理器核心

**关键特性**:
- 基于Berkeley Rocket-chip架构
- 扩展RoCC协处理器接口
- 支持自定义指令和CSR
- 集成BuckyBall特有的功能扩展

**核心文件**:
- `RocketCoreBB.scala`: BuckyBall版本的Rocket核心
- `RocketTileBB.scala`: 处理器Tile实现
- `LazyRoCCBB.scala`: RoCC协处理器框架
- `CSRBB.scala`: 控制状态寄存器扩展

### builtin/ - 内置组件库

**主要功能**: 提供标准化的硬件组件实现

**组件分类**:
- **memdomain**: 内存域组件，包含存储器和DMA引擎
- **frontend**: 前端处理组件
- **util**: 框架级工具函数

**设计特点**:
- 模块化和可配置设计
- 标准化的接口定义
- 支持参数化实例化

### blink/ - 系统互连

**主要功能**: 实现系统级互连和通信协议

**关键组件**:
- `blink.scala`: 互连协议实现
- `bbus.scala`: 系统总线定义
- `ball.scala`: 球域通信机制

**互连特性**:
- 支持多种总线协议
- 提供仲裁和路由功能
- 延迟和带宽管理

## 使用方法

### 框架集成

**配置系统**:
```scala
class BuckyBallConfig extends Config(
  new WithBuckyBallRocket ++
  new WithBuiltinComponents ++
  new WithBlinkInterconnect ++
  new BaseConfig
)
```

**模块实例化**:
```scala
class BuckyBallSystem(implicit p: Parameters) extends LazyModule {
  val rocket = LazyModule(new RocketTileBB)
  val memdomain = LazyModule(new MemDomain)
  val interconnect = LazyModule(new BlinkInterconnect)

  // 连接各模块
  interconnect.node := rocket.masterNode
  memdomain.node := interconnect.slaveNode
}
```

### 扩展开发

**添加新组件**:
1. 在builtin目录下创建新的组件模块
2. 实现标准的LazyModule接口
3. 在配置系统中注册新组件
4. 更新互连和路由逻辑

**自定义处理器**:
1. 扩展RocketCoreBB实现
2. 添加自定义指令解码
3. 实现相应的执行单元
4. 更新CSR和异常处理

### 注意事项

1. **参数传递**: 使用Chipyard的Parameters系统进行配置传递
2. **时钟域**: 注意不同组件间的时钟域crossing
3. **复位策略**: 确保各模块的复位顺序和依赖关系
4. **性能优化**: 关注关键路径和时序约束
5. **调试支持**: 集成必要的调试和监控接口

## 相关文档

- [Blink 互连系统](blink/README.md) - 系统互连实现
- [内置组件库](builtin/README.md) - 标准硬件组件
- [Rocket 核心扩展](rocket/README.md) - 处理器核心实现
- [BuckyBall 源码概览](../README.md) - 上层架构说明
