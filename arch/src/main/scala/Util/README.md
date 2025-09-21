# BuckyBall 工具函数库

## 概述

该目录包含了 BuckyBall 框架中的通用工具函数和辅助模块，主要提供可复用的硬件设计组件。目录位于 `arch/src/main/scala/Util` 下，在整个架构中作为基础工具层，为其他模块提供通用的硬件构建块。

主要功能包括：
- **Pipeline**: 流水线控制和管理工具
- 通用的硬件设计模式实现

## 代码结构

```
Util/
└── Pipeline.scala    - 流水线控制实现
```

### 文件依赖关系

**Pipeline.scala** (基础工具层)
- 提供通用的流水线控制逻辑
- 被其他需要流水线功能的模块引用
- 实现标准的流水线接口和控制信号

## 模块说明

### Pipeline.scala

**主要功能**: 提供通用的流水线控制和管理功能

**关键组件**:

```scala
class Pipeline extends Module {
  val io = IO(new Bundle {
    val flush = Input(Bool())
    val stall = Input(Bool())
    val valid_in = Input(Bool())
    val ready_out = Output(Bool())
    val valid_out = Output(Bool())
  })

  // 流水线控制逻辑
  val pipeline_valid = RegInit(false.B)

  when(io.flush) {
    pipeline_valid := false.B
  }.elsewhen(!io.stall) {
    pipeline_valid := io.valid_in
  }

  io.ready_out := !io.stall
  io.valid_out := pipeline_valid && !io.flush
}
```

**流水线控制信号**:
- **flush**: 流水线冲刷信号，清空所有流水线级
- **stall**: 流水线暂停信号，保持当前状态
- **valid_in**: 输入数据有效信号
- **ready_out**: 准备接收新数据信号
- **valid_out**: 输出数据有效信号

**输入输出**:
- 输入: 控制信号(flush, stall)和数据有效信号
- 输出: 流水线状态和数据有效指示
- 边缘情况: flush优先级高于stall，确保正确的流水线行为

**依赖项**: Chisel3基础库，标准的Module和Bundle接口

## 使用方法

### 使用方法

**集成流水线控制**:
```scala
class MyModule extends Module {
  val pipeline = Module(new Pipeline)

  // 连接控制信号
  pipeline.io.flush := flush_condition
  pipeline.io.stall := stall_condition
  pipeline.io.valid_in := input_valid

  // 使用流水线输出
  val output_enable = pipeline.io.valid_out
}
```

### 设计模式

**流水线级联**:
- 支持多级流水线的级联连接
- 提供标准的ready/valid握手协议
- 确保数据流的正确性和时序

**背压处理**:
- 实现标准的背压传播机制
- 支持上游模块的暂停和恢复
- 保证数据不丢失和不重复

### 注意事项

1. **时序约束**: flush信号应该在时钟上升沿同步断言
2. **复位行为**: 流水线在复位时应该清空所有有效位
3. **组合逻辑**: ready信号是组合逻辑，避免时序路径问题
4. **扩展性**: 设计支持参数化的流水线深度和数据宽度
