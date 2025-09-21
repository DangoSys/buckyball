# VecBall - 向量处理球

## 概述

VecBall 是 BuckyBall 系统中的向量处理加速器封装，它将向量处理单元（VecUnit）集成到 Ball 域架构中。VecBall 通过 Diplomacy 协议与系统的存储子系统进行连接，提供高性能的向量计算能力。

## 二、文件结构

```
vecball/
└── VecBall.scala  - VecBall 封装实现
```

## 三、核心功能

### VecBall LazyModule

VecBall 是一个 LazyModule，负责：
- 封装 VecUnit 核心向量处理逻辑
- 通过 Diplomacy 协议协商存储带宽
- 连接 scratchpad 和 accumulator 存储器
- 提供标准的 Ball 域接口

**主要特性**：
- **存储带宽协商**: 自动请求所需的 scratchpad 和 accumulator 带宽
- **灵活连接**: 支持多 bank 的存储器连接
- **标准接口**: 提供统一的命令请求/响应接口

## 四、架构设计

### Diplomacy 节点配置

```scala
val node = new BallNode(Seq(BBusParams(
  sramReadBW = b.sp_banks + b.acc_banks,   // 读带宽需求
  sramWriteBW = b.sp_banks + b.acc_banks   // 写带宽需求
)))
```

VecBall 请求的带宽包括：
- **scratchpad banks**: `b.sp_banks` 个读写端口
- **accumulator banks**: `b.acc_banks` 个读写端口

### 存储器连接策略

**Scratchpad 连接**：
```scala
// 前 b.sp_banks 个端口连接到 scratchpad
for (i <- 0 until b.sp_banks) {
  bundle.data.sramRead(i) <> vecUnit.io.sramRead(i)
  bundle.data.sramWrite(i) <> vecUnit.io.sramWrite(i)
}
```

**Accumulator 连接**：
```scala
// 接下来 b.acc_banks 个端口连接到 accumulator
for (i <- 0 until b.acc_banks) {
  val readIdx = b.sp_banks + i
  val writeIdx = b.sp_banks + i
  bundle.data.sramRead(readIdx) <> vecUnit.io.accRead(i)
  bundle.data.sramWrite(writeIdx) <> vecUnit.io.accWrite(i)
}
```

### 接口设计

```scala
val io = IO(new Bundle {
  val cmdReq = Flipped(Decoupled(new BallRsIssue))     // 命令请求输入
  val cmdResp = Decoupled(new BallRsComplete)          // 命令响应输出
})
```

## 五、工作流程

### 初始化阶段
1. **带宽协商**: 通过 Diplomacy 协议向系统请求所需的存储带宽
2. **参数验证**: 检查协商后的带宽是否满足 VecUnit 的需求
3. **连接建立**: 将 VecUnit 的存储接口连接到系统存储器

### 运行阶段
1. **命令接收**: 从保留站接收向量处理命令
2. **向量执行**: VecUnit 执行向量计算操作
3. **存储访问**: 通过 scratchpad 和 accumulator 进行数据读写
4. **完成响应**: 向保留站发送执行完成信号

### 带宽管理
```scala
// 检查协商后的参数
val negotiatedParams = node.edges.out.map(e => (e.sramReadBW, e.sramWriteBW))
require(negotiatedParams.forall(p =>
  p._1 >= (b.sp_banks + b.acc_banks) &&
  p._2 >= (b.sp_banks + b.acc_banks)),
  "negotiated bandwidth must support VecUnit requirements")
```

## 六、配置参数

### 关键配置项
- **sp_banks**: Scratchpad bank 数量
- **acc_banks**: Accumulator bank 数量
- **向量长度**: VecUnit 支持的向量长度
- **数据位宽**: 向量元素的位宽

### 带宽计算
总带宽需求 = scratchpad banks + accumulator banks
- 读带宽: `b.sp_banks + b.acc_banks`
- 写带宽: `b.sp_banks + b.acc_banks`

## 七、存储器访问模式

### Scratchpad 访问
- **用途**: 存储输入向量数据和中间结果
- **访问模式**: 支持随机访问和顺序访问
- **bank 数量**: 可配置，影响并行访问能力

### Accumulator 访问
- **用途**: 存储累加结果和最终输出
- **访问模式**: 主要用于累加操作
- **bank 数量**: 通常较少，专门用于累加计算

## 八、性能特性

### 并行处理能力
- **多 bank 并行**: 支持多个存储 bank 的并行访问
- **向量并行**: VecUnit 内部的 SIMD 并行处理
- **流水线执行**: 支持指令级流水线

### 带宽优化
- **智能分配**: 根据实际需求动态分配带宽
- **冗余处理**: 自动处理多余的端口连接
- **错误检查**: 验证带宽协商结果

## 九、使用示例

### 基本配置
```scala
// 配置 VecBall 参数
class MyBuckyBallConfig extends CustomBuckyBallConfig {
  override val sp_banks = 4    // 4 个 scratchpad banks
  override val acc_banks = 2   // 2 个 accumulator banks
}

// 实例化 VecBall
val vecBall = LazyModule(new VecBall)
```

### 系统集成
```scala
// 在 Ball 域中集成 VecBall
class BallDomain extends LazyModule {
  val vecBall = LazyModule(new VecBall)
  val memorySystem = LazyModule(new MemorySystem)

  // 通过 Diplomacy 连接
  memorySystem.node := vecBall.node
}
```

## 十、调试和监控

### 状态监控
- **命令队列状态**: 监控命令请求和响应的状态
- **存储器利用率**: 监控各个 bank 的使用情况
- **带宽利用率**: 监控实际带宽使用情况

### 性能分析
- **吞吐量**: 每秒处理的向量操作数
- **延迟**: 单个向量操作的完成时间
- **资源利用率**: 各个组件的利用率统计

## 十一、扩展和优化

### 功能扩展
- **多 VecUnit 支持**: 支持多个 VecUnit 实例
- **参数配置**: 向量处理参数配置
- **错误处理**: 增强的错误检测和恢复机制

### 性能优化
- **缓存优化**: 改进数据缓存策略
- **预取机制**: 实现数据预取提高性能
- **负载均衡**: 在多个处理单元间平衡负载

## 十二、相关文档

- [Ball域概览](../README.md)
- [保留站和ROB](../rs/README.md)
- [向量处理单元](../../../prototype/vector/README.md)
- [Blink通信框架](../../../framework/blink/README.md)
