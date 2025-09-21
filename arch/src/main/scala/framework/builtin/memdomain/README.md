# 内存域 (MemDomain) 实现

## 概述

内存域模块是 BuckyBall 框架中负责内存管理和数据传输的核心组件。该模块位于 `framework/builtin/memdomain` 路径下，实现了高性能的内存子系统，包括内存控制器、DMA 引擎、TLB 管理和地址转换等功能。

内存域在整个系统架构中扮演关键角色，为上层的处理器核心和加速器提供统一的内存访问接口，同时管理片上存储器（scratchpad、accumulator）和外部内存之间的数据流动。

## 二、文件结构

```
memdomain/
├── MemDomain.scala         - 内存域顶层模块
├── MemController.scala     - 内存控制器
├── MemLoader.scala         - 内存加载器
├── MemStorer.scala         - 内存存储器
├── DomainDecoder.scala     - 域解码器
├── DISA.scala             - 地址空间管理
├── dma/                   - DMA 引擎实现
├── mem/                   - 存储器管理
├── rs/                    - 保留站实现
└── tlb/                   - TLB 管理
```

## 三、核心组件

### MemDomain - 内存域顶层模块

MemDomain 是内存域的顶层模块，负责：
- 集成所有内存子系统组件
- 提供统一的外部接口
- 管理 DMA 和 Ball 域的访问协调

**主要接口**：
```scala
class MemDomain extends Module {
  val io = IO(new Bundle {
    // 全局解码器输入
    val gDecoderIn = Flipped(Decoupled(new PostGDCmd))

    // Ball 域 SRAM 接口
    val ballDomain = new Bundle {
      val sramRead = Vec(sp_banks, new SramReadIO)
      val sramWrite = Vec(sp_banks, new SramWriteIO)
      val accRead = Vec(acc_banks, new SramReadIO)
      val accWrite = Vec(acc_banks, new SramWriteIO)
    }

    // DMA 接口
    val dma = new Bundle {
      val read = new Bundle {
        val req = Decoupled(new BBReadRequest)
        val resp = Flipped(Decoupled(new BBReadResponse))
      }
      val write = new Bundle {
        val req = Decoupled(new BBWriteRequest)
        val resp = Flipped(Decoupled(new BBWriteResponse))
      }
    }

    // TLB 接口
    val tlb = Vec(2, Flipped(new BBTLBIO))
    val ptw = Vec(2, new TLBPTWIO)
    val tlbExp = Vec(2, new BBTLBExceptionIO)
  })
}
```

### MemController - 内存控制器

MemController 封装了 scratchpad 和 accumulator 的控制逻辑：

**功能特性**：
- **双端口设计**: 同时支持 DMA 和 Ball 域访问
- **存储器抽象**: 提供统一的存储器访问接口
- **资源管理**: 管理 scratchpad 和 accumulator 资源

**接口设计**：
```scala
class MemController extends Module {
  val io = IO(new Bundle {
    // DMA 接口 - 用于 MemLoader 和 MemStorer
    val dma = new Bundle {
      val sramRead = Vec(sp_banks, new SramReadIO)
      val sramWrite = Vec(sp_banks, new SramWriteIO)
      val accRead = Vec(acc_banks, new SramReadIO)
      val accWrite = Vec(acc_banks, new SramWriteIO)
    }

    // Ball 域接口 - 用于 BallController
    val ballDomain = new Bundle {
      val sramRead = Vec(sp_banks, new SramReadIO)
      val sramWrite = Vec(sp_banks, new SramWriteIO)
      val accRead = Vec(acc_banks, new SramReadIO)
      val accWrite = Vec(acc_banks, new SramWriteIO)
    }
  })
}
```

### 存储器架构

**Scratchpad 存储器**：
- **用途**: 存储输入数据和中间结果
- **配置**: `b.sp_banks` 个 bank，每个 `b.spad_bank_entries` 条目
- **位宽**: `b.spad_w` 位数据位宽
- **访问模式**: 支持随机访问和顺序访问

**Accumulator 存储器**：
- **用途**: 存储累加结果和最终输出
- **配置**: `b.acc_banks` 个 bank，每个 `b.acc_bank_entries` 条目
- **位宽**: `b.acc_w` 位数据位宽
- **访问模式**: 主要用于累加操作

## 四、数据流架构

### 访问路径

```
外部内存 ←→ DMA引擎 ←→ MemController ←→ Scratchpad/Accumulator
                                    ↕
                              Ball域加速器
```

### 双端口访问机制

MemController 实现了双端口访问机制：
1. **DMA 端口**: 用于与外部内存的数据传输
2. **Ball 域端口**: 用于加速器的计算访问

```scala
// 连接示例
io.dma.sramRead <> spad.io.dma.sramread
io.ballDomain.sramRead <> spad.io.exec.sramread
```

## 五、子模块说明

### dma/ - DMA 引擎
实现高性能的直接内存访问：
- **BBStreamReader**: 流式数据读取器
- **BBStreamWriter**: 流式数据写入器
- **地址转换**: 集成 TLB 支持虚拟地址

### mem/ - 存储器管理
包含存储器的具体实现：
- **Scratchpad**: 片上暂存器实现
- **SRAM接口**: 标准化的存储器访问接口
- **Bank管理**: 多 bank 并行访问支持

### tlb/ - TLB 管理
提供虚拟地址转换服务：
- **BBTLBCluster**: TLB 集群管理器
- **地址转换**: 虚拟到物理地址映射
- **异常处理**: TLB 缺失和权限异常

### rs/ - 保留站
内存域专用的保留站实现：
- **指令调度**: 内存访问指令的调度
- **依赖管理**: 内存访问依赖关系管理

## 六、配置参数

### 关键配置项
```scala
class CustomBuckyBallConfig {
  val sp_banks = 4              // Scratchpad bank 数量
  val spad_bank_entries = 1024  // 每个 bank 的条目数
  val spad_w = 64              // Scratchpad 数据位宽
  val spad_mask_len = 8        // 掩码长度

  val acc_banks = 2            // Accumulator bank 数量
  val acc_bank_entries = 512   // 每个 bank 的条目数
  val acc_w = 64              // Accumulator 数据位宽
  val acc_mask_len = 8        // 掩码长度
}
```

### 性能调优
- **Bank 数量**: 影响并行访问能力
- **条目数量**: 影响存储容量
- **数据位宽**: 影响传输带宽
- **TLB 大小**: 影响地址转换性能

## 七、使用示例

### 基本配置
```scala
// 实例化内存域
implicit val config = new CustomBuckyBallConfig
val memDomain = Module(new MemDomain)

// 连接全局解码器
memDomain.io.gDecoderIn <> globalDecoder.io.memDomainOut

// 连接 Ball 域
ballController.io.sramRead <> memDomain.io.ballDomain.sramRead
ballController.io.sramWrite <> memDomain.io.ballDomain.sramWrite
```

### DMA 操作
```scala
// 配置 DMA 读取
memDomain.io.dma.read.req.valid := true.B
memDomain.io.dma.read.req.bits.addr := sourceAddress
memDomain.io.dma.read.req.bits.len := transferLength

// 处理 DMA 响应
when(memDomain.io.dma.read.resp.valid) {
  // 处理读取的数据
  val data = memDomain.io.dma.read.resp.bits.data
}
```

## 八、性能特性

### 并发访问
- **双端口设计**: DMA 和 Ball 域可并发访问
- **多 Bank 并行**: 支持多个 bank 的并行操作
- **流水线处理**: 实现深度流水线提高吞吐量

### 带宽优化
- **智能调度**: 优化内存访问调度
- **缓存策略**: 实现数据缓存和预取
- **带宽管理**: 动态分配访问带宽

## 九、调试和监控

### 状态监控
- **访问统计**: 监控各个 bank 的访问频率
- **带宽利用率**: 监控实际带宽使用情况
- **TLB 命中率**: 监控地址转换性能

### 性能分析
- **延迟统计**: 测量内存访问延迟
- **吞吐量监控**: 监控数据传输吞吐量
- **冲突检测**: 检测访问冲突和瓶颈

## 十、相关文档

- [DMA 引擎文档](dma/README.md)
- [存储器管理文档](mem/README.md)
- [TLB 管理文档](tlb/README.md)
- [内存域保留站文档](rs/README.md)
- [框架核心文档](../README.md)
