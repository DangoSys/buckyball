# 前端处理组件

## 概述

该目录实现了 BuckyBall 的指令解码和调度功能，位于 `arch/src/main/scala/framework/builtin/frontend` 下，作为指令处理前端，负责指令的全局解码和后续调度管理。

实现的核心组件：
- **GlobalDecoder**: 全局指令解码器，区分指令类型
- **rs/**: 保留站相关组件，包含指令调度和重排序缓冲

## 代码结构

```
frontend/
├── GobalDecoder.scala    - 全局指令解码器
└── rs/                   - 保留站组件
    ├── CommitQueue.scala        - 提交队列
    ├── IssueQueue.scala         - 发射队列
    ├── NextROBIdCounter.scala   - ROB ID计数器
    ├── ReorderBuffer.scala      - 重排序缓冲
    └── ReservationStation.scala - 保留站
```

### 文件依赖关系

**GobalDecoder.scala** (指令解码层)
- 接收 RoCCCommandBB 指令
- 区分 Ball 指令和内存指令
- 输出 PostGDCmd 给后续处理

**rs/** (指令调度层)
- 依赖 GlobalDecoder 的输出
- 实现乱序执行的指令调度
- 管理指令的发射、执行和提交

## 模块说明

### GobalDecoder.scala

**主要功能**: 对 RoCC 指令进行全局解码和分类

**关键组件**:

```scala
class PostGDCmd(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val is_ball = Bool()    // Ball指令(包括FENCE)
  val is_mem = Bool()     // 内存指令(load/store)
  val raw_cmd = new RoCCCommandBB  // 原始指令信息
}

class GlobalDecoder(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val id_i = Flipped(Decoupled(new Bundle {
      val cmd = new RoCCCommandBB
    }))
    val id_o = Decoupled(new PostGDCmd)
  })
}
```

**指令分类逻辑**:
```scala
val func7 = io.id_i.bits.cmd.inst.funct
val is_mem_instr = (func7 === MVIN_BITPAT) || (func7 === MVOUT_BITPAT)
val is_ball_instr = !is_mem_instr
```

**输入输出**:
- 输入: RoCCCommandBB 指令流
- 输出: 分类后的 PostGDCmd
- 边缘情况: 通过 ready/valid 握手处理背压

**依赖项**: framework.rocket.RoCCCommandBB, framework.builtin.memdomain.DISA

### rs/ 保留站组件

**主要功能**: 实现乱序执行的指令调度机制

**组件说明**:
- **ReservationStation**: 保留站主体，管理指令的暂存和调度
- **IssueQueue**: 发射队列，控制指令的发射时机
- **ReorderBuffer**: 重排序缓冲，保证指令按序提交
- **CommitQueue**: 提交队列，管理指令的最终提交
- **NextROBIdCounter**: ROB ID 分配计数器

**数据流向**:
```
GlobalDecoder → ReservationStation → IssueQueue → 执行单元
                      ↓
              ReorderBuffer → CommitQueue → 提交
```

## 使用方法

### 使用方法

**指令处理流程**:
1. RoCC 指令进入 GlobalDecoder
2. 根据 funct 字段区分指令类型
3. 分类后的指令进入对应的保留站
4. 保留站管理指令的调度和执行

**配置参数**:
```scala
// 在 CustomBuckyBallConfig 中配置
implicit val config: CustomBuckyBallConfig
```

### 注意事项

1. **指令分类**: 仅区分内存指令(MVIN/MVOUT)和其他指令
2. **背压处理**: 使用标准的 Decoupled 接口处理流控
3. **依赖关系**: 需要正确配置 DISA 模块的指令位模式
4. **ROB 管理**: 保留站需要与 ROB 协调保证指令顺序
