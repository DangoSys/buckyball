# Im2col Ball 图像到列转换球域

## 概述

该目录包含了 Im2col (Image to Column) 加速器的球域封装实现。Im2col 是卷积神经网络中的关键操作，将图像数据重新排列为列矩阵格式，以便进行高效的矩阵乘法运算。该模块将 prototype.im2col.Im2col 核心加速器封装为符合 BuckyBall 球域架构的 Ball 节点。

Im2colBall 在球域架构中的作用：
- 提供标准的 Ball 节点接口，集成到球域总线系统
- 封装 Im2col 核心加速器，提供 SRAM 访问能力
- 支持通过保留站(RS)进行指令调度和执行控制
- 实现 diplomacy 协议的参数协商和连接管理

## 代码结构

```
im2col/
└── Im2colBall.scala    - Im2col球域封装模块
```

### 文件依赖关系

**Im2colBall.scala** (封装模块)
- 继承 LazyModule，实现 diplomacy 协议
- 封装 prototype.im2col.Im2col 核心加速器
- 提供 BallNode 接口，连接到球域总线
- 实现与保留站的命令接口

## 模块说明

### Im2colBall.scala

**主要功能**: Im2col 加速器的球域封装，提供标准 Ball 节点接口

**关键组件**:

```scala
class Im2colBall extends LazyModule {
  // 创建Ball节点，支持SRAM访问
  val node = new BallNode(Seq(BBusParams(
    sramReadBW = b.sp_banks,
    sramWriteBW = b.sp_banks
  )))

  // 实例化Im2col核心模块
  val im2col = Module(new Im2col)

  // 外部命令接口
  val io = IO(new Bundle {
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)
  })
}
```

**接口连接**:
- **命令接口**: 直接连接 Im2col 核心的命令请求和响应
- **SRAM接口**: 通过 diplomacy 连接前 `sp_banks` 个 SRAM 端口
- **多余端口**: 超出需求的 SRAM 端口设置为无效状态

**Diplomacy 协商**:
```scala
val negotiatedParams = node.edges.out.map(e => (e.sramReadBW, e.sramWriteBW))
require(negotiatedParams.forall(p => p._1 >= b.sp_banks && p._2 >= b.sp_banks),
        "negotiated bandwidth must support Im2col requirements")
```

**SRAM 连接逻辑**:
```scala
// 连接需要的SRAM端口
for (i <- 0 until b.sp_banks) {
  bundle.data.sramRead(i) <> im2col.io.sramRead(i)
  bundle.data.sramWrite(i) <> im2col.io.sramWrite(i)
}

// 处理多余端口
for (i <- b.sp_banks until edge.sramReadBW) {
  bundle.data.sramRead(i).req.ready := false.B
  bundle.data.sramRead(i).resp.valid := false.B
  bundle.data.sramRead(i).resp.bits := DontCare
}
```

**输入输出**:
- 输入: BallRsIssue 命令请求，来自保留站的指令调度
- 输出: BallRsComplete 命令响应，返回执行完成状态
- SRAM: 通过 diplomacy 连接的 scratchpad 读写接口
- 边缘情况: 协商带宽不足时触发 require 断言失败

**依赖项**:
- prototype.im2col.Im2col (核心加速器)
- framework.blink.BallNode (Ball节点接口)
- examples.toy.balldomain.rs (保留站接口定义)

## 使用方法

### 设计特点

1. **标准封装**: 遵循 BuckyBall 球域架构的标准 Ball 节点设计模式
2. **资源协商**: 通过 diplomacy 协议确保获得足够的 SRAM 带宽资源
3. **接口统一**: 提供与其他 Ball 节点一致的命令和数据接口
4. **资源优化**: 只使用必需的 SRAM 端口，多余端口设为无效以节省资源

### 配置参数

Im2colBall 的配置依赖于以下参数：
- `b.sp_banks`: 需要的 scratchpad bank 数量
- `sramReadBW/sramWriteBW`: 协商的 SRAM 读写带宽

### 使用示例

```scala
// 创建Im2col Ball实例
val im2colBall = LazyModule(new Im2colBall)

// 连接到球域总线
ballBus.ballNodes(0) := im2colBall.node

// 连接命令接口到保留站
reservationStation.io.im2colReq <> im2colBall.module.io.cmdReq
reservationStation.io.im2colResp <> im2colBall.module.io.cmdResp
```

### 与核心加速器的关系

Im2colBall 是对 `prototype.im2col.Im2col` 的封装：
- 核心加速器实现具体的 Im2col 算法逻辑
- Ball 封装提供球域架构的标准接口和资源管理
- 通过这种分层设计，核心算法与系统架构解耦

### 注意事项

1. **带宽要求**: 确保球域总线能提供足够的 SRAM 带宽
2. **命令协议**: 命令接口必须与保留站的协议保持一致
3. **资源分配**: 只使用前 `sp_banks` 个 SRAM 端口，需要合理规划资源
4. **时序约束**: diplomacy 连接的时序要求需要与核心加速器匹配
