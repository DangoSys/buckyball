# MatrixBall - 矩阵处理球

## 概述

MatrixBall 是 BuckyBall 系统中的矩阵处理加速器封装，它将 BBFP（BuckyBall Floating Point）控制器集成到 Ball 域架构中。MatrixBall 专门用于高性能的矩阵运算，特别是浮点矩阵乘法和相关的线性代数操作。

## 二、文件结构

```
matrixball/
├── MatrixBall.scala  - MatrixBall 封装实现
└── README.md        - 本文档
```

## 三、核心功能

### MatrixBall LazyModule

MatrixBall 是一个 LazyModule，负责：
- 封装 BBFP_Control 核心矩阵处理逻辑
- 通过 Diplomacy 协议协商存储带宽
- 连接 scratchpad 和 accumulator 存储器
- 提供标准的 Ball 域接口
- 支持矩阵乘法的权重静态（Weight Stationary）模式

**主要特性**：
- **浮点运算**: 专门优化的浮点矩阵运算
- **存储带宽协商**: 自动请求所需的存储带宽
- **权重静态支持**: 支持 Weight Stationary 数据流模式
- **标准接口**: 提供统一的命令请求/响应接口

## 四、架构设计

### Diplomacy 节点配置

```scala
val node = new BallNode(Seq(BBusParams(
  sramReadBW = b.sp_banks + b.acc_banks,   // 读带宽需求
  sramWriteBW = b.sp_banks + b.acc_banks   // 写带宽需求
)))
```

MatrixBall 请求的带宽包括：
- **scratchpad banks**: `b.sp_banks` 个读写端口，用于存储输入矩阵
- **accumulator banks**: `b.acc_banks` 个读写端口，用于存储输出结果

### 核心组件连接

**BBFP_Control 实例化**：
```scala
val bbfpControl = Module(new BBFP_Control)
```

**存储器连接策略**：
```scala
// Scratchpad 连接 - 存储输入矩阵数据
for (i <- 0 until b.sp_banks) {
  bundle.data.sramRead(i) <> bbfpControl.io.sramRead(i)
  bundle.data.sramWrite(i) <> bbfpControl.io.sramWrite(i)
}

// Accumulator 连接 - 存储累加结果
for (i <- 0 until b.acc_banks) {
  val readIdx = b.sp_banks + i
  val writeIdx = b.sp_banks + i
  bundle.data.sramRead(readIdx) <> bbfpControl.io.accRead(i)
  bundle.data.sramWrite(writeIdx) <> bbfpControl.io.accWrite(i)
}
```

### 接口设计

```scala
val io = IO(new Bundle {
  val cmdReq = Flipped(Decoupled(new BallRsIssue))     // 命令请求输入
  val cmdResp = Decoupled(new BallRsComplete)          // 命令响应输出
  val is_matmul_ws = Input(Bool())                     // 权重静态模式控制
})
```

## 五、工作流程

### 矩阵乘法执行流程

1. **命令接收**: 从保留站接收矩阵运算命令
2. **模式配置**: 根据 `is_matmul_ws` 信号配置数据流模式
3. **数据加载**: 从 scratchpad 加载输入矩阵数据
4. **矩阵运算**: BBFP_Control 执行浮点矩阵乘法
5. **结果存储**: 将计算结果存储到 accumulator
6. **完成响应**: 向保留站发送执行完成信号

### 权重静态模式

**Weight Stationary 特点**：
- 权重矩阵保持在本地存储器中
- 输入数据流式传输
- 减少权重数据的重复加载
- 提高计算效率和能耗比

```scala
bbfpControl.io.is_matmul_ws := io.is_matmul_ws
```

## 六、存储器访问模式

### Scratchpad 使用模式
- **输入矩阵A**: 存储在 scratchpad 的前半部分
- **权重矩阵B**: 存储在 scratchpad 的后半部分
- **访问模式**: 支持行优先和列优先访问
- **访问策略**: 数据缓存和访问管理

### Accumulator 使用模式
- **部分积累**: 存储矩阵乘法的中间结果
- **最终结果**: 存储完整的输出矩阵
- **累加模式**: 支持多次累加操作
- **输出格式**: 支持多种输出数据格式

## 七、性能特性

### 计算性能
- **浮点精度**: 支持单精度和半精度浮点运算
- **并行度**: 多个乘加单元并行工作
- **吞吐量**: 高吞吐量的矩阵乘法操作
- **延迟优化**: 流水线设计减少计算延迟

### 存储性能
- **带宽利用**: 高效利用存储带宽
- **数据重用**: 最大化数据重用减少访存
- **缓存命中**: 优化的缓存策略提高命中率

## 八、配置参数

### 关键配置项
```scala
class MyBuckyBallConfig extends CustomBuckyBallConfig {
  override val sp_banks = 8     // Scratchpad bank 数量
  override val acc_banks = 4    // Accumulator bank 数量
  // BBFP 相关配置
  val matrix_size = 64          // 支持的矩阵大小
  val fp_precision = 32         // 浮点精度位数
}
```

### 性能调优参数
- **bank 数量**: 影响并行访存能力
- **矩阵分块大小**: 影响缓存效率
- **流水线深度**: 影响延迟和吞吐量平衡

## 九、使用示例

### 基本矩阵乘法
```scala
// 配置 MatrixBall
val matrixBall = LazyModule(new MatrixBall)

// 执行矩阵乘法
matrixBall.module.io.cmdReq.valid := true.B
matrixBall.module.io.cmdReq.bits.cmd.bid := 2.U  // BBFP ball ID
matrixBall.module.io.is_matmul_ws := true.B      // 启用权重静态模式
```

### 系统集成
```scala
class BallDomain extends LazyModule {
  val matrixBall = LazyModule(new MatrixBall)
  val reservationStation = Module(new BallReservationStation)

  // 连接保留站
  matrixBall.module.io.cmdReq <> reservationStation.io.issue_o.ball2
  reservationStation.io.commit_i.ball2 <> matrixBall.module.io.cmdResp
}
```

## 十、应用场景

### 机器学习
- **神经网络推理**: 前向传播中的矩阵乘法
- **训练加速**: 反向传播中的梯度计算
- **权重更新**: 参数更新中的矩阵运算

### 科学计算
- **线性代数**: 基础的矩阵运算
- **数值求解**: 线性方程组求解
- **信号处理**: 滤波器和变换运算

### 图形处理
- **3D 变换**: 顶点变换矩阵运算
- **着色器计算**: GPU 风格的并行计算
- **图像处理**: 卷积和滤波操作

## 十一、调试和监控

### 性能监控
- **计算吞吐量**: 每秒执行的矩阵运算数
- **存储带宽利用率**: 实际使用的存储带宽
- **缓存命中率**: 数据缓存的命中统计
- **功耗分析**: 各个组件的功耗分布

### 调试接口
- **状态寄存器**: 查看内部状态和配置
- **性能计数器**: 统计各种性能指标
- **错误检测**: 检测计算错误和异常情况

## 十二、优化建议

### 算法优化
- **分块策略**: 优化矩阵分块大小
- **数据布局**: 改进数据在存储器中的布局
- **计算顺序**: 优化计算的执行顺序

### 硬件优化
- **流水线深度**: 调整流水线深度平衡延迟和面积
- **并行度**: 增加计算单元提高并行度
- **存储层次**: 优化存储器层次结构

## 十三、相关文档

- [Ball域概览](../README.md)
- [保留站和ROB](../rs/README.md)
- [矩阵运算引擎](../../../prototype/matrix/README.md)
- [BBFP控制器详细文档](../../../prototype/matrix/README.md)
- [Blink通信框架](../../../framework/blink/README.md)
