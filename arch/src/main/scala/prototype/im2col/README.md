# Im2col 图像处理加速器

## 概述

该目录实现了 BuckyBall 的 Im2col 操作加速器，用于卷积神经网络中的图像到列矩阵转换。位于 `arch/src/main/scala/prototype/im2col` 下，作为图像处理加速器，将卷积操作转换为矩阵乘法操作以提高计算效率。

实现的核心组件：
- **im2col.scala**: Im2col 加速器主体实现

## 代码结构

```
im2col/
└── im2col.scala  - Im2col 加速器实现
```

### 模块职责

**Im2col.scala** (加速器实现层)
- 实现图像到列矩阵的转换逻辑
- 管理 SRAM 读写操作
- 提供 Ball 域命令接口

## 模块说明

### im2col.scala

**主要功能**: 实现卷积窗口的滑动和数据重排

**状态机定义**:
```scala
val idle :: read :: read_and_convert :: complete :: Nil = Enum(4)
val state = RegInit(idle)
```

**关键寄存器**:
```scala
val ConvertBuffer = RegInit(VecInit(Seq.fill(4)(VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W))))))
val rowptr = RegInit(0.U(10.W))    // 卷积窗口左上角行指针
val colptr = RegInit(0.U(5.W))     // 卷积窗口左上角列指针
val krow_reg = RegInit(0.U(log2Up(b.veclane).W))  // 卷积核行数
val kcol_reg = RegInit(0.U(log2Up(b.veclane).W))  // 卷积核列数
```

**命令解析**:
```scala
when(io.cmdReq.fire) {
  rowptr := io.cmdReq.bits.cmd.special(37,28)      // 起始行
  colptr := io.cmdReq.bits.cmd.special(27,23)      // 起始列
  kcol_reg := io.cmdReq.bits.cmd.special(3,0)      // 卷积核列数
  krow_reg := io.cmdReq.bits.cmd.special(7,4)      // 卷积核行数
  incol_reg := io.cmdReq.bits.cmd.special(12,8)    // 输入矩阵列数
  inrow_reg := io.cmdReq.bits.cmd.special(22,13)   // 输入矩阵行数
}
```

**数据转换逻辑**:
```scala
// 填充窗口数据
for (i <- 0 until 4; j <- 0 until 4) {
  when(i.U < krow_reg && j.U < kcol_reg) {
    val bufferRow = (rowcnt + i.U) % krow_reg
    val bufferCol = (colptr + j.U) % incol_reg
    window((i.U * kcol_reg) + j.U) := ConvertBuffer(bufferRow)(bufferCol)
  }.otherwise {
    window((i.U * kcol_reg) + j.U) := 0.U
  }
}
```

**SRAM 接口**:
```scala
val io = IO(new Bundle {
  val cmdReq = Flipped(Decoupled(new BallRsIssue))
  val cmdResp = Decoupled(new BallRsComplete)
  val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(...)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(...)))
})
```

**处理流程**:
1. **idle**: 等待命令，解析卷积参数
2. **read**: 读取初始的卷积核大小的数据到缓冲区
3. **read_and_convert**: 滑动窗口，转换数据并写回
4. **complete**: 发送完成信号

**输入输出**:
- 输入: Ball 域命令，包含卷积参数和地址信息
- 输出: 转换后的列矩阵数据，完成信号
- 边缘情况: 边界处理时填充零值

## 使用方法

### 算法原理

**Im2col 转换**: 将卷积操作转换为矩阵乘法
- 输入: H×W 的图像，K×K 的卷积核
- 输出: (H-K+1)×(W-K+1) 个 K×K 的窗口，展开为列向量

**滑动窗口**:
- 按行优先顺序滑动卷积窗口
- 每个窗口位置生成一个列向量
- 使用循环缓冲区优化内存访问

### 注意事项

1. **缓冲区管理**: 使用 4×veclane 的转换缓冲区存储窗口数据
2. **边界处理**: 超出图像边界的位置填充零值
3. **地址计算**: 支持可配置的起始地址和 bank 选择
4. **流水线优化**: 在转换过程中提前发送下一行的读请求
5. **参数限制**: 最大支持 4×4 的卷积核大小
