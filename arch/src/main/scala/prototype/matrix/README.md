# 矩阵运算加速器

## 概述

该目录实现了 BuckyBall 的矩阵运算加速器，用于矩阵乘法和相关运算。位于 `arch/src/main/scala/prototype/matrix` 下，作为矩阵计算加速器，支持多种数据格式和运算模式。

实现的核心组件：
- **bbfp_control.scala**: 矩阵运算控制器
- **bbfp_pe.scala**: 处理单元(PE)和MAC单元
- **bbfp_buffer.scala**: 数据缓冲管理
- **bbfp_load.scala**: 数据加载单元
- **bbfp_ex.scala**: 执行单元
- **bbfpIns_decode.scala**: 指令解码器

## 代码结构

```
matrix/
├── bbfp_control.scala   - 控制器主体
├── bbfp_pe.scala        - 处理单元实现
├── bbfp_buffer.scala    - 缓冲管理
├── bbfp_load.scala      - 加载单元
├── bbfp_ex.scala        - 执行单元
└── bbfpIns_decode.scala - 指令解码
```

### 文件依赖关系

**bbfp_control.scala** (控制器层)
- 集成各个子模块(ID, LU, EX等)
- 管理 SRAM 和 Accumulator 接口
- 处理 Ball 域命令

**bbfp_pe.scala** (计算核心层)
- 实现 MacUnit 乘累加单元
- 定义 PEControl 控制信号
- 处理有符号/无符号运算

**其他模块** (功能支持层)
- 提供数据缓冲、加载、执行等支持功能

## 模块说明

### bbfp_control.scala

**主要功能**: 矩阵运算加速器的顶层控制模块

**模块集成**:
```scala
class BBFP_Control extends Module {
  val BBFP_ID = Module(new BBFP_ID)
  val ID_LU = Module(new ID_LU)
  val BBFP_LoadUnit = Module(new BBFP_LoadUnit)
  val LU_EX = Module(new LU_EX)
}
```

**接口定义**:
```scala
val io = IO(new Bundle {
  val cmdReq = Flipped(Decoupled(new BallRsIssue))
  val cmdResp = Decoupled(new BallRsComplete)
  val is_matmul_ws = Input(Bool())
  val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(...)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(...)))
  val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(...)))
  val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(...)))
})
```

**数据流向**:
```
cmdReq → BBFP_ID → ID_LU → BBFP_LoadUnit → LU_EX
                              ↓
                         SRAM/ACC 接口
```

### bbfp_pe.scala

**主要功能**: 实现矩阵运算的基本处理单元

**MAC 单元定义**:
```scala
class MacUnit extends Module {
  val io = IO(new Bundle {
    val in_a = Input(UInt(7.W))    // [6]=sign, [5]=flag, [4:0]=value
    val in_b = Input(UInt(7.W))    // [6]=sign, [5]=flag, [4:0]=value
    val in_c = Input(UInt(32.W))   // [31]=sign, [30:0]=value
    val out_d = Output(UInt(32.W)) // 输出结果
  })
}
```

**数据格式处理**:
```scala
// 提取符号位和数值
val sign_a = io.in_a(6)
val sign_b = io.in_b(6)
val flag_a = io.in_a(5)
val flag_b = io.in_b(5)
val value_a = io.in_a(4, 0)
val value_b = io.in_b(4, 0)

// 根据flag位决定是否左移
val shifted_a = Mux(flag_a === 1.U, value_a << 2, value_a)
val shifted_b = Mux(flag_b === 1.U, value_b << 2, value_b)
```

**有符号运算**:
```scala
val a_signed = Mux(sign_a === 1.U, -(shifted_a.zext), shifted_a.zext).asSInt
val b_signed = Mux(sign_b === 1.U, -(shifted_b.zext), shifted_b.zext).asSInt
```

**控制信号**:
```scala
class PEControl extends Bundle {
  val propagate = UInt(1.W)   // 传播控制
}
```

## 使用方法

### 数据格式

**输入格式**: 7位压缩格式
- bit[6]: 符号位 (0=正数, 1=负数)
- bit[5]: 标志位 (1=左移2位)
- bit[4:0]: 5位数值

**输出格式**: 32位有符号数
- bit[31]: 符号位
- bit[30:0]: 31位数值

### 运算特性

**MAC 运算**: 乘累加操作 (Multiply-Accumulate)
- 支持有符号和无符号运算
- 可配置的位移操作
- 32位累加器输出

**流水线结构**:
- ID: 指令解码阶段
- LU: 加载单元阶段
- EX: 执行单元阶段

### 注意事项

1. **数据格式**: 使用自定义的7位压缩格式减少存储开销
2. **符号处理**: 支持有符号数的正确运算和符号扩展
3. **位移优化**: 通过flag位控制数据的预处理位移
4. **接口兼容**: 与 SRAM 和 Accumulator 接口完全兼容
5. **流水线设计**: 多级流水线提高吞吐量
