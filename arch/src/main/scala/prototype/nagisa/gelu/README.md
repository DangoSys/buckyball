# GELU加速单元 (GELU Accelerator)

## 概述

GELU (Gaussian Error Linear Unit) 加速单元是BuckyBall框架中的专用计算加速器，用于高效执行GELU激活函数运算。GELU是现代深度学习模型（如Transformer、BERT、GPT等）中广泛使用的非线性激活函数。

## 架构设计

### 流水线结构

GELU加速单元采用4级流水线架构：

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│   Ctrl     │───▶│   Load     │───▶│   Execute  │───▶│   Store    │
│   Unit     │    │   Unit     │    │   Unit     │    │   Unit     │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
     ↓                  ↓                  ↓                  ↓
  Decode          Load Data         Compute GELU      Write Back
  Command         from SRAM/ACC     Approximation     to SRAM/ACC
```

### 模块组成

1. **GeluCtrlUnit**: 控制单元，负责指令解码和状态机管理
2. **GeluLoadUnit**: 加载单元，负责从SRAM/ACC读取数据
3. **GeluEXUnit**: 执行单元，负责GELU计算
4. **GeluStoreUnit**: 存储单元，负责写回结果到SRAM/ACC
5. **GeluUnit**: 顶层模块，集成所有子单元
6. **GeluBall**: Ball封装，实现BallRegist接口

## 功能特性

- **数据格式**:
  - SRAM模式：INT8输入 → GELU → INT8输出
  - ACC模式：INT32输入 → GELU → INT32输出

- **向量化处理**: 每次处理16个元素（veclane=16）

- **双模式支持**:
  - 支持从Scratchpad读写（INT8数据）
  - 支持从Accumulator读写（INT32数据）

- **流水线处理**: 高吞吐量的流水线设计

## 指令格式

### 指令参数

| 参数 | 位宽 | 描述 |
|-----|------|------|
| `iter` | 10 | 要处理的向量个数 (1-1024) |
| `op1_bank` | 2+ | 输入数据所在的Bank号 |
| `op1_bank_addr` | 12+ | 输入起始地址 |
| `wr_bank` | 2+ | 输出数据写入的Bank号 |
| `wr_bank_addr` | 12+ | 输出起始地址 |
| `is_acc` | 1 | 数据类型选择 (0=SRAM, 1=ACC) |

### 指令编码

- **Opcode**: `GELU` (func7 = 0b0100011 = 35)
- **Ball ID**: 4
- **格式**: `rs1[spAddrLen-1:0]` = op1地址, `rs2[spAddrLen-1:0]` = wr地址, `rs2[spAddrLen+9:spAddrLen]` = iter

## 使用示例

### 示例1: 处理单个向量 (INT8)

```
iter = 1
op1_bank = 0, op1_bank_addr = 0x100
wr_bank = 1, wr_bank_addr = 0x200
is_acc = 0

输入：SRAM[0][0x100] 的16个INT8元素
输出：SRAM[1][0x200] 的16个INT8元素
```

### 示例2: 批量处理 (INT32)

```
iter = 64
op1_bank = 4 (ACC bank 0), op1_bank_addr = 0x000
wr_bank = 5 (ACC bank 1), wr_bank_addr = 0x000
is_acc = 1

输入：ACC[0][0x000~0x03F] 的64个向量（1024个INT32元素）
输出：ACC[1][0x000~0x03F] 的64个向量（1024个INT32元素）
```

## GELU实现

当前实现采用简化的分段线性近似算法：

```scala
// 对于输入 x:
when(x >= threshold) {
  GELU(x) ≈ x                    // 正值区域
}.elsewhen(x <= -threshold) {
  GELU(x) ≈ 0                    // 强负值区域
}.elsewhen(x >= 0) {
  GELU(x) ≈ x                    // 小正值区域
}.otherwise {
  GELU(x) ≈ x/2                  // 小负值区域
}
```

其中 `threshold = 3`。

**注意**: 这是一个简化的实现。如需更高精度，可以：
1. 使用查找表 (LUT) 实现tanh近似
2. 使用多项式近似
3. 增加分段数量

## 集成说明

GELU Ball已经成功集成到系统中：

1. ✅ 在DISA中定义了GELU指令 (opcode = 35)
2. ✅ 在DomainDecoder中添加了GELU指令解码
3. ✅ 在BBusModule中注册了GeluBall (ID = 4)
4. ✅ 在BallRSModule中注册了GeluBall

## 文件结构

```
prototype/nagisa/gelu/
├── README.md              # 本文档
├── GeluBundle.scala       # Bundle定义
├── GeluCtrlUnit.scala     # 控制单元
├── GeluLoadUnit.scala     # 加载单元
├── GeluEXUnit.scala       # 执行单元
├── GeluStoreUnit.scala    # 存储单元
└── GeluUnit.scala         # 顶层模块

examples/toy/balldomain/geluball/
└── GeluBall.scala         # Ball封装
```

## 性能特性

- **延迟**: ~15-20个时钟周期（单个向量）
- **吞吐率**: 1个向量/周期（流水线饱和时）
- **数据宽度**: 16个元素并行处理
- **支持的迭代数**: 1-1024个向量

## 参考规范

详细的硬件规范请参考：`spec.md`
