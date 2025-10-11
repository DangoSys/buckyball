# Softmax加速单元

## 概述

Softmax加速单元是BuckyBall框架中的专用计算加速器，用于高效执行Softmax激活函数运算。Softmax是深度学习模型（如Transformer、BERT、GPT等）中广泛使用的归一化激活函数，特别是在注意力机制(Attention)中起到关键作用。

### 主要特性

- **数据格式**: 输入为INT8/INT32，内部计算FP32，输出为INT8/INT32
- **向量化处理**: 每次处理16个元素（veclane=16）
- **流水线架构**: 多级流水线设计（Load, FindMax, ExpSum, Normalize, Store）
- **计算方法**: 采用数值稳定的Softmax算法（减最大值）
- **存储接口**: 支持Scratchpad和Accumulator读写

### 数学定义

标准Softmax函数：
```
Softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

数值稳定的实现：
```
max_x = max(x_1, x_2, ..., x_n)
Softmax(x_i) = exp(x_i - max_x) / Σ exp(x_j - max_x)
```

## 系统架构

### 流水线结构

```
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│  Load  │───▶│FindMax │───▶│ExpSum  │───▶│ Norm   │───▶│ Store  │
│ Stage  │    │ Stage  │    │ Stage  │    │ Stage  │    │ Stage  │
└────────┘    └────────┘    └────────┘    └────────┘    └────────┘
    │             │             │             │             │
  Decode      Find Max     Compute exp()   Normalize    Write Back
  Command     from input   and accumulate   by dividing   to ACC/SRAM
              vectors       sum             sum total
```

### 子模块说明

1. **SoftmaxCtrlUnit**: 控制单元，负责指令解析和流程控制
2. **SoftmaxLoadUnit**: 数据加载单元，从SRAM/ACC读取输入数据
3. **SoftmaxFindMaxUnit**: 查找最大值单元，实现数值稳定性
4. **SoftmaxExpSumUnit**: 指数计算和累加单元
5. **SoftmaxNormalizeUnit**: 归一化单元，执行除法运算
6. **SoftmaxStoreUnit**: 数据存储单元，写回结果到SRAM/ACC

## 接口说明

### 指令格式

**Ball ID**: 6 (在busRegister.scala中注册)

**func7**: 37 (0x25)

**RS1编码** (32位):
- [1:0]: op1_bank - 输入数据Bank号
- [13:2]: op1_spaddr - 输入起始地址
- [15:14]: wr_bank - 输出数据Bank号
- [27:16]: wr_spaddr - 输出起始地址

**RS2编码** (32位):
- [9:0]: iter - 向量迭代次数
- [10]: is_acc - 数据类型选择 (0=SRAM/INT8, 1=ACC/INT32)
- [20:11]: dim_len - Softmax维度长度
- [30:21]: batch - 批次大小
- [31]: log_mode - LogSoftmax模式标志

### C API

```c
void bb_softmax(uint32_t op1_bank, uint32_t op1_addr,
                uint32_t wr_bank, uint32_t wr_addr,
                uint32_t iter, uint32_t is_acc,
                uint32_t dim_len, uint32_t batch,
                uint32_t log_mode);
```

## 使用示例

### 基本使用

```c
#include <bbhw/isa/isa.h>

// 输入数据: 16个INT8元素
elem_t input[16] = {1, 2, 3, ..., 16};
elem_t output[16];

// 加载到SRAM bank 0
bb_mvin((uintptr_t)input, spad_addr(0, 0), 1, 1);
bb_fence();

// 执行Softmax: dim_len=16, batch=1
bb_softmax(0, 0, 1, 0, 1, 0, 16, 1, 0);
bb_fence();

// 读取结果
bb_mvout((uintptr_t)output, spad_addr(1, 0), 1);
bb_fence();
```

### 批量处理

```c
// 处理4个独立的Softmax组，每组16个元素
bb_softmax(0, 0, 1, 0, 4, 0, 16, 4, 0);
```

## 文件结构

```
softmax/
├── README.md                    # 本文档
├── spec.md                      # 详细设计规范
├── SoftmaxBundle.scala          # Bundle和接口定义
├── SoftmaxCtrlUnit.scala        # 控制单元
├── SoftmaxLoadUnit.scala        # 数据加载单元
├── SoftmaxFindMaxUnit.scala     # 最大值查找单元
├── SoftmaxExpSumUnit.scala      # 指数计算和求和单元
├── SoftmaxNormalizeUnit.scala   # 归一化单元
├── SoftmaxStoreUnit.scala       # 数据存储单元
└── SoftmaxUnit.scala            # 顶层模块
```

## 测试

### Ctest测试

```bash
cd /home/mio/Code/buckyball
make -C bb-tests/workloads/build softmax_test
./bb-tests/workloads/build/bin/softmax_test
```

### Verilator仿真

```bash
# 生成Verilog
bbdev verilator verilog

# 编译仿真器
bbdev verilator build

# 运行测试
bbdev verilator sim softmax_test
```

## 性能特性

### 延迟估算

- **单个Softmax (16元素)**: ~30周期
- **单个Softmax (256元素)**: ~250周期

### 吞吐率

- 批量处理可实现流水线重叠
- 稳态吞吐率: 1 Softmax / (dim_len/16 + overhead) 周期

## 限制与注意事项

1. **维度限制**: dim_len范围为1-1024
2. **批次限制**: batch范围为1-1024
3. **迭代限制**: iter范围为1-1024
4. **数值精度**: 使用近似exp计算，精度约为±10%
5. **内存对齐**: 输入/输出地址应按16字节对齐

## 集成信息

### 注册位置

- **BBus注册**: `arch/src/main/scala/examples/toy/balldomain/bbus/busRegister.scala`
- **Ball封装**: `arch/src/main/scala/examples/toy/balldomain/softmaxball/SoftmaxBall.scala`
- **Ball ID**: 6

### ISA支持

- **指令定义**: `bb-tests/workloads/lib/bbhw/isa/37_softmax.c`
- **测试用例**: `bb-tests/workloads/src/CTest/softmax_test.c`

## 相关文档

- [详细设计规范](spec.md)
- [GELU加速单元](../gelu/README.md)
- [LayerNorm加速单元](../layernorm/README.md)

## 版本历史

- v1.0 (2025-10): 初始实现，支持基本Softmax和LogSoftmax功能
