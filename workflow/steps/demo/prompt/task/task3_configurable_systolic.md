# 任务 3：支持可配置位宽和量化精度的 WS/OS 脉动阵列

## 目标

设计并实现一个支持 **Weight Stationary (WS)** 和 **Output Stationary (OS)** 两种数据流模式的脉动阵列，支持**运行时可配置的数据位宽**和**量化精度**，以适应不同精度的神经网络推理需求。

## 技术规格

### 1. 脉动阵列架构

#### 支持的数据流模式

**Weight Stationary (WS)**：
- 权重保持在 PE 中
- 输入激活从左向右流动
- 部分和向下累积
- 适合卷积神经网络

**Output Stationary (OS)**：
- 输出部分和保持在 PE 中
- 权重从上向下流动
- 激活从左向右流动
- 节省能量消耗

#### 阵列参数
```scala
val arrayDim = 8          // 8x8 脉动阵列（可配置为 4x4, 8x8, 16x16）
val maxDataWidth = 16     // 最大支持 16-bit
val accWidth = 32         // 32-bit 累加器
```

### 2. 可配置位宽支持

#### 支持的数据类型

| 位宽 | 数据类型 | 量化方式 | 动态范围 |
|------|---------|---------|---------|
| 2-bit | INT2 | 对称量化 | [-2, 1] |
| 4-bit | INT4 | 对称量化 | [-8, 7] |
| 8-bit | INT8 | 对称/非对称 | [-128, 127] |
| 16-bit | INT16 | 全精度 | [-32768, 32767] |
| 8-bit | UINT8 | 非对称量化 | [0, 255] |

#### 量化参数
```scala
// 每层可独立配置量化参数
case class QuantConfig(
  dataWidth: Int,        // 数据位宽 (2/4/8/16)
  weightScale: Float,    // 权重缩放因子
  inputScale: Float,     // 输入缩放因子
  outputScale: Float,    // 输出缩放因子
  zeroPoint: Int,        // 零点偏移（非对称量化）
  signedMode: Boolean    // 有符号/无符号
)
```

### 3. Ball 接口设计

#### 控制寄存器
```scala
// 基本配置 (rs1)
rs1(3:0)    : data_width     // 0=2bit, 1=4bit, 2=8bit, 3=16bit
rs1(7:4)    : matrix_dim     // 矩阵维度（4/8/16）
rs1(8)      : dataflow_mode  // 0=WS, 1=OS
rs1(9)      : signed_mode    // 0=unsigned, 1=signed
rs1(10)     : symmetric_quant // 0=非对称, 1=对称

// 量化参数 (rs2)
rs2(7:0)    : weight_scale   // 权重缩放因子 (Q4.4 定点)
rs2(15:8)   : input_scale    // 输入缩放因子
rs2(23:16)  : output_scale   // 输出缩放因子
rs2(31:24)  : zero_point     // 零点偏移值

// 扩展配置 (special)
special(7:0)  : acc_shift    // 累加后右移位数（用于缩放）
special(8)    : requant_en   // 启用重新量化
special(9)    : clip_en      // 启用结果截断
special(23:16): clip_min     // 截断最小值
special(31:24): clip_max     // 截断最大值
```

#### 状态寄存器
```scala
status(0)   : busy           // 计算中
status(1)   : overflow       // 累加溢出
status(2)   : underflow      // 下溢
status(7:4) : current_width  // 当前配置的位宽
status(31:16): util_rate     // PE 利用率 (百分比)
```

### 4. 内存布局

#### 输入数据（支持位宽打包）
```
// 8-bit 模式：每个地址存储 1 个元素
SRAM[0x0000]: [elem0]
SRAM[0x0001]: [elem1]

// 4-bit 模式：每个地址存储 2 个元素
SRAM[0x0000]: [elem1:elem0]
SRAM[0x0001]: [elem3:elem2]

// 2-bit 模式：每个地址存储 4 个元素
SRAM[0x0000]: [elem3:elem2:elem1:elem0]
```

#### 地址映射
- **权重 (INT8/INT4/INT2)**：`SRAM[0x0000 - 0x0FFF]`
- **输入 (INT8/INT4/INT2)**：`SRAM[0x1000 - 0x1FFF]`
- **量化参数**：`SRAM[0x2000 - 0x20FF]`
- **输出 (INT32)**：`ACC[0x0000 - 0x0FFF]`
- **统计信息**：`ACC[0x1000 - 0x10FF]`

## 实现要求

### 1. Chisel 代码结构

需要生成以下文件：

```
arch/src/main/scala/prototype/generated/configurable/
├── ConfigurableSystolicUnit.scala   // 主计算单元
├── ConfigurableSystolicBall.scala   // Ball 包装器
├── FlexiblePE.scala                 // 可配置位宽 PE
├── QuantizationUnit.scala           // 量化/反量化单元
├── BitWidthConverter.scala          // 位宽转换逻辑
└── AccumulatorScaler.scala          // 累加器缩放单元
```

### 2. PE 设计（支持多位宽）

```scala
class FlexiblePE(maxWidth: Int = 16) extends Module {
  val io = IO(new Bundle {
    val dataWidth = Input(UInt(2.W))   // 0=2bit, 1=4bit, 2=8bit, 3=16bit
    val signedMode = Input(Bool())
    
    val weightIn = Input(UInt(maxWidth.W))
    val activIn = Input(UInt(maxWidth.W))
    val partialIn = Input(UInt(32.W))
    
    val activOut = Output(UInt(maxWidth.W))
    val partialOut = Output(UInt(32.W))
  })
  
  // 根据 dataWidth 动态提取有效位
  val weight = Wire(SInt(maxWidth.W))
  val activ = Wire(SInt(maxWidth.W))
  
  when(io.dataWidth === 0.U) {  // 2-bit
    weight := io.weightIn(1, 0).asSInt
    activ := io.activIn(1, 0).asSInt
  }.elsewhen(io.dataWidth === 1.U) {  // 4-bit
    weight := io.weightIn(3, 0).asSInt
    activ := io.activIn(3, 0).asSInt
  }.elsewhen(io.dataWidth === 2.U) {  // 8-bit
    weight := io.weightIn(7, 0).asSInt
    activ := io.activIn(7, 0).asSInt
  }.otherwise {  // 16-bit
    weight := io.weightIn.asSInt
    activ := io.activIn.asSInt
  }
  
  // MAC 操作
  val product = weight * activ
  io.partialOut := (io.partialIn.asSInt + product).asUInt
  io.activOut := io.activIn  // 流水传递
}
```

### 3. 状态机设计

```scala
object State extends ChiselEnum {
  val sIdle         = Value  // 空闲
  val sConfig       = Value  // 配置量化参数
  val sLoadWeights  = Value  // 加载权重
  val sUnpack       = Value  // 解包低位宽数据
  val sCompute      = Value  // 计算
  val sAccumScale   = Value  // 累加器缩放
  val sRequantize   = Value  // 重新量化
  val sClip         = Value  // 截断
  val sWriteback    = Value  // 写回结果
}
```

### 4. C Test Workload

创建测试文件：`tests/gemmini_configurable_test.c`

```c
#include <stdint.h>
#include <stdio.h>
#include "gemmini.h"

// ============ 测试用例 1：INT8 对称量化 ============
void test_int8_symmetric() {
  printf("=== Test 1: INT8 Symmetric Quantization ===\n");
  
  // 配置：8-bit, 8x8, WS 模式，对称量化
  uint64_t config = 
    (2 << 0) |      // data_width = 8-bit
    (8 << 4) |      // matrix_dim = 8
    (0 << 8) |      // WS mode
    (1 << 9) |      // signed
    (1 << 10);      // symmetric
  
  // 量化参数：scale = 0.1 (Q4.4 = 0x19)
  uint64_t quant_param = 
    (0x19 << 0) |   // weight_scale
    (0x19 << 8) |   // input_scale
    (0x19 << 16);   // output_scale
  
  gemmini_config_ex(GEMMINI_CONFIG_CFG, config, quant_param);
  
  // 加载 INT8 权重和输入
  load_int8_matrix(WEIGHT_ADDR, 8, 8);
  load_int8_matrix(INPUT_ADDR, 8, 8);
  
  // 启动计算
  gemmini_compute(8, 8, 8);
  gemmini_fence();
  
  // 验证结果
  verify_int8_result();
  printf("INT8 test PASSED\n\n");
}

// ============ 测试用例 2：INT4 模式（2倍打包） ============
void test_int4_packed() {
  printf("=== Test 2: INT4 Packed Mode ===\n");
  
  // 配置：4-bit, 8x8, OS 模式
  uint64_t config = 
    (1 << 0) |      // data_width = 4-bit
    (8 << 4) |      // matrix_dim = 8
    (1 << 8) |      // OS mode
    (1 << 9);       // signed
  
  gemmini_config_ex(GEMMINI_CONFIG_CFG, config, 0);
  
  // INT4 数据：每个字节打包 2 个元素
  // 例如：0x3A = [3][A] = [3][-6]
  uint8_t packed_weights[32];  // 8x8 matrix, packed to 64/2 = 32 bytes
  pack_int4_matrix(packed_weights, 8, 8);
  
  gemmini_load_packed(WEIGHT_ADDR, packed_weights, 32);
  
  // 计算并验证
  gemmini_compute(8, 8, 8);
  gemmini_fence();
  
  verify_int4_result();
  printf("INT4 test PASSED\n\n");
}

// ============ 测试用例 3：INT2 模式（4倍打包） ============
void test_int2_ultra_packed() {
  printf("=== Test 3: INT2 Ultra-Packed Mode ===\n");
  
  // 配置：2-bit, 8x8
  uint64_t config = (0 << 0) | (8 << 4) | (0 << 8) | (1 << 9);
  gemmini_config_ex(GEMMINI_CONFIG_CFG, config, 0);
  
  // INT2 数据：每个字节打包 4 个元素
  // 取值范围：-2, -1, 0, 1
  uint8_t ultra_packed[16];  // 8x8 = 64 elements, packed to 64/4 = 16 bytes
  
  for (int i = 0; i < 16; i++) {
    ultra_packed[i] = 0xE4;  // [11][10][01][00] = [-2][-1][1][0]
  }
  
  gemmini_load_packed(WEIGHT_ADDR, ultra_packed, 16);
  gemmini_load_packed(INPUT_ADDR, ultra_packed, 16);
  
  gemmini_compute(8, 8, 8);
  gemmini_fence();
  
  printf("INT2 test PASSED\n\n");
}

// ============ 测试用例 4：混合精度计算 ============
void test_mixed_precision() {
  printf("=== Test 4: Mixed Precision Computation ===\n");
  
  // 第一层：INT4
  gemmini_config_ex(GEMMINI_CONFIG_CFG, (1 << 0) | (8 << 4), 0);
  gemmini_compute(8, 8, 8);
  
  // 第二层：INT8（使用第一层的输出）
  gemmini_config_ex(GEMMINI_CONFIG_CFG, (2 << 0) | (8 << 4), 0);
  gemmini_compute(8, 8, 8);
  
  gemmini_fence();
  printf("Mixed precision test PASSED\n\n");
}

// ============ 测试用例 5：量化参数配置 ============
void test_quantization_params() {
  printf("=== Test 5: Quantization Parameter Configuration ===\n");
  
  // 非对称量化：zero_point = 128
  uint64_t config = (2 << 0) | (8 << 4) | (0 << 9) | (0 << 10);  // UINT8
  uint64_t quant = (0x10 << 0) | (0x10 << 8) | (0x10 << 16) | (128 << 24);
  
  gemmini_config_ex(GEMMINI_CONFIG_CFG, config, quant);
  
  // 启用重新量化和截断
  uint64_t special = 
    (4 << 0) |      // acc_shift = 4 (右移4位)
    (1 << 8) |      // requant_enable
    (1 << 9) |      // clip_enable
    (0 << 16) |     // clip_min = 0
    (255 << 24);    // clip_max = 255
  
  gemmini_config_ex(GEMMINI_CONFIG_SPECIAL, special, 0);
  
  gemmini_compute(8, 8, 8);
  gemmini_fence();
  
  // 验证输出在 [0, 255] 范围内
  verify_clipped_output(0, 255);
  printf("Quantization params test PASSED\n\n");
}

// ============ 性能测试：位宽对比 ============
void benchmark_bitwidth_comparison() {
  printf("=== Benchmark: Bitwidth Comparison ===\n");
  
  uint64_t start, end;
  int dim = 16;
  
  // INT16 基线
  gemmini_config_ex(GEMMINI_CONFIG_CFG, (3 << 0) | (dim << 4), 0);
  start = read_cycles();
  gemmini_compute(dim, dim, dim);
  gemmini_fence();
  end = read_cycles();
  printf("INT16: %llu cycles\n", end - start);
  
  // INT8
  gemmini_config_ex(GEMMINI_CONFIG_CFG, (2 << 0) | (dim << 4), 0);
  start = read_cycles();
  gemmini_compute(dim, dim, dim);
  gemmini_fence();
  end = read_cycles();
  printf("INT8:  %llu cycles\n", end - start);
  
  // INT4
  gemmini_config_ex(GEMMINI_CONFIG_CFG, (1 << 0) | (dim << 4), 0);
  start = read_cycles();
  gemmini_compute(dim, dim, dim);
  gemmini_fence();
  end = read_cycles();
  printf("INT4:  %llu cycles\n", end - start);
  
  // INT2
  gemmini_config_ex(GEMMINI_CONFIG_CFG, (0 << 0) | (dim << 4), 0);
  start = read_cycles();
  gemmini_compute(dim, dim, dim);
  gemmini_fence();
  end = read_cycles();
  printf("INT2:  %llu cycles\n", end - start);
  
  printf("\n");
}

// ============ 主函数 ============
int main() {
  printf("\n========================================\n");
  printf("Gemmini Configurable Systolic Array Test\n");
  printf("========================================\n\n");
  
  gemmini_flush(0);
  
  test_int8_symmetric();
  test_int4_packed();
  test_int2_ultra_packed();
  test_mixed_precision();
  test_quantization_params();
  benchmark_bitwidth_comparison();
  
  printf("All tests PASSED!\n");
  return 0;
}
```

## 验证标准

### 功能验证
✅ INT2/INT4/INT8/INT16 模式均能正确计算
✅ WS 和 OS 模式在所有位宽下均正常工作
✅ 对称和非对称量化正确实现
✅ 位宽打包和解包逻辑正确
✅ 重新量化和截断功能正常
✅ 混合精度计算流水线正确

### 性能要求
- INT2 吞吐量 = INT16 的 8 倍（理论值）
- INT4 吞吐量 = INT16 的 4 倍
- INT8 吞吐量 = INT16 的 2 倍
- PE 利用率 > 90%（8x8 阵列）

### 精度要求
- INT8 量化误差 < 0.5%
- INT4 量化误差 < 2%
- INT2 量化误差 < 5%

### 编译验证
✅ 所有 Chisel 代码编译通过
✅ C test workload 编译通过
✅ 综合后面积和时序满足约束

## 参考资料

1. **量化方法**：
   - Jacob et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
   
2. **低位宽推理**：
   - Rastegari et al. (2016). "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks"
   
3. **Gemmini 参考实现**：
   - `arch/src/main/scala/prototype/generated/matmul/` - 基础矩阵乘法
   - `arch/src/main/scala/prototype/vector/` - Ball 接口

## 开始实现

**第一步**：设计 FlexiblePE，支持多位宽 MAC 操作
**第二步**：实现位宽转换和打包/解包逻辑
**第三步**：实现量化参数配置和缩放单元
**第四步**：编写 C test workload 验证各种配置
**第五步**：性能测试和优化

