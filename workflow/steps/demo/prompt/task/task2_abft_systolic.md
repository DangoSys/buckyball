# 任务 2：支持 WS/OS 的 ABFT 可靠性脉动阵列

## 目标

设计并实现一个支持 **Weight Stationary (WS)** 和 **Output Stationary (OS)** 两种数据流模式的脉动阵列，并集成 **ABFT (Algorithm-Based Fault Tolerance)** 可靠性机制。

## 技术规格

### 1. 脉动阵列架构

#### 支持的数据流模式

**Weight Stationary (WS)**：
- 权重保持在 PE (Processing Element) 中
- 输入激活从左向右流动
- 部分和向下累积
- 适合卷积神经网络的卷积层

**Output Stationary (OS)**：
- 输出部分和保持在 PE 中
- 权重从上向下流动
- 激活从左向右流动
- 减少累加器访问，节省能量

#### 阵列参数
```scala
val arrayDim = 4          // 4x4 脉动阵列
val dataWidth = 8         // 8-bit 数据位宽
val accWidth = 32         // 32-bit 累加器位宽
```

### 2. ABFT 可靠性机制

#### 校验和方法

**行校验和（Row Checksum）**：
```
C_row(i) = Σ A(i,j)  // 对矩阵 A 的第 i 行求和
```

**列校验和（Column Checksum）**：
```
C_col(j) = Σ A(i,j)  // 对矩阵 A 的第 j 列求和
```

**全校验和（Full Checksum）**：
```
C_full = Σ C_row(i) = Σ C_col(j)
```

#### 错误检测机制

1. **在线校验**：
   - 在计算过程中同步计算校验和
   - 比较计算结果的校验和与预期校验和
   
2. **错误定位**：
   - 行校验失败 → 定位到错误的行
   - 列校验失败 → 定位到错误的列
   - 交叉定位精确到错误的 PE

3. **错误恢复**：
   - 重新计算错误的 tile
   - 使用冗余计算路径
   - 记录错误统计信息

### 3. Ball 接口设计

#### 控制寄存器
```scala
// 基本配置
rs1(15:0)   : matrixDim      // 矩阵维度 (N)
rs1(16)     : dataflow_mode  // 0=WS, 1=OS
rs1(17)     : abft_enable    // 启用 ABFT
rs1(18)     : checksum_mode  // 0=行列校验, 1=全校验

// ABFT 配置
rs2(31:0)   : error_mask     // 错误注入掩码（用于测试）
```

#### 状态寄存器
```scala
status(0)   : busy           // 计算中
status(1)   : error_detected // 检测到错误
status(7:4) : error_row      // 错误行号
status(11:8): error_col      // 错误列号
status(31:16): error_count   // 累计错误次数
```

### 4. 内存布局

#### 输入数据
- **矩阵 A**：`SRAM[0x0000 - 0x0FFF]`
- **矩阵 B**：`SRAM[0x1000 - 0x1FFF]`
- **行校验和 A**：`SRAM[0x2000 - 0x20FF]`
- **列校验和 B**：`SRAM[0x2100 - 0x21FF]`

#### 输出数据
- **矩阵 C**：`ACC[0x0000 - 0x0FFF]`
- **校验和 C**：`ACC[0x1000 - 0x10FF]`
- **错误日志**：`ACC[0x1100 - 0x11FF]`

## 实现要求

### 1. Chisel 代码结构

需要生成以下文件：

```
arch/src/main/scala/prototype/generated/abft/
├── ABFTSystolicUnit.scala       // 主计算单元
├── ABFTSystolicBall.scala       // Ball 包装器
├── ProcessingElement.scala      // PE 实现
├── ChecksumUnit.scala           // 校验和计算单元
└── ErrorDetector.scala          // 错误检测逻辑
```

### 2. 状态机设计

```scala
object State extends ChiselEnum {
  val sIdle         = Value  // 空闲
  val sLoadWeights  = Value  // 加载权重（WS 模式）
  val sLoadInputs   = Value  // 加载输入
  val sCompute      = Value  // 计算
  val sChecksum     = Value  // 校验和验证
  val sErrorCheck   = Value  // 错误检查
  val sErrorCorrect = Value  // 错误纠正
  val sWriteback    = Value  // 写回结果
}
```

### 3. C Test Workload

创建测试文件：`tests/gemmini_abft_test.c`

```c
// 测试用例 1：无错误的矩阵乘法
void test_abft_no_error() {
  // WS 模式，4x4 矩阵乘法
  setup_matrix_a(4, 4);
  setup_matrix_b(4, 4);
  compute_checksums();
  
  // 配置 Ball：WS 模式，启用 ABFT
  write_csr(GEMMINI_ABFT_CONFIG, 
    (4 << 0) | (0 << 16) | (1 << 17));
  
  // 启动计算
  write_csr(GEMMINI_ABFT_CMD, CMD_START);
  
  // 等待完成
  while (read_csr(GEMMINI_ABFT_STATUS) & 0x1);
  
  // 检查结果
  verify_result();
  assert_no_error();
}

// 测试用例 2：错误注入与检测
void test_abft_error_detection() {
  // 注入单比特错误
  write_csr(GEMMINI_ABFT_ERROR_INJECT, 
    (2 << 0) | (3 << 4));  // 行2，列3
  
  // 启动计算
  compute_with_abft();
  
  // 验证错误被检测到
  uint32_t status = read_csr(GEMMINI_ABFT_STATUS);
  assert(status & (1 << 1));  // error_detected
  assert((status >> 4) & 0xF == 2);   // error_row = 2
  assert((status >> 8) & 0xF == 3);   // error_col = 3
}

// 测试用例 3：OS 模式测试
void test_abft_os_mode() {
  // 配置为 OS 模式
  write_csr(GEMMINI_ABFT_CONFIG, 
    (4 << 0) | (1 << 16) | (1 << 17));
  
  compute_and_verify();
}

// 性能测试
void benchmark_abft_overhead() {
  uint64_t start, end;
  
  // 不启用 ABFT
  start = read_cycle();
  compute_without_abft();
  end = read_cycle();
  printf("Without ABFT: %llu cycles\n", end - start);
  
  // 启用 ABFT
  start = read_cycle();
  compute_with_abft();
  end = read_cycle();
  printf("With ABFT: %llu cycles\n", end - start);
  
  // 计算开销
  printf("ABFT overhead: %.2f%%\n", 
    (end - start) * 100.0 / (end - start));
}
```

## 验证标准

### 功能验证
✅ WS 模式下正确计算矩阵乘法
✅ OS 模式下正确计算矩阵乘法
✅ 校验和计算正确
✅ 单比特错误能被检测并定位
✅ 多比特错误能被检测
✅ 错误统计功能正常

### 性能要求
- ABFT 开销 < 15%（相比无 ABFT）
- 错误检测延迟 < 10 个周期
- 支持 128x128 矩阵计算

### 编译验证
✅ 所有 Chisel 代码编译通过
✅ C test workload 编译通过
✅ 集成测试通过

## 参考资料

1. **ABFT 原理**：
   - Huang, K. H., & Abraham, J. A. (1984). "Algorithm-Based Fault Tolerance for Matrix Operations"
   
2. **脉动阵列**：
   - Kung, H. T., & Leiserson, C. E. (1980). "Systolic Arrays"
   
3. **Gemmini 架构**：
   - `arch/src/main/scala/prototype/generated/matmul/` - 基础矩阵乘法实现
   - `arch/src/main/scala/prototype/vector/` - Ball 接口参考

## 开始实现

**第一步**：阅读参考代码，理解 Ball 接口和脉动阵列基本结构
**第二步**：设计 PE 和校验和计算单元
**第三步**：实现 ABFT 错误检测逻辑
**第四步**：编写 C test workload
**第五步**：编译验证并性能测试

