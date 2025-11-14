# 任务 4：支持 WS/OS/RS 三种数据流的脉动阵列

## 目标

设计并实现一个支持 **Weight Stationary (WS)**、**Output Stationary (OS)** 和 **Row Stationary (RS)** 三种数据流模式的脉动阵列，能够根据不同的计算模式和数据访问模式动态选择最优的数据流方式。

## 技术规格

### 1. 脉动阵列架构

#### 支持的三种数据流模式

**Weight Stationary (WS)**：
- **特点**：权重固定在 PE 中，激活流动
- **适用场景**：
  - 权重重用率高（卷积层）
  - 权重访存带宽受限
- **数据流**：
  - 权重：静态存储在 PE
  - 输入：水平流动（左→右）
  - 输出：垂直累积（上→下）

**Output Stationary (OS)**：
- **特点**：输出部分和固定在 PE 中
- **适用场景**：
  - 输出重用率高
  - 累加器访问能耗敏感
- **数据流**：
  - 权重：垂直流动（上→下）
  - 输入：水平流动（左→右）
  - 输出：静态累积在 PE

**Row Stationary (RS)**：
- **特点**：同时重用权重、输入和输出
- **适用场景**：
  - 综合重用需求
  - 内存带宽受限
  - 能效最优化
- **数据流**：
  - 权重：对角线流动
  - 输入：沿行流动
  - 输出：沿列累积
  - **核心优势**：平衡所有数据的重用，最小化 DRAM 访问

#### 阵列参数
```scala
val arrayDim = 16         // 16x16 脉动阵列
val dataWidth = 16        // 16-bit 数据
val accWidth = 32         // 32-bit 累加器
val localRFSize = 64      // 每个 PE 的本地寄存器堆大小（仅 RS 模式）
```

### 2. 数据流模式对比

| 特性 | WS | OS | RS |
|------|----|----|-----|
| 权重重用 | ✅ 高 | ❌ 无 | ✅ 中 |
| 输入重用 | ❌ 无 | ❌ 无 | ✅ 中 |
| 输出重用 | ❌ 无 | ✅ 高 | ✅ 中 |
| DRAM 访问 | 中 | 高 | **✅ 最低** |
| 能效 | 中 | 低 | **✅ 最高** |
| PE 复杂度 | 低 | 低 | **❌ 高**（需要本地 RF） |
| 适用场景 | Conv | FC | **通用** |

### 3. Ball 接口设计

#### 控制寄存器
```scala
// 基本配置 (rs1)
rs1(1:0)    : dataflow_mode   // 0=WS, 1=OS, 2=RS
rs1(7:2)    : array_dim       // 阵列维度 (4/8/16)
rs1(15:8)   : M_dim           // 矩阵 M 维度
rs1(23:16)  : N_dim           // 矩阵 N 维度
rs1(31:24)  : K_dim           // 矩阵 K 维度

// RS 模式专用配置 (rs2)
rs2(7:0)    : tile_m          // RS 模式 tile M 大小
rs2(15:8)   : tile_n          // RS 模式 tile N 大小
rs2(23:16)  : tile_k          // RS 模式 tile K 大小
rs2(24)     : enable_prefetch // 启用数据预取
rs2(25)     : enable_double_buffer // 启用双缓冲
rs2(31:26)  : pe_rf_size      // PE 本地寄存器堆大小

// 性能优化配置 (special)
special(0)  : enable_pipeline // 启用流水线
special(1)  : enable_broadcast// 启用广播优化
special(7:4): skew_factor     // RS 模式斜向延迟因子
special(15:8): drain_cycles   // 排空周期数
```

#### 状态寄存器
```scala
status(0)   : busy            // 计算中
status(2:1) : current_mode    // 当前数据流模式
status(15:8): pe_utilization  // PE 利用率 (%)
status(31:16): cycles_elapsed // 已消耗周期数
```

### 4. PE 设计（支持三种模式）

```scala
class TripleDataflowPE extends Module {
  val io = IO(new Bundle {
    val mode = Input(UInt(2.W))  // 0=WS, 1=OS, 2=RS
    
    // WS 模式
    val ws_weight = Input(SInt(16.W))      // 预加载权重
    val ws_activIn = Input(SInt(16.W))     // 输入激活
    val ws_activOut = Output(SInt(16.W))   // 输出激活
    val ws_psumIn = Input(SInt(32.W))      // 输入部分和
    val ws_psumOut = Output(SInt(32.W))    // 输出部分和
    
    // OS 模式
    val os_weightIn = Input(SInt(16.W))    // 流动权重
    val os_weightOut = Output(SInt(16.W))
    val os_activIn = Input(SInt(16.W))
    val os_activOut = Output(SInt(16.W))
    val os_psum = Reg(SInt(32.W))          // 固定部分和
    
    // RS 模式
    val rs_dataIn = Input(Vec(4, SInt(16.W)))   // 多路输入
    val rs_dataOut = Output(Vec(4, SInt(16.W))) // 多路输出
    val rs_localRF = Reg(Vec(64, SInt(16.W)))   // 本地寄存器堆
  })
  
  val result = Wire(SInt(32.W))
  
  switch(io.mode) {
    is(0.U) {  // WS 模式
      val product = io.ws_weight * io.ws_activIn
      result := io.ws_psumIn + product
      io.ws_psumOut := result
      io.ws_activOut := io.ws_activIn  // 激活流水传递
    }
    is(1.U) {  // OS 模式
      val product = io.os_weightIn * io.os_activIn
      io.os_psum := io.os_psum + product
      io.os_weightOut := io.os_weightIn  // 权重流水传递
      io.os_activOut := io.os_activIn    // 激活流水传递
    }
    is(2.U) {  // RS 模式
      // 复杂的 tile 内数据重用逻辑
      // 从本地 RF 读取，计算，写回 RF
      // 实现对角线数据流
    }
  }
}
```

### 5. 内存布局

#### WS 模式
```
SRAM[0x0000 - 0x0FFF]: 权重矩阵 (预加载)
SRAM[0x1000 - 0x1FFF]: 输入激活 (流式)
ACC[0x0000 - 0x0FFF]:  输出结果
```

#### OS 模式
```
SRAM[0x0000 - 0x0FFF]: 权重矩阵 (流式)
SRAM[0x1000 - 0x1FFF]: 输入激活 (流式)
ACC[0x0000 - 0x0FFF]:  输出结果
```

#### RS 模式（Tiling）
```
SRAM[0x0000 - 0x03FF]: Tile A (M_tile x K_tile)
SRAM[0x0400 - 0x07FF]: Tile B (K_tile x N_tile)
SRAM[0x0800 - 0x0BFF]: Tile A Buffer (双缓冲)
SRAM[0x0C00 - 0x0FFF]: Tile B Buffer (双缓冲)
ACC[0x0000 - 0x0FFF]:  输出 Tile C
```

## 实现要求

### 1. Chisel 代码结构

需要生成以下文件：

```
arch/src/main/scala/prototype/generated/triple/
├── TripleSystolicUnit.scala        // 主计算单元
├── TripleSystolicBall.scala        // Ball 包装器
├── TripleDataflowPE.scala          // 三模式 PE
├── DataflowController.scala        // 数据流控制器
├── WS_Controller.scala             // WS 模式专用控制
├── OS_Controller.scala             // OS 模式专用控制
├── RS_Controller.scala             // RS 模式专用控制（最复杂）
├── TileScheduler.scala             // RS 模式 Tile 调度器
└── PrefetchUnit.scala              // 预取单元
```

### 2. RS 模式详细设计

RS 模式是最复杂的，需要特别注意：

```scala
// RS 模式的 Tile 映射
// 对于 C = A × B，其中 A(M×K), B(K×N), C(M×N)
// 将计算分解为多个 Tile：
// C_tile(i,j) = Σ_k A_tile(i,k) × B_tile(k,j)

class RS_Controller(arrayDim: Int) extends Module {
  val io = IO(new Bundle {
    val config = Input(new RSConfig)
    val sramRead = Flipped(new SramReadPort)
    val sramWrite = Flipped(new SramWritePort)
    val peArray = Vec(arrayDim, Vec(arrayDim, new PEControlIO))
  })
  
  // 状态机
  val sIdle :: sLoadTileA :: sLoadTileB :: sCompute :: sWriteback :: Nil = Enum(5)
  val state = RegInit(sIdle)
  
  // Tile 索引
  val tile_i = Reg(UInt(8.W))
  val tile_j = Reg(UInt(8.W))
  val tile_k = Reg(UInt(8.W))
  
  // RS 数据流的关键：对角线调度
  // PE(i,j) 在时钟周期 t 处理：
  //   A(tile_i + i, tile_k + (t-i-j) % tile_k)
  //   B(tile_k + (t-i-j) % tile_k, tile_j + j)
  
  val cycle_counter = Reg(UInt(16.W))
  val skew_offset = Wire(Vec(arrayDim, Vec(arrayDim, UInt(8.W))))
  
  for (i <- 0 until arrayDim) {
    for (j <- 0 until arrayDim) {
      skew_offset(i)(j) := (cycle_counter - i.U - j.U) % io.config.tile_k
    }
  }
  
  // 双缓冲逻辑：在当前 tile 计算时预取下一个 tile
  val buffer_select = RegInit(false.B)
  when(state === sCompute && cycle_counter === io.config.tile_k - 1.U) {
    buffer_select := ~buffer_select
    // 触发预取下一个 tile
  }
}
```

### 3. 状态机设计（三模式统一）

```scala
object DataflowState extends ChiselEnum {
  val sIdle         = Value
  val sConfig       = Value  // 配置模式和参数
  
  // WS 专用状态
  val sWS_LoadWeights   = Value
  val sWS_Stream        = Value
  
  // OS 专用状态
  val sOS_ClearAcc      = Value
  val sOS_Stream        = Value
  val sOS_Drain         = Value
  
  // RS 专用状态（最复杂）
  val sRS_TileSchedule  = Value
  val sRS_LoadTileA     = Value
  val sRS_LoadTileB     = Value
  val sRS_Compute       = Value
  val sRS_Accumulate    = Value
  val sRS_NextTile      = Value
  
  val sWriteback        = Value
}
```

### 4. C Test Workload

创建测试文件：`tests/gemmini_triple_dataflow_test.c`

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gemmini.h"
#include "gemmini_testutils.h"

#define DIM 16
#define VERIFY_TOLERANCE 0.001

// ============ 辅助函数 ============
void print_matrix(int8_t *mat, int M, int N, const char *name) {
  printf("%s (%dx%d):\n", name, M, N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%4d ", mat[i * N + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void init_random_matrix(int8_t *mat, int M, int N) {
  for (int i = 0; i < M * N; i++) {
    mat[i] = (rand() % 20) - 10;  // [-10, 9]
  }
}

void matmul_reference(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = 0;
      for (int k = 0; k < K; k++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

int verify_result(int32_t *result, int32_t *golden, int M, int N) {
  int errors = 0;
  for (int i = 0; i < M * N; i++) {
    if (result[i] != golden[i]) {
      if (errors < 10) {  // 只打印前 10 个错误
        printf("ERROR at [%d]: got %d, expected %d\n", 
               i, result[i], golden[i]);
      }
      errors++;
    }
  }
  return errors;
}

// ============ 测试用例 1：WS 模式 ============
void test_ws_mode() {
  printf("========================================\n");
  printf("Test 1: Weight Stationary (WS) Mode\n");
  printf("========================================\n");
  
  int M = DIM, N = DIM, K = DIM;
  
  // 分配内存
  int8_t *A = malloc(M * K * sizeof(int8_t));
  int8_t *B = malloc(K * N * sizeof(int8_t));
  int32_t *C_result = malloc(M * N * sizeof(int32_t));
  int32_t *C_golden = malloc(M * N * sizeof(int32_t));
  
  // 初始化
  init_random_matrix(A, M, K);
  init_random_matrix(B, K, N);
  
  // 软件参考实现
  matmul_reference(A, B, C_golden, M, N, K);
  
  // 配置 Gemmini：WS 模式
  uint64_t config = 
    (0 << 0) |       // dataflow_mode = 0 (WS)
    (DIM << 2) |     // array_dim
    (M << 8) |       // M_dim
    (N << 16) |      // N_dim
    (K << 24);       // K_dim
  
  gemmini_config_ex(GEMMINI_CONFIG_EX, config, 0);
  
  // 加载数据到 SRAM
  gemmini_load_matrix(WEIGHT_ADDR, B, K, N);  // WS: 权重是 B
  gemmini_load_matrix(INPUT_ADDR, A, M, K);   // WS: 输入是 A
  
  // 执行计算
  uint64_t start = read_cycles();
  gemmini_compute_ws(M, N, K);
  gemmini_fence();
  uint64_t end = read_cycles();
  
  // 读取结果
  gemmini_store_matrix(OUTPUT_ADDR, C_result, M, N);
  
  // 验证
  int errors = verify_result(C_result, C_golden, M, N);
  
  printf("Cycles: %llu\n", end - start);
  printf("Result: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("\n");
  
  free(A); free(B); free(C_result); free(C_golden);
}

// ============ 测试用例 2：OS 模式 ============
void test_os_mode() {
  printf("========================================\n");
  printf("Test 2: Output Stationary (OS) Mode\n");
  printf("========================================\n");
  
  int M = DIM, N = DIM, K = DIM;
  
  int8_t *A = malloc(M * K * sizeof(int8_t));
  int8_t *B = malloc(K * N * sizeof(int8_t));
  int32_t *C_result = malloc(M * N * sizeof(int32_t));
  int32_t *C_golden = malloc(M * N * sizeof(int32_t));
  
  init_random_matrix(A, M, K);
  init_random_matrix(B, K, N);
  matmul_reference(A, B, C_golden, M, N, K);
  
  // 配置 Gemmini：OS 模式
  uint64_t config = 
    (1 << 0) |       // dataflow_mode = 1 (OS)
    (DIM << 2) |
    (M << 8) |
    (N << 16) |
    (K << 24);
  
  gemmini_config_ex(GEMMINI_CONFIG_EX, config, 0);
  
  // OS 模式：权重和激活都是流式的
  gemmini_load_matrix(WEIGHT_ADDR, B, K, N);
  gemmini_load_matrix(INPUT_ADDR, A, M, K);
  
  uint64_t start = read_cycles();
  gemmini_compute_os(M, N, K);
  gemmini_fence();
  uint64_t end = read_cycles();
  
  gemmini_store_matrix(OUTPUT_ADDR, C_result, M, N);
  
  int errors = verify_result(C_result, C_golden, M, N);
  
  printf("Cycles: %llu\n", end - start);
  printf("Result: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("\n");
  
  free(A); free(B); free(C_result); free(C_golden);
}

// ============ 测试用例 3：RS 模式（核心测试） ============
void test_rs_mode() {
  printf("========================================\n");
  printf("Test 3: Row Stationary (RS) Mode\n");
  printf("========================================\n");
  
  int M = DIM, N = DIM, K = DIM;
  
  int8_t *A = malloc(M * K * sizeof(int8_t));
  int8_t *B = malloc(K * N * sizeof(int8_t));
  int32_t *C_result = malloc(M * N * sizeof(int32_t));
  int32_t *C_golden = malloc(M * N * sizeof(int32_t));
  
  init_random_matrix(A, M, K);
  init_random_matrix(B, K, N);
  matmul_reference(A, B, C_golden, M, N, K);
  
  // 配置 Gemmini：RS 模式
  uint64_t config = 
    (2 << 0) |       // dataflow_mode = 2 (RS)
    (DIM << 2) |
    (M << 8) |
    (N << 16) |
    (K << 24);
  
  // RS 专用配置：tile 大小
  uint64_t rs_config = 
    (8 << 0) |       // tile_m = 8
    (8 << 8) |       // tile_n = 8
    (8 << 16) |      // tile_k = 8
    (1 << 24) |      // enable_prefetch
    (1 << 25) |      // enable_double_buffer
    (64 << 26);      // pe_rf_size = 64
  
  gemmini_config_ex(GEMMINI_CONFIG_EX, config, 0);
  gemmini_config_ex(GEMMINI_CONFIG_RS, rs_config, 0);
  
  // RS 模式：以 tile 方式加载
  gemmini_load_matrix(INPUT_ADDR, A, M, K);
  gemmini_load_matrix(WEIGHT_ADDR, B, K, N);
  
  uint64_t start = read_cycles();
  gemmini_compute_rs(M, N, K);
  gemmini_fence();
  uint64_t end = read_cycles();
  
  gemmini_store_matrix(OUTPUT_ADDR, C_result, M, N);
  
  int errors = verify_result(C_result, C_golden, M, N);
  
  printf("Cycles: %llu\n", end - start);
  printf("Result: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("\n");
  
  free(A); free(B); free(C_result); free(C_golden);
}

// ============ 测试用例 4：三模式性能对比 ============
void benchmark_dataflow_comparison() {
  printf("========================================\n");
  printf("Benchmark: Dataflow Mode Comparison\n");
  printf("========================================\n");
  
  int sizes[] = {4, 8, 16, 32};
  
  printf("%-8s %-12s %-12s %-12s\n", "Size", "WS (cycles)", "OS (cycles)", "RS (cycles)");
  printf("----------------------------------------------------\n");
  
  for (int s = 0; s < 4; s++) {
    int dim = sizes[s];
    
    int8_t *A = malloc(dim * dim * sizeof(int8_t));
    int8_t *B = malloc(dim * dim * sizeof(int8_t));
    init_random_matrix(A, dim, dim);
    init_random_matrix(B, dim, dim);
    
    gemmini_load_matrix(INPUT_ADDR, A, dim, dim);
    gemmini_load_matrix(WEIGHT_ADDR, B, dim, dim);
    
    // WS
    gemmini_config_ex(GEMMINI_CONFIG_EX, (0 << 0) | (dim << 8), 0);
    uint64_t ws_start = read_cycles();
    gemmini_compute_ws(dim, dim, dim);
    gemmini_fence();
    uint64_t ws_cycles = read_cycles() - ws_start;
    
    // OS
    gemmini_config_ex(GEMMINI_CONFIG_EX, (1 << 0) | (dim << 8), 0);
    uint64_t os_start = read_cycles();
    gemmini_compute_os(dim, dim, dim);
    gemmini_fence();
    uint64_t os_cycles = read_cycles() - os_start;
    
    // RS
    gemmini_config_ex(GEMMINI_CONFIG_EX, (2 << 0) | (dim << 8), 0);
    gemmini_config_ex(GEMMINI_CONFIG_RS, (8 << 0) | (8 << 8) | (8 << 16), 0);
    uint64_t rs_start = read_cycles();
    gemmini_compute_rs(dim, dim, dim);
    gemmini_fence();
    uint64_t rs_cycles = read_cycles() - rs_start;
    
    printf("%-8dx%-2d %-12llu %-12llu %-12llu\n", 
           dim, dim, ws_cycles, os_cycles, rs_cycles);
    
    free(A); free(B);
  }
  
  printf("\n");
}

// ============ 测试用例 5：能耗对比（模拟） ============
void test_energy_estimation() {
  printf("========================================\n");
  printf("Test 5: Energy Estimation (Simulated)\n");
  printf("========================================\n");
  
  // 假设的能耗模型（单位：pJ）
  const double SRAM_READ_ENERGY = 5.0;
  const double DRAM_READ_ENERGY = 200.0;
  const double MAC_ENERGY = 0.2;
  const double ACC_WRITE_ENERGY = 1.0;
  
  int M = 16, N = 16, K = 16;
  int total_ops = M * N * K;
  
  // WS 模式能耗估算
  double ws_weight_read = K * N * SRAM_READ_ENERGY;  // 权重预加载
  double ws_input_read = M * K * DRAM_READ_ENERGY;   // 输入流式读取
  double ws_mac = total_ops * MAC_ENERGY;
  double ws_output_write = M * N * ACC_WRITE_ENERGY;
  double ws_total = ws_weight_read + ws_input_read + ws_mac + ws_output_write;
  
  // OS 模式能耗估算
  double os_weight_read = M * K * N * SRAM_READ_ENERGY / K;  // 权重重复读取
  double os_input_read = M * K * N * SRAM_READ_ENERGY / M;
  double os_mac = total_ops * MAC_ENERGY;
  double os_output_write = M * N * ACC_WRITE_ENERGY;
  double os_total = os_weight_read + os_input_read + os_mac + os_output_write;
  
  // RS 模式能耗估算（最优）
  double rs_dram_read = (M * K + K * N + M * N) * DRAM_READ_ENERGY;  // 每个数据只读一次
  double rs_mac = total_ops * MAC_ENERGY;
  double rs_rf_access = total_ops * 0.05;  // 本地 RF 访问能耗很低
  double rs_total = rs_dram_read + rs_mac + rs_rf_access;
  
  printf("Energy (nJ) for %dx%d matrix multiplication:\n", M, N);
  printf("  WS mode: %.2f nJ\n", ws_total / 1000.0);
  printf("  OS mode: %.2f nJ\n", os_total / 1000.0);
  printf("  RS mode: %.2f nJ (%.1f%% of WS)\n", 
         rs_total / 1000.0, (rs_total / ws_total) * 100.0);
  printf("\n");
  printf("RS mode achieves %.1fx better energy efficiency than WS\n", 
         ws_total / rs_total);
  printf("\n");
}

// ============ 主函数 ============
int main() {
  printf("\n");
  printf("========================================\n");
  printf("Gemmini Triple Dataflow Systolic Array\n");
  printf("========================================\n\n");
  
  gemmini_flush(0);
  
  test_ws_mode();
  test_os_mode();
  test_rs_mode();
  benchmark_dataflow_comparison();
  test_energy_estimation();
  
  printf("========================================\n");
  printf("All tests completed!\n");
  printf("========================================\n");
  
  return 0;
}
```

## 验证标准

### 功能验证
✅ WS 模式正确实现并验证
✅ OS 模式正确实现并验证
✅ RS 模式正确实现并验证（重点）
✅ 三种模式结果一致
✅ Tile 调度正确（RS 模式）
✅ 双缓冲和预取功能正常

### 性能要求
- RS 模式 DRAM 访问量 = WS 的 30-50%
- RS 模式能效 = WS 的 1.5-2.5 倍
- 大矩阵（64x64）下 RS 优势显著
- PE 利用率 > 85%（所有模式）

### 面积和复杂度
- PE 面积：RS 模式 PE ≈ WS 模式 PE × 1.5（增加本地 RF）
- 控制逻辑：RS 模式控制器比 WS 复杂 3-4 倍
- 可接受的面积开销：< 50%（相比单模式）

### 编译验证
✅ 所有 Chisel 代码编译通过
✅ C test workload 编译通过
✅ 综合后时序满足约束
✅ 功能仿真全部通过

## 参考资料

1. **Row Stationary 论文**：
   - Chen et al. (2016). "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks"
   - **核心思想**：通过空间展开和时间复用，最大化所有数据的重用
   
2. **数据流对比**：
   - Chen et al. (2017). "Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows: A Data-Centric Approach"
   
3. **Gemmini 参考**：
   - Genc et al. (2021). "Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration"
   
4. **本地参考实现**：
   - `arch/src/main/scala/prototype/generated/matmul/` - 基础矩阵乘法
   - `arch/src/main/scala/prototype/vector/` - Ball 接口

## 开始实现

**第一步**：实现 WS 和 OS 模式（相对简单）
**第二步**：设计 RS 模式的 PE 和本地寄存器堆
**第三步**：实现 RS 模式的 Tile 调度器（最复杂）
**第四步**：实现对角线数据流和 skew 机制
**第五步**：编写 C test workload 验证三种模式
**第六步**：性能和能耗对比测试

## 重点难点

⚠️ **RS 模式的实现难度最高**：
1. 需要实现复杂的 Tile 映射算法
2. 对角线数据流需要精确的时序控制
3. 双缓冲和预取逻辑复杂
4. 调试困难（需要可视化工具）

建议：
- 先实现并验证 WS 和 OS 模式
- RS 模式可以从小规模（4x4）开始验证
- 使用波形图仔细检查数据流时序
- 编写详细的单元测试

