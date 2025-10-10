# README_step

> by -王博涵
>
> 本文档会逐步更新，作者也在不断出现的问题的中尝试解决并总结。

本文档用于说明一个完整的`buckyball`开发流程的步骤顺序和各种问题的解决思路。我们以构建一个用于执行`relu()`函数的ball算子模块为例，第一步我们需要完成该模块的硬件代码编写，即编写`scala`语言的chisel硬件语言代码，并生成对应的`verilog`代码；第二步我们需要编写测试软件去实现`relu()`，需要一个在`cpu`执行软件代码的对照函数和一个在我们第一步编写的专用硬件执行软件代码的实验函数，测试结果一致即成功。第三步，在硬件层上进行仿真，查看波形图进行debug。此外，还有一些其他的细节，比如编译文档的更改，指令集的更新等细节，下文会依次说明。

在开发中遇到问题，可访问[DangoSys/buckyball | DeepWiki](https://deepwiki.com/DangoSys/buckyball)或者[项目概览 - Buckyball Technical Documentation](https://dangosys.github.io/buckyball/index.html)

Chisel学习资源：[binder](https://mybinder.org/v2/gh/freechipsproject/chisel-bootcamp/master)

在正式开始之前，我们先启动环境：

```
cd /path/to/buckyball  
source ./env.sh
// 全文所有路径都是以./buckyball为起点的相对路径
```

## 一、 编写Chisel硬件模块

在 `arch/src/main/scala/prototype/` 目录下创建`ReLU`加速器的Chisel实现。参考现有的加速器结构，建议在 `prototype/` 下创建新的子目录，例如 `prototype/relu/Relu.scala`，编写硬件代码。此外，还要创建一个新的 Ball 执行单元`class ReluUnit`来处理` ReLU`操作。

## 二、 编写测试软件与编译设置

### 1. 创建测试文件

在 `bb-tests/workloads/src/CTest/` 下创建 `relu_test.c`, 编写测试代码，代码中核心函数会执行`void bb_relu(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter);` 下文中要注意该函数的声明和定义。

### 2. 修改CMakeLists.txt

在 `bb-tests/workloads/src/CTest/CMakeLists.txt` 中添加测试目标： CMakeLists.txt:120-127

```
add_cross_platform_test_target(ctest_relu_test relu_test.c)
```

并在总构建目标中添加： CMakeLists.txt:137-162

```
add_custom_target(buckyball-CTest-build ALL DEPENDS  
  # ... 其他测试 ...  
  ctest_relu_test  
  COMMENT "Building all workloads for Buckyball"  
  VERBATIM)
```

### 3. 需要添加ReLU指令API

#### a. isa.h

在 `bb-tests/workloads/lib/bbhw/isa/isa.h` 中添加`ReLU`指令的声明：` isa.h:33-43`

在 `InstructionType` 枚举中添加：

```
RELU_FUNC7 = 35,  // 0x23 - ReLU function code (或您选择的其他值)
```

在函数声明部分添加： `isa.h:72-73`

```
void bb_relu(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter);
```

#### b.  isa.c

在`bb-tests/workloads/lib/bbhw/isa`添加`35_relu.c`,在里面实现`void bb_relu(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter)`

在 `bb-tests/workloads/lib/bbhw/isa/isa.c` 中添加声明: `isa.c:53-76`

```
case RELU_FUNC7:
	return &relu_config; 
```

在`isa.c:37-47`

```
extern const InstructionConfig relu_config;
```

### 4. 更新 CMakeLists.txt

在 `bb-tests/workloads/lib/bbhw/isa/CMakeLists.txt` 中的三个编译命令中都添加 `35_relu.c` 的编译和链接:

1. **Linux 版本**:在 `add_custom_command` 的 `COMMAND` 中添加:

   ```
   && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/35_relu.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-35_relu.o
   ```

   并在 `ar rcs` 命令中添加 `linux-35_relu.o`

2. **Baremetal 版本**:在 `add_custom_command` 的 `COMMAND` 中添加:

   ```
   && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/35_relu.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-35_relu.o
   ```

   并在 `ar rcs` 命令中添加 `baremetal-35_relu.o`

3. **x86 版本**:在 `add_custom_command` 的 `COMMAND` 中添加:

   ```
   && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/35_relu.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-35_relu.o
   ```

   并在 `ar rcs` 命令中添加 `x86-35_relu.o`

## 三、硬件指令解码

需要在**硬件端**添加 ReLU 指令的支持。软件端的 `bb_relu` 函数已经实现,但硬件解码器还不认识这个指令。

#### 1. 在 Ball 域解码器中添加 ReLU 指令

找到 `arch/src/main/scala/examples/toy/balldomain/DomainDecoder.scala` 文件,在解码列表中添加 ReLU 指令的解码项。参考其他指令(如 TRANSPOSE_FUNC7 = 35)的实现方式,您需要:

```
// 在 BallDecodeFields 的 ListLookup 中添加  
RELU_BITPAT -> List(Y, ...) // 根据 ReLU 指令的具体需求填写解码字段，列表参数的数量一定要一致，可以参考其他指令
```

#### 2. 在 DISA.scala 中定义 RELU_BITPAT

在 `arch/src/main/scala/examples/toy/balldomain/DISA.scala` 中添加 ReLU 指令的位模式定义:

```
val RELU_BITPAT = BitPat("b0100011") // func7 = 35 = 0x23
```

#### 3. 编写ReluBall的接口文件

在`arch/src/main/scala/examples/toy/balldomain`文件中创建`reluball`文件夹，进入文件夹后创建`ReluBall.scala`编写接口代码。

#### 4. 添加ReLuBall生成器并进行注册

a. 在`arch/src/main/scala/examples/toy/balldomain/bbus/busRegister.scala`文件中找到并添加ReLuBall的新ID。

```
class BBusModule(implicit b: CustomBuckyBallConfig, p: Parameters)
    extends BBus(
      // 定义要注册的Ball设备生成器
      Seq(
        () => new examples.toy.balldomain.vecball.VecBall(0),
        () => new examples.toy.balldomain.matrixball.MatrixBall(1),
        () => new examples.toy.balldomain.im2colball.Im2colBall(2),
        () => new examples.toy.balldomain.transposeball.TransposeBall(3),
        () =>
          new examples.toy.balldomain.reluball.ReluBall(4) // Ball ID 4 - 新添加
      )
    ) {
  override lazy val desiredName = "BBusModule"
}
```

b. 在`arch/src/main/scala/examples/toy/balldomain/rs/rsRegister.scala`文件中进行Ball的注册与连接:

```
class BallRSModule(implicit b: CustomBuckyBallConfig, p: Parameters)
    extends BallReservationStation(
      // 定义要注册的Ball设备信息
      Seq(
        BallRsRegist(ballId = 0, ballName = "VecBall"),
        BallRsRegist(ballId = 1, ballName = "MatrixBall"),
        BallRsRegist(ballId = 2, ballName = "Im2colBall"),
        BallRsRegist(ballId = 3, ballName = "TransposeBall"),
        BallRsRegist(ballId = 4, ballName = "ReluBall") // Ball ID 4 - 新添加
      )
    ) {
  override lazy val desiredName = "BallRSModule"
}
```

## 四、 测试操作步骤

### 步骤1: 编译测试程序

```
cd bb-tests/build  
rm -rf * 
cmake -G Ninja ../  
```

**Warning**：执行`rm -rf * `之前一定要检查是否在`bb-tests/build `目录里面，否则在错误的文件夹里面强制删除重要文件将会带来灾难！

若灾难发生了，可以从GitHub重新拉取初始文档，但自己在服务器端更新的文件无法复原。

```
ninja ctest_relu_test // 软件编译
ninja sync-bin  // 同步二进制文件
```

若`ninja ctest_relu_test`执行后报错，这是软件编译没有通过，请检查**”二、 编写测试软件“**等相关文件。

### 步骤2: 生成Verilog

```
cd buckyball  
bbdev verilator --verilog
```

若`bbdev verilator --verilog`执行后报错，这是硬件编译没有通过，请检查**“一、 编写Chisel硬件模块 三、编译适配准备”**相关文件。

### 步骤3: 编译仿真器

```
bbdev verilator --build "--jobs 16"
```

### 步骤4: 运行仿真

```
bbdev verilator --sim "--binary ctest_relu_test_singlecore-baremetal --b
```

若`bbdev verilator --verilog`执行后报错，这是硬件系统有超时，卡死等问题，请检查**一、 编写Chisel硬件模块**相关文件。

### 步骤5：查看仿真文件

在`arch/waveform/仿真文件名(E.g.2025-10-08-00-03-ctest_vecunit_matmul_random1_singlecore-baremetal)`中，将`waveform.fst`文件用本地系统的`Filezilla`等软件下载到本地上，在本地的仿真波形查看器(E.g. GTKWave)进行波形查看。

注意，仿真文件所在的文件夹只能由`waveform.fst`一个文件，若存在`waveform.fst.hier`文件则代表仿真失败.

若波形没有满足理论情况，请在软件测试代码没有问题的情况下检查**一、 编写Chisel硬件模块**相关文件。

软件代码是否有问题可以参考其在`cpu`执行的结果，可以更改`relu_test.c`文件暂时完全移除硬件加速器调用，只测试 CPU 版本。
