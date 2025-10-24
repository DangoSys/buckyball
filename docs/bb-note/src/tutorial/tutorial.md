# Tutorial for buckyball

> by -王博涵
>
> 本文档会逐步更新，作者也在不断出现的问题的中尝试解决并总结。

本文档用于说明一个完整的`buckyball`开发流程的步骤顺序和各种问题的解决思路。我们以构建一个用于执行`relu()`函数的ball算子模块为例：

第一步我们需要完成该模块的硬件代码编写，即编写`scala`语言的chisel硬件语言代码，并生成对应的`verilog`代码。

第二步我们需要编写测试软件去实现`relu()`，可以编写一个在`cpu`执行软件代码的对照函数和一个在我们第一步编写的专用硬件执行软件代码的实验函数，测试结果一致即成功，或者做第三步来测试。

第三步，在硬件层上进行仿真，查看波形图进行debug。此外，还有一些其他的细节，比如编译文档的更改，指令集的更新等细节，下文会依次说明。

在开发中遇到问题，可访问[DangoSys/buckyball | DeepWiki](https://deepwiki.com/DangoSys/buckyball)或者[项目概览 - Buckyball Technical Documentation](https://dangosys.github.io/buckyball/index.html)

Chisel学习资源：[binder](https://mybinder.org/v2/gh/freechipsproject/chisel-bootcamp/master)

在正式开始之前，我们先启动环境：

```
cd /path/to/buckyball
source env.sh
// source ./env.sh 若报错试试这个
// 全文所有路径都是以./buckyball为起点的相对路径
```

## 一、 编写Chisel硬件模块

在 `arch/src/main/scala/prototype/` 目录下创建`ReLU`加速器的Chisel实现。参考现有的加速器结构，建议在 `prototype/` 下创建新的子目录，例如 `prototype/relu/Relu.scala`，编写硬件代码。

## 二、硬件指令解码

接下来对硬件指令进行解码，需要在**硬件端**添加 ReLU 指令的支持，让硬件解码器认识这个指令，编写注册该ball的指令集。

该工作主要分为下面五个方面：

- 指令枚举（DISA）定义了 func7 → 指令名（RELU）
- 解码器（DomainDecoder）定义了 func7 → 解码规则（读/写/地址/iter）→ BID（例如 4）
- 总线注册（busRegister）定义了 BID → 实际的 Ball 实例（索引为 4 的 ReluBall）
- 保留站注册（rsRegister）用于 RS/发射描述，与 BID 对齐，便于系统的发射/完成管理与调试
若任一环缺失或不一致，都会导致 ReLU 这条指令无法正确被识别/路由/落到实际硬件执行。
- 创建一个新的 Ball 执行单元`class ReluUnit`来处理` ReLU`操作。

#### 1. 在 DISA.scala 中定义 RELU_BITPAT

`arch/src/main/scala/examples/toy/balldomain/DISA.scala` 定义 Ball 指令的 funct7 编码（BitPat），比如 TRANSPOSE、IM2COL 等。可以视作“指令集的枚举表”，供解码器匹配。

在此文件中添加 ReLU 指令的位模式定义:

```
val RELU_BITPAT = BitPat("b0100110") // func7 = 38 = 0x23
```

#### 2. 在 Ball 域解码器中添加 ReLU 指令

 `arch/src/main/scala/examples/toy/balldomain/DomainDecoder.scala` 是Ball域解码器。
作用如下：
- 输入：来自全局解码的 PostGDCmd（已经判断这是 Ball 类别的命令）。
- 输出：结构化的 BallDecodeCmd，包括：
  - 是否使用 op1/op2、是否写回 scratchpad、操作数是否来自 scratchpad
  - 操作数/写回的 bank 与地址
  - 迭代次数 iter
  - 目标 Ball ID（BID）
  - 其它专用字段 special 等
- 内部通过 ListLookup(func7, ...)，把不同 funct7 的指令映射到一套布尔开关和字段抽取规则。

此文件中在解码列表中添加 ReLU 指令的解码项。参考其他指令(如 TRANSPOSE_FUNC7 = 38)的实现方式,您需要:

```
// 在 BallDecodeFields 的 ListLookup 中添加
RELU                 -> List(Y,N,Y,Y,N, rs1(spAddrLen-1,0), 0.U(spAddrLen.W), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen), 7.U, rs2(63,spAddrLen + 10), Y) // 根据 ReLU 指令的具体需求填写解码字段，列表参数的数量一定要一致，可以参考其他指令
```





#### 3. 添加ReLuBall生成器并进行注册

a. `arch/src/main/scala/examples/toy/balldomain/bbus/busRegister.scala`是Ball 总线注册表，用一个 `Seq(() => new 某Ball(...)) `注册系统里实际要实例化的 Ball 模块。

在此文件中找到并添加ReLuBall的新ID。

```
class BBusModule(implicit b: CustomBuckyBallConfig, p: Parameters)
    extends BBus(
      // 定义要注册的Ball设备生成器
      Seq(
        () => new examples.toy.balldomain.vecball.VecBall(0),
        () => new examples.toy.balldomain.matrixball.MatrixBall(1),
        () => new examples.toy.balldomain.im2colball.Im2colBall(2),
        () => new examples.toy.balldomain.transposeball.TransposeBall(3),
        ...
        () =>new examples.toy.balldomain.reluball.ReluBall(7) // Ball ID 7 - 新添加
      )
    ) {
  override lazy val desiredName = "BBusModule"
}
```

b. `arch/src/main/scala/examples/toy/balldomain/rs/rsRegister.scala`是"Ball 保留站"的注册表，用一个列表注册系统里有哪些 Ball（按 ballId 指定 ID、指定名称）。保留站（RS）负责管理 Ball 的发射、占用、完成等元信息，通常也用于可视化/统计、命名与日志。

在此文件中进行ReluBall的注册:

```
class BallRSModule(implicit b: CustomBuckyBallConfig, p: Parameters)
    extends BallReservationStation(
      // 定义要注册的Ball设备信息
      Seq(
        BallRsRegist(ballId = 0, ballName = "VecBall"),
        BallRsRegist(ballId = 1, ballName = "MatrixBall"),
        BallRsRegist(ballId = 2, ballName = "Im2colBall"),
        BallRsRegist(ballId = 3, ballName = "TransposeBall"),
        ...
        BallRsRegist(ballId = 7, ballName = "ReluBall") // Ball ID 7 - 新添加
      )
    ) {
  override lazy val desiredName = "BallRSModule"
}
```
#### 4. 编写ReluBall的接口文件

在`arch/src/main/scala/examples/toy/balldomain`文件中创建`reluball`文件夹，进入文件夹后创建`ReluBall.scala`编写接口代码。

## 三、 编写测试软件与编译设置

### 1. 创建测试文件

在 `bb-tests/workloads/src/CTest/toy/` 下创建 `relu_test.c`, 编写测试代码，代码中核心函数会执行`void bb_relu(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter);` 下文中要注意该函数的声明和定义。

### 2. 修改CMakeLists.txt

在 `bb-tests/workloads/src/CTest/toy/CMakeLists.txt` 中添加测试目标： CMakeLists.txt:120-127

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

- 在 `bb-tests/workloads/lib/bbhw/isa/isa.h` 中添加`ReLU`指令的声明：` isa.h:33-43`

- 在 `InstructionType` 枚举中添加：

```
RELU_FUNC7 = 38,  // 0x26 - ReLU function code (或您选择的其他值)
```

- 在函数声明部分添加： `isa.h:72-73`

```
void bb_relu(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter);
```

#### b.  isa.c

- 在`bb-tests/workloads/lib/bbhw/isa`添加`38_relu.c`,在里面实现`void bb_relu(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter)`

- 在 `bb-tests/workloads/lib/bbhw/isa/isa.c` 中添加声明: `isa.c:53-76`

```
case RELU_FUNC7:
	return &relu_config;
```

- 在`isa.c:37-47`

```
extern const InstructionConfig relu_config;
```

### 4. 更新 CMakeLists.txt

在 `bb-tests/workloads/lib/bbhw/isa/CMakeLists.txt` 中的三个编译命令中都添加 `38_relu.c` 的编译和链接:

1. **Linux 版本**:在 `add_custom_command` 的 `COMMAND` 中添加:

   ```
   && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/38_relu.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-38_relu.o
   ```

   并在 `ar rcs` 命令中添加 `linux-38_relu.o`

2. **Baremetal 版本**:在 `add_custom_command` 的 `COMMAND` 中添加:

   ```
   && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/38_relu.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-38_relu.o
   ```

   并在 `ar rcs` 命令中添加 `baremetal-38_relu.o`

3. **x86 版本**:在 `add_custom_command` 的 `COMMAND` 中添加:

   ```
   && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/38_relu.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-38_relu.o
   ```

   并在 `ar rcs` 命令中添加 `x86-38_relu.o`

4. 开头的ISA子模块库需要添加对应的**38_relu.c**文件。


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

cp -p "/home/MikeNotFound/code/buckyball/bb-tests/build/workloads/src/CTest/toy/ctest_relu_test_multicore-baremetal" "/home/MikeNotFound/code/buckyball/bb-tests/output/workloads/src/CTest/"

cp -p "/home/MikeNotFound/code/buckyball/bb-tests/build/workloads/src/CTest/toy/ctest_relu_test_singlecore-baremetal" "/home/MikeNotFound/code/buckyball/bb-tests/output/workloads/src/CTest/"

cp -p "/home/MikeNotFound/code/buckyball/bb-tests/build/workloads/src/CTest/toy/ctest_relu_test-linux" "/home/MikeNotFound/code/buckyball/bb-tests/output/workloads/src/CTest/"
```

若`ninja ctest_relu_test`执行后报错，这是软件编译没有通过，请检查**”三、 编写测试软件“**等相关文件。

### 步骤2: 生成Verilog

```
cd buckyball
bbdev verilator --verilog
```

若`bbdev verilator --verilog`执行后报错，这是硬件编译没有通过，请检查**“一、 编写Chisel硬件模块 二、编译适配准备”**相关文件。


### 步骤3: 运行仿真

```
bbdev verilator --run '--jobs 16 --binary ctest_relu_test_singlecore-baremetal --batch'
```

若`bbdev verilator --verilog`执行后报错，这是硬件系统有超时，卡死等问题，请检查**一、 编写Chisel硬件模块**相关文件。

### 步骤5：查看仿真文件

在`arch/waveform/仿真文件名(E.g.2025-10-08-00-03-ctest_vecunit_matmul_random1_singlecore-baremetal)`中，将`waveform.fst`文件用本地系统的`Filezilla`等软件下载到本地上，在本地的仿真波形查看器(E.g. GTKWave)进行波形查看。

注意，仿真文件所在的文件夹只能由`waveform.fst`一个文件，若存在`waveform.fst.hier`文件则代表仿真失败.

若波形没有满足理论情况，请在软件测试代码没有问题的情况下检查**一、 编写Chisel硬件模块**相关文件。

软件代码是否有问题可以参考其在`cpu`执行的结果，可以更改`relu_test.c`文件暂时完全移除硬件加速器调用，只测试 CPU 版本。

## 五、仿真波形
将`waveform.fst`导入到本地后，用[GTKWAVE](https://zhuanlan.zhihu.com/p/647533706)，在工程索引中寻找：
`TOP.TestHarness.chiptop0.system.tile_prci_domain.element_reset_domain_tile.buckyball.ballDomain.bbus.balls_4.reluUnit`该文件下的常量就是我们Relu.scala用到的所用硬件常量，双击便可查看波形！

> 不同例程的一些命名可能不会完全一样，但基本相差不大

## 六、性能测试

### 查询所用时钟周期数量-速度性能衡量参数
```Scala
cat /home/MikeNotFound/code/buckyball/arch/log/2025-10-24-16-59-ctest_relu_test_singlecore-baremetal/disasm.log | grep "PMC"
```
