# Tutorial for buckyball

> by - Bohan Wang
>
> This document will be gradually updated as the author continues to solve and summarize encountered issues.

This document explains the step-by-step process and problem-solving approaches for a complete `buckyball` development workflow. We use building a ball operator module for executing the `relu()` function as an example:

First, we need to complete the hardware code writing for this module, i.e., write hardware code in Scala's Chisel language and generate corresponding `verilog` code.

Second, we need to write test software to implement `relu()`, which can be a reference function that runs on `CPU` with software code and an experimental function that runs software code on the dedicated hardware written in step one. If the test results match, it's successful, or proceed to step three for testing.

Third, simulate at the hardware level, view waveform diagrams for debugging. Additionally, there are other details such as compiler documentation changes, instruction set updates, etc., which will be explained below.

When encountering issues during development, you can visit [DangoSys/buckyball | DeepWiki](https://deepwiki.com/DangoSys/buckyball) or [Project Overview - Buckyball Technical Documentation](https://dangosys.github.io/buckyball/index.html)

Chisel learning resources: [binder](https://mybinder.org/v2/gh/freechipsproject/chisel-bootcamp/master)

Before starting officially, let's initialize the environment:

```
cd /path/to/buckyball
source env.sh
// source ./env.sh if this gives an error
// All paths in this document are relative paths starting from ./buckyball
```

## I. Writing Chisel Hardware Module

Create a Chisel implementation of the `ReLU` accelerator in the `arch/src/main/scala/prototype/` directory. Referring to existing accelerator structures, it's recommended to create a new subdirectory under `prototype/`, for example `prototype/relu/Relu.scala`, and write the hardware code.

## II. Hardware Instruction Decoding

Next, decode hardware instructions. Support for ReLU instructions needs to be added on the **hardware side** so that the hardware decoder recognizes this instruction, and register the instruction set for this ball.

This work is mainly divided into the following five aspects:

- Instruction enumeration (DISA) defines func7 → instruction name (RELU)
- Decoder (DomainDecoder) defines func7 → decoding rules (read/write/address/iter) → BID (e.g., 4)
- Bus registration (busRegister) defines BID → actual Ball instance (ReluBall indexed at 4)
- Reservation station registration (rsRegister) is used for RS/issue descriptions, aligned with BID, facilitating system issue/completion management and debugging
If any link is missing or inconsistent, the ReLU instruction cannot be correctly recognized/routed/executed on actual hardware.
- Create a new Ball execution unit `class ReluUnit` to handle ReLU operations.

#### 1. Define RELU_BITPAT in DISA.scala

`arch/src/main/scala/examples/toy/balldomain/DISA.scala` defines the funct7 encoding (BitPat) for Ball instructions, such as TRANSPOSE, IM2COL, etc. It can be viewed as an "instruction set enumeration table" for decoder matching.

Add the bit pattern definition for the ReLU instruction in this file:

```
val RELU_BITPAT = BitPat("b0100110") // func7 = 38 = 0x23
```

#### 2. Add ReLU instruction to Ball domain decoder

`arch/src/main/scala/examples/toy/balldomain/DomainDecoder.scala` is the Ball domain decoder.
Its functions are as follows:
- Input: PostGDCmd from global decoding (already determined to be a Ball category command).
- Output: Structured BallDecodeCmd, including:
  - Whether to use op1/op2, whether to write back to scratchpad, whether operands come from scratchpad
  - Operand/writeback bank and address
  - Iteration count iter
  - Target Ball ID (BID)
  - Other dedicated fields special, etc.
- Internally maps different funct7 instructions to a set of boolean switches and field extraction rules through ListLookup(func7, ...).

Add the decoding entry for the ReLU instruction in the decoding list in this file. Referring to the implementation of other instructions (e.g., TRANSPOSE_FUNC7 = 38), you need:

```
// Add to BallDecodeFields ListLookup
RELU                 -> List(Y,N,Y,Y,N, rs1(spAddrLen-1,0), 0.U(spAddrLen.W), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen), 7.U, rs2(63,spAddrLen + 10), Y) // Fill in decoding fields according to specific ReLU instruction requirements, the number of list parameters must be consistent, you can refer to other instructions
```

#### 3. Add ReLuBall generator and register it

a. `arch/src/main/scala/examples/toy/balldomain/bbus/busRegister.scala` is the Ball bus registration table, using a `Seq(() => new SomeBall(...))` to register the actual Ball modules to be instantiated in the system.

Find and add the new ID for ReLuBall in this file.

```
class BBusModule(implicit b: CustomBuckyballConfig, p: Parameters)
    extends BBus(
      // Define Ball device generator to register
      Seq(
        () => new examples.toy.balldomain.vecball.VecBall(0),
        () => new examples.toy.balldomain.matrixball.MatrixBall(1),
        () => new examples.toy.balldomain.im2colball.Im2colBall(2),
        () => new examples.toy.balldomain.transposeball.TransposeBall(3),
        ...
        () =>new examples.toy.balldomain.reluball.ReluBall(7) // Ball ID 7 - newly added
      )
    ) {
  override lazy val desiredName = "BBusModule"
}
```

b. `arch/src/main/scala/examples/toy/balldomain/rs/rsRegister.scala` is the "Ball reservation station" registration table, using a list to register which Balls exist in the system (specifying ID and name by ballId). The reservation station (RS) is responsible for managing Ball issue, occupancy, completion and other metadata, usually also used for visualization/statistics, naming and logging.

Register ReluBall in this file:

```
class BallRSModule(implicit b: CustomBuckyballConfig, p: Parameters)
    extends BallReservationStation(
      // Define Ball device information to register
      Seq(
        BallRsRegist(ballId = 0, ballName = "VecBall"),
        BallRsRegist(ballId = 1, ballName = "MatrixBall"),
        BallRsRegist(ballId = 2, ballName = "Im2colBall"),
        BallRsRegist(ballId = 3, ballName = "TransposeBall"),
        ...
        BallRsRegist(ballId = 7, ballName = "ReluBall") // Ball ID 7 - newly added
      )
    ) {
  override lazy val desiredName = "BallRSModule"
}
```
#### 4. Write ReluBall interface file

Create a `reluball` folder in the `arch/src/main/scala/examples/toy/balldomain` directory, enter the folder and create `ReluBall.scala` to write the interface code.

## III. Writing Test Software and Compilation Settings

### 1. Create test file

Create `relu_test.c` under `bb-tests/workloads/src/CTest/toy/`, write test code. The core function in the code will execute `void bb_relu(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter);` Note the declaration and definition of this function below.

### 2. Modify CMakeLists.txt

Add test target in `bb-tests/workloads/src/CTest/toy/CMakeLists.txt`: CMakeLists.txt:120-127

```
add_cross_platform_test_target(ctest_relu_test relu_test.c)
```

And add to the main build target: CMakeLists.txt:137-162

```
add_custom_target(buckyball-CTest-build ALL DEPENDS
  # ... other tests ...
  ctest_relu_test
  COMMENT "Building all workloads for Buckyball"
  VERBATIM)
```

### 3. Need to add ReLU instruction API

#### a. isa.h

- Add declaration for `ReLU` instruction in `bb-tests/workloads/lib/bbhw/isa/isa.h`: `isa.h:33-43`

- Add to `InstructionType` enum:

```
RELU_FUNC7 = 38,  // 0x26 - ReLU function code (or other value you choose)
```

- Add to function declaration section: `isa.h:72-73`

```
void bb_relu(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter);
```

#### b. isa.c

- Add `38_relu.c` in `bb-tests/workloads/lib/bbhw/isa`, implement `void bb_relu(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter)` inside

- Add declaration in `bb-tests/workloads/lib/bbhw/isa/isa.c`: `isa.c:53-76`

```
case RELU_FUNC7:
	return &relu_config;
```

- In `isa.c:37-47`

```
extern const InstructionConfig relu_config;
```

### 4. Update CMakeLists.txt

Add compilation and linking of `38_relu.c` in all three compilation commands in `bb-tests/workloads/lib/bbhw/isa/CMakeLists.txt`:

1. **Linux version**: Add in `COMMAND` of `add_custom_command`:

   ```
   && riscv64-unknown-linux-gnu-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/38_relu.c -march=rv64gc -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o linux-38_relu.o
   ```

   And add `linux-38_relu.o` to the `ar rcs` command

2. **Baremetal version**: Add in `COMMAND` of `add_custom_command`:

   ```
   && riscv64-unknown-elf-gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/38_relu.c -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf -D__BAREMETAL__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o baremetal-38_relu.o
   ```

   And add `baremetal-38_relu.o` to the `ar rcs` command

3. **x86 version**: Add in `COMMAND` of `add_custom_command`:

   ```
   && gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/38_relu.c -fPIC -D__x86_64__ -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -o x86-38_relu.o
   ```

   And add `x86-38_relu.o` to the `ar rcs` command

4. The ISA submodule library at the beginning needs to add the corresponding **38_relu.c** file.


## IV. Test Operation Steps

### Step 1: Compile test program

```
cd bb-tests/build
rm -rf *
cmake -G Ninja ../
```

**Warning**: Before executing `rm -rf *`, make sure you are in the `bb-tests/build` directory, otherwise forcing deletion in the wrong folder will be catastrophic!

If a disaster occurs, you can pull the initial documents from GitHub again, but files updated on the server side cannot be recovered.

```
ninja ctest_relu_test // Software compilation
```

If `ninja ctest_relu_test` reports an error after execution, this means software compilation failed, please check **"III. Writing Test Software"** and related files.

```
bbdev workload --build
```

Compile/package the selected workload source code or configuration into artifacts (such as executable files, images, runtime scripts, input data packages, etc.) that can be used in the simulation or runtime environment for subsequent running on the Verilator/simulation platform or host side.

### Step 2: Generate Verilog

```
cd buckyball
bbdev verilator --verilog '--config sims.verilator.BuckyballToyVerilatorConfig'
```

If `bbdev verilator --verilog` reports an error after execution, this means hardware compilation failed, please check **"I. Writing Chisel Hardware Module II. Compilation Adaptation Preparation"** related files.


### Step 3: Run simulation

```
bbdev verilator --run '--jobs 16 --binary ctest_relu_test_singlecore-baremetal --batch'
```

If `bbdev verilator --verilog` reports an error after execution, this means the hardware system has timeout, deadlock and other issues, please check **I. Writing Chisel Hardware Module** related files.

### Step 5: View simulation files

In `arch/waveform/SimulationFileName(E.g.2025-10-08-00-03-ctest_vecunit_matmul_random1_singlecore-baremetal)`, download the `waveform.fst` file to your local system using software like `Filezilla`, and view the waveform using a local simulation waveform viewer (E.g. GTKWave).

Note that the simulation file folder should only contain the `waveform.fst` file. If a `waveform.fst.hier` file exists, it means the simulation failed.

If the waveform does not meet theoretical conditions, check **I. Writing Chisel Hardware Module** related files when the software test code is correct.

To check if the software code has problems, you can refer to its execution results on `CPU`. You can temporarily completely remove hardware accelerator calls from the `relu_test.c` file and only test the CPU version.

## V. Simulation Waveform
After importing `waveform.fst` locally, use [GTKWAVE](https://zhuanlan.zhihu.com/p/647533706) to find in the project index:
`TOP.TestHarness.chiptop0.system.tile_prci_domain.element_reset_domain_tile.buckyball.ballDomain.bbus.balls_4.reluUnit` The constants under this file are all hardware constants used by Relu.scala, double-click to view the waveform!

> Some naming for different routines may not be exactly the same, but they are basically similar

## VI. Performance Testing

### Query number of clock cycles used - speed performance metric
```Scala
cat /home/MikeNotFound/code/buckyball/arch/log/2025-10-24-16-59-ctest_relu_test_singlecore-baremetal/disasm.log | grep "PMC"
```

### DC Test - Check Timing, Frequency, Area, and Related Parameters

* **Preparation**

1. In the `/home/<server_name>/bash.sh` file, add the required environment variables at the end:

   ```bash
   export SNPSLMD_LICENSE_FILE=27000@amax
   export PATH="$PATH:/opt/riscv/bin"
   export VCS_HOME="/data0/tools/Synopsys/vcs/vcs/W-2024.09-SP1"
   export PATH="$PATH:$VCS_HOME/bin"
   export VERDI_HOME="/data0/tools/Synopsys/verdi/verdi/W-2024.09-SP1"
   export PATH="$PATH:$VERDI_HOME/bin"
   export SCL_HOME="/data0/tools/Synopsys/scl/scl/2024.06"
   export PATH="$PATH:$SCL_HOME/linux64/bin"
   export DC_HOME="/data0/tools/Synopsys/dc/syn/W-2024.09-SP1"
   export PATH="$PATH:$DC_HOME/bin"
   export PT_HOME="/data0/tools/Synopsys/ptpx/prime/W-2024.09-SP1/"
   export PATH="$PATH:$PT_HOME/bin"

   export LM_LICENSE_FILE=/data0/tools/Synopsys/lic/Synopsys.dat

   alias vcs="vcs -full64"
   alias lmli="lmgrd -c /data0/tools/Synopsys/lic/Synopsys.dat"
   ```

2. In the `/home/<server_name>/code/buckyball/evals/run-dc.sh` file, remove the `-retime` option around line 126.

---

* **Formal Test**

1. Go back to the `buckyball` directory and run the command

   ```bash
   bbdev verilator --verilog "--balltype ReluBall --output_dir ReluBall_1"
   ```

   This will generate a Verilog folder for the specified ball under the `arch` directory.

2. Grant execution permission to the script:

   ```bash
   chmod 777 evals/run-dc.sh
   ```

3. Run the DC command:

   ```bash
   ./evals/run-dc.sh --srcdir arch/ReluBall_1 --top ReluBall
   ```

   This means performing the DC test on the top-level file `ReluBall.sv` located in the `arch/ReluBall_1` folder.

4. You can find the test results in

   ```
   /home/<server_name>/buckyball/bb-tests/output/dc/reports
   ```
