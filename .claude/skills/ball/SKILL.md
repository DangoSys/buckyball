---
name: ball
description: 创建一个名为 $ARGUMENTS 的新 Buckyball Ball 算子，完成从实现到验证的全流程。当用户要求创建新 Ball、新算子、新加速单元，或说"加一个 XX Ball"、"实现 XX 操作"时，使用此 skill。即使用户没有明确说"Ball"，只要意图是在 Buckyball 框架中增加一个新的计算算子，都应触发。
---

**重要：编译、仿真等操作必须通过 MCP 工具（validate、bbdev_workload_build、bbdev_verilator_run 等）调用，禁止直接使用 bbdev CLI 或 nix develop 命令。**

## 阶段 1 — 需求收集

1. 读取当前注册状态，确定新 Ball 的 ballId 和 funct7：
   - `arch/src/main/scala/framework/balldomain/configs/default.json`
   - `arch/src/main/scala/examples/toy/balldomain/DISA.scala`
2. 检查是否已部分存在（增量模式）：
   - 搜索 `arch/src/main/scala/framework/balldomain/prototype/` 下是否有同名目录
   - 搜索 `bb-tests/workloads/lib/bbhw/isa/` 下是否有对应的 ISA 宏文件
   - 搜索 `bb-tests/workloads/src/CTest/toy/` 下是否有对应的 CTest
   - 如果部分文件已存在，只补齐缺失部分
3. 向用户确认以下信息：
   - Ball 的计算语义（做什么运算）
   - inBW / outBW（读/写 bank 端口数量）
   - 是否需要第二个操作数（op2）
   - iter（迭代次数）的含义

## 阶段 2 — 实现 Ball

1. 读取参考代码，理解 Blink 协议和现有 Ball 的写法：
   - 简单参考：`arch/src/main/scala/framework/balldomain/prototype/relu/ReluBall.scala` 和 `Relu.scala`
   - 复杂参考：`arch/src/main/scala/framework/balldomain/prototype/systolicarray/`
   - Blink 协议：`arch/src/main/scala/framework/balldomain/blink/blink.scala`、`bank.scala`、`status.scala`
   - SRAM 接口：`arch/src/main/scala/framework/memdomain/backend/banks/SramIO.scala`
2. 在 `arch/src/main/scala/framework/balldomain/prototype/<name>/` 下创建文件，使用 `references/` 中的模板作为起点。

### Ball wrapper 模板（`<Name>Ball.scala`）

```scala
package framework.balldomain.prototype.<name>

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink}
import framework.top.GlobalConfig

@instantiable
class <Name>Ball(val b: GlobalConfig) extends Module with HasBlink {
  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "<Name>Ball")
    .getOrElse(throw new IllegalArgumentException("<Name>Ball not found in config"))
  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))
  def blink: BlinkIO = io

  val core: Instance[<Name>] = Instantiate(new <Name>(b))

  core.io.cmdReq  <> io.cmdReq
  core.io.cmdResp <> io.cmdResp
  for (i <- 0 until inBW)  { core.io.bankRead(i)  <> io.bankRead(i)  }
  for (i <- 0 until outBW) { core.io.bankWrite(i) <> io.bankWrite(i) }
  io.status <> core.io.status
}
```

### Core 模板（`<Name>.scala`）

```scala
package framework.balldomain.prototype.<name>

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig

@instantiable
class <Name>(val b: GlobalConfig) extends Module {
  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "<Name>Ball")
    .getOrElse(throw new IllegalArgumentException("<Name>Ball not found in config"))
  val inBW  = ballMapping.inBW
  val outBW = ballMapping.outBW

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  // Latch rob_id on cmdReq.fire
  val rob_id_reg = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  when(io.cmdReq.fire) { rob_id_reg := io.cmdReq.bits.rob_id }

  // Propagate rob_id to bank metadata
  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id  := rob_id_reg
    io.bankRead(i).ball_id := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id  := rob_id_reg
    io.bankWrite(i).ball_id := 0.U
  }

  // FSM: idle -> read -> compute -> write -> complete -> idle
  val idle :: sRead :: sCompute :: sWrite :: complete :: Nil = Enum(5)
  val state = RegInit(idle)

  // Default port assignments (override in FSM states)
  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
    io.bankRead(i).bank_id          := 0.U
    io.bankRead(i).group_id         := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := false.B
    io.bankWrite(i).bank_id           := 0.U
    io.bankWrite(i).group_id          := 0.U
  }

  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := rob_id_reg

  // Latch command fields
  val rbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iter_reg  = RegInit(0.U(b.frontend.iter_len.W))

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        rbank_reg := io.cmdReq.bits.cmd.op1_bank
        wbank_reg := io.cmdReq.bits.cmd.wr_bank
        iter_reg  := io.cmdReq.bits.cmd.iter
        state     := sRead
      }
    }
    is(sRead) {
      // TODO: implement read logic
      // Key: SRAM read latency = 1 cycle (resp.valid on next cycle after req.fire)
    }
    is(sCompute) {
      // TODO: implement compute logic
    }
    is(sWrite) {
      // TODO: implement write logic
    }
    is(complete) {
      io.cmdResp.valid       := true.B
      io.cmdResp.bits.rob_id := rob_id_reg
      when(io.cmdResp.fire) { state := idle }
    }
  }

  io.status.idle    := (state === idle)
  io.status.running := (state =/= idle) && (state =/= complete)
}
```

### Param 模板（`configs/<Name>BallParam.scala`）

```scala
package framework.balldomain.prototype.<name>.configs

import upickle.default._

case class <Name>BallParam(
  // TODO: add Ball-specific parameters
)

object <Name>BallParam {
  implicit val rw: ReadWriter[<Name>BallParam] = macroRW

  def apply(): <Name>BallParam = {
    val jsonStr = scala.io.Source
      .fromFile("src/main/scala/framework/balldomain/prototype/<name>/configs/default.json")
      .mkString
    read[<Name>BallParam](jsonStr)
  }
}
```

关键约束：
- SRAM 读延迟 = 1 cycle（req.fire 后下一周期 resp.valid）
- cmdReq.fire 时 latch 命令字段到寄存器
- FSM 基本模式：idle → 读数据 → 计算 → 写数据 → complete → idle
- status.idle 和 status.running 必须映射 FSM 状态

## 阶段 3 — 注册

按顺序更新以下四个文件：
1. `arch/src/main/scala/framework/balldomain/configs/default.json` — 追加 ballIdMappings 条目，更新 ballNum
2. `arch/src/main/scala/examples/toy/balldomain/bbus/busRegister.scala` — 添加 import 和 match case
3. `arch/src/main/scala/examples/toy/balldomain/DISA.scala` — 添加 `val XXX = BitPat("bxxxxxxx")`
4. `arch/src/main/scala/examples/toy/balldomain/DomainDecoder.scala` — 添加 ListLookup 解码行，BID = ballId.U

## 阶段 4 — ISA C 宏

在 `bb-tests/workloads/lib/bbhw/isa/` 下创建 `<funct7十进制>_<name>.c`：

```c
#ifndef _BB_<NAME>_H_
#define _BB_<NAME>_H_

#include "isa.h"

#define BB_<NAME>_FUNC7 <funct7_decimal>

#define bb_<name>(bank_id, wr_bank_id, iter)                                     \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) |          \
                             BB_RD0 | BB_WR | BB_ITER(iter)),                    \
                            0, BB_<NAME>_FUNC7)

#endif // _BB_<NAME>_H_
```

在 `bb-tests/workloads/lib/bbhw/isa/isa.h` 中 `#include` 新文件。

## 阶段 5 — CTest

在 `bb-tests/workloads/src/CTest/toy/` 下创建 `<name>_test.c`：

```c
#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 16

// Fixed input and expected output matrices
static elem_t input_matrix[DIM * DIM] __attribute__((aligned(64))) = { /* ... */ };
static elem_t expected_matrix[DIM * DIM] __attribute__((aligned(64))) = { /* ... */ };
static elem_t output_matrix[DIM * DIM] __attribute__((aligned(64)));

void hw_<name>(const char *test_name, elem_t *a, elem_t *b, int size) {
  uint32_t op1_bank_id = 0;
  uint32_t wr_bank_id = 1;

  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(wr_bank_id, 1, 1);

  bb_mvin((uintptr_t)a, op1_bank_id, size, 1);
  bb_<name>(op1_bank_id, wr_bank_id, size);
  bb_mvout((uintptr_t)b, wr_bank_id, size, 1);
  bb_fence();
}

int main() {
  clear_i8_matrix(output_matrix, DIM, DIM);
  hw_<name>("<Name>", input_matrix, output_matrix, DIM);

  if (compare_i8_matrices(output_matrix, expected_matrix, DIM, DIM)) {
    printf("<Name> test PASSED\n");
    return 0;
  } else {
    printf("<Name> test FAILED\n");
    return 1;
  }
}
```

在 `bb-tests/workloads/src/CTest/toy/CMakeLists.txt` 中用 `add_cross_platform_test_target` 注册。
在 `bb-tests/sardine/tests/test_ctest.py` 的 `ctest_workloads` 列表中追加对应条目。

## 阶段 6 — 校验 + 编译 + 仿真

1. 调用 MCP 工具 `validate` 做静态校验，确认 6 项不变量全部通过
2. 调用 MCP 工具 `bbdev_workload_build` 编译 CTest
3. 调用 MCP 工具 `bbdev_verilator_run` 跑仿真，指定新 Ball 的 CTest binary
4. 解析仿真结果：
   - PASSED → 完成
   - FAILED → 使用 `/debug` skill 进入调试流程
