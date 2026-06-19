//===- BallCyclePMC.scala ------ Ball Performance Counter -----------------===//
//
// Copyright 2026 The Buckyball Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This DPI-C module is used to collect the performance counter data from each
// ball mounted on the bbus. Each ball has its own PMCModule.
// PMCModule just triggers at two points:
//
// 1. When the ball issues a command.
// 2. When the ball completes a command.
//
//===----------------------------------------------------------------------===//

package framework.balldomain.bbus.pmc

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.rs.BallRsIssue
import framework.balldomain.rs.BallRsComplete
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class BallCyclePMC(val b: GlobalConfig) extends Module {
  val numBalls = b.ballDomain.ballNum

  @public
  val io = IO(new Bundle {
    val cmdReq_i  = Input(Vec(numBalls, Valid(new BallRsIssue(b))))
    val cmdResp_o = Input(Vec(numBalls, Valid(new BallRsComplete(b))))
  })

// A global cycle counter after reset.
  val cycleCounter = RegInit(0.U(64.W))
  val startTime    = RegInit(VecInit(Seq.fill(b.frontend.rob_entries)(0.U(64.W))))
  val pmcTraces    = Seq.fill(numBalls)(Module(new PMCTraceDPI))

  cycleCounter := cycleCounter + 1.U

//===---------- Initialize the PMC traces ----------===//
  for (pt <- pmcTraces) {
    pt.io.ball_id := 0.U
    pt.io.rob_id  := 0.U
    pt.io.elapsed := 0.U
    pt.io.enable  := false.B
  }

//===---------- Collect the performance counter data ----------===//
// Trigger at the start of the command.
  for (i <- 0 until numBalls) {
    when(io.cmdReq_i(i).valid) {
      startTime(io.cmdReq_i(i).bits.rob_id) := cycleCounter
    }
  }

// Trigger at the end of the command.
  for (i <- 0 until numBalls) {
    when(io.cmdResp_o(i).valid) {
      val robId   = io.cmdResp_o(i).bits.rob_id
      val elapsed = cycleCounter - startTime(robId)
      pmcTraces(i).io.ball_id := i.U
      pmcTraces(i).io.rob_id  := robId
      pmcTraces(i).io.elapsed := elapsed
      pmcTraces(i).io.enable  := true.B
    }
  }
}
