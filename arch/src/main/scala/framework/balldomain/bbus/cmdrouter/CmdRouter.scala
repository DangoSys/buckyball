//===- CmdRouter.scala ------ Command Router of Ball Domain ---------------===//
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
// Request channel (req): N inputs (cmdReq_i) -> 1 output (cmdReq_o)
//
// Response channel (resp): N inputs (cmdResp_i) -> N outputs (cmdResp_o), with
//                          direct point-to-point connections
//
//===----------------------------------------------------------------------===//
package framework.balldomain.bbus.cmdrouter

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class CmdRouter(val b: GlobalConfig) extends Module {
  val numBalls = b.ballDomain.ballNum

  @public
  val io = IO(new Bundle {
    val cmdReq_i  = Vec(numBalls, Flipped(Decoupled(new BallRsIssue(b))))
    val cmdResp_i = Vec(numBalls, Flipped(Decoupled(new BallRsComplete(b))))
    val cmdReq_o  = Decoupled(new BallRsIssue(b))
    val cmdResp_o = Vec(numBalls, Decoupled(new BallRsComplete(b)))
    val ballIdle  = Input(Vec(numBalls, Bool()))
  })

  val arbiter   = Module(new RRArbiter(new BallRsIssue(b), numBalls))
  val ballIdleR = RegNext(io.ballIdle, VecInit(Seq.fill(numBalls)(false.B)))

  for (i <- 0 until numBalls) {
    arbiter.io.in(i).valid := io.cmdReq_i(i).valid && ballIdleR(i)
    arbiter.io.in(i).bits  := io.cmdReq_i(i).bits
    io.cmdReq_i(i).ready   := arbiter.io.in(i).ready && ballIdleR(i)
  }

  io.cmdReq_o <> arbiter.io.out

  for (i <- 0 until numBalls) {
    io.cmdResp_o(i) <> io.cmdResp_i(i)
  }
}
