//===- blink.scala --------- Ball Link Interface --------------------------===//
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
package framework.acc.balldomain.ball.blink

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import framework.acc.memdomain.{BankRead, BankWrite}
import framework.acc.balldomain.mmio_bank.{MmioRead, MmioWriteReq}

/**
 * One Ball operation. Banks are already renamed; each operand covers `col` groups starting at `group`.
 */
class BallCommand(fp: FrontendParam, mp: MemDomainParam) extends Bundle {
  val rob_id       = UInt(log2Up(fp.rob_entries).W)
  val funct        = UInt(7.W)
  val rs2          = UInt(fp.commandDataBits.W)
  val iter         = UInt(fp.iter_len.W)
  val op1_bank     = UInt(fp.bank_id_len.W)
  val op2_bank     = UInt(fp.bank_id_len.W)
  val wr_bank      = UInt(fp.bank_id_len.W)
  val op1_group    = UInt(mp.groupIdWidth.W)
  val op2_group    = UInt(mp.groupIdWidth.W)
  val wr_group     = UInt(mp.groupIdWidth.W)
  val op1_col      = UInt(mp.groupCountWidth.W)
  val op2_col      = UInt(mp.groupCountWidth.W)
  val wr_col       = UInt(mp.groupCountWidth.W)
  // Renamed bank holding the previous destination value for accumulation.
  val wr_read_bank = UInt(fp.bank_id_len.W)
  // Destination bank before renaming; identifies chained writes to one architectural bank.
  val wr_arch_bank = UInt(fp.bank_id_len.W)
}

class BallResponse(fp: FrontendParam) extends Bundle {
  val rob_id = UInt(log2Up(fp.rob_entries).W)
}

class BallStatus extends Bundle {
  val idle    = Output(Bool())
  val running = Output(Bool())
}

class BlinkIO(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Bundle {
  val ball = bp.balls.find(_.id == id).get

  val mmioRead  = Vec(ball.mmioReadBW, Flipped(new MmioRead(mp)))
  val mmioWrite = Vec(ball.mmioWriteBW, Decoupled(new MmioWriteReq(mp)))
  val status    = new BallStatus
  val cmdReq    = Flipped(Decoupled(new BallCommand(fp, mp)))
  val cmdResp   = Decoupled(new BallResponse(fp))
  val bankRead  = Vec(ball.inBW, Flipped(new BankRead(fp, mp)))
  val bankWrite = Vec(ball.outBW, Flipped(new BankWrite(fp, mp)))
}
