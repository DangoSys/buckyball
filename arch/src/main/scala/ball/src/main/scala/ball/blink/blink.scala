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
package framework.ball.blink

import framework.generic.contract.blink.BallResponse

import chisel3._
import chisel3.util._
import framework.ball.mmio_bank.{MmioRead, MmioWriteReq}
import framework.generic.contract.configs.BallRouteParam
import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.memdomain.BankRead
import framework.generic.contract.memdomain.BankWrite
import framework.generic.contract.blink.BallCommand

class BallStatus extends Bundle {
  val idle    = Output(Bool())
  val running = Output(Bool())
}

/** The whole boundary of a Ball: its link parameters and its own route are all it is built from. */
class BlinkIO(link: BlinkParam, route: BallRouteParam) extends Bundle {
  val mmioRead  = Vec(route.mmioReadBW, Flipped(new MmioRead(link)))
  val mmioWrite = Vec(route.mmioWriteBW, Decoupled(new MmioWriteReq(link)))
  val status    = new BallStatus
  val cmdReq    = Flipped(Decoupled(new BallCommand(link)))
  val cmdResp   = Decoupled(new BallResponse(link))
  val bankRead  = Vec(route.inBW, Flipped(new BankRead(link)))
  val bankWrite = Vec(route.outBW, Flipped(new BankWrite(link)))
}
