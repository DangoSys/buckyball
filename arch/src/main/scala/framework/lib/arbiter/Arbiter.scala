// SPDX-License-Identifier: Apache-2.0
// Arbitration follows chisel3.util.Arbiter and RRArbiter.
package framework.lib.arbiter

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class Arbiter[T <: Data](gen: T, n: Int) extends Module {
  @public
  val io = IO(new ArbiterIO(gen, n))

  io.chosen   := (n - 1).U
  io.out.bits := io.in(n - 1).bits
  for (i             <- n - 2 to 0 by -1) {
    when(io.in(i).valid) {
      io.chosen   := i.U
      io.out.bits := io.in(i).bits
    }
  }
  val preceding = io.in.map(_.valid).scanLeft(false.B)(_ || _).dropRight(1)
  for ((input, busy) <- io.in.zip(preceding)) input.ready := !busy && io.out.ready
  io.out.valid := io.in.map(_.valid).reduce(_ || _)
}

@instantiable
class RRArbiter[T <: Data](gen: T, n: Int) extends Module {
  @public
  val io = IO(new ArbiterIO(gen, n))

  val lastGrant = RegInit(0.U(math.max(1, log2Ceil(n)).W))
  val mask      = (0 until n).map(_.U > lastGrant)
  val masked    = io.in.zip(mask).map { case (input, eligible) => input.valid && eligible }
  val requests  = masked ++ io.in.map(_.valid)
  val grant     = requests.scanLeft(false.B)(_ || _).dropRight(1).map(!_)

  io.chosen := (n - 1).U
  for (i <- n - 2 to 0 by -1) when(io.in(i).valid)(io.chosen := i.U)
  for (i <- n - 1 to 1 by -1) when(masked(i))(io.chosen := i.U)
  for (i <- 0 until n) io.in(i).ready := (grant(i) && mask(i) || grant(i + n)) && io.out.ready
  io.out.valid                := io.in(io.chosen).valid
  io.out.bits                 := io.in(io.chosen).bits
  when(io.out.fire)(lastGrant := io.chosen)
}
