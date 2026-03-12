package framework.balldomain.prototype.trace

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink}
import framework.top.GlobalConfig

/**
 * TraceBall — Debug trace Ball providing cycle counters and SRAM backdoor.
 *
 * Uses two funct7 encodings:
 *   - bdb_counter  (funct7=48): cycle counter management
 *   - bdb_backdoor (funct7=49): SRAM backdoor read/write via DPI-C
 */
@instantiable
class TraceBall(val b: GlobalConfig) extends Module with HasBlink {

  val ballCommonConfig = b.ballDomain.ballIdMappings
    .find(_.ballName == "TraceBall")
    .getOrElse(throw new IllegalArgumentException("TraceBall not found in config"))

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink: BlinkIO = io

  val traceUnit: Instance[Trace] = Instantiate(new Trace(b))

  traceUnit.io.cmdReq <> io.cmdReq
  traceUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    traceUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    traceUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  io.status <> traceUnit.io.status
}
