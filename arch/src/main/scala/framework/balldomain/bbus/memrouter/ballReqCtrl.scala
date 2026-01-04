package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.blink.{MultiBankRead, MultiBankWrite}

@instantiable
class BallReqCtrl(val b: GlobalConfig) extends Module {
  val numBalls = b.ballDomain.ballNum

  @public
  val io = IO(new Bundle {
    val readReq_i  = Vec(numBalls, Flipped(new MultiBankRead(b, numBalls)))
    val writeReq_i = Vec(numBalls, Flipped(new MultiBankWrite(b, numBalls)))

    val readReq_o  = Vec(numBalls, new MultiBankRead(b, numBalls))
    val writeReq_o = Vec(numBalls, new MultiBankWrite(b, numBalls))
  })

}
