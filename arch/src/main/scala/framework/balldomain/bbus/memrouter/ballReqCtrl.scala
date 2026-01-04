package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.blink.BankRead

@instantiable
class BallReqCtrl(val b: GlobalConfig) extends Module {
  @public
  val io = IO(new Bundle {})
}
