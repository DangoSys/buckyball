package framework.hart.hlink

import chisel3._
import chisel3.util._

/** Payload beats of a management request; its descriptor defines length and direction. */
class ServiceData extends Bundle {
  val send    = Decoupled(UInt(128.W))
  val receive = Flipped(Decoupled(UInt(128.W)))
}
