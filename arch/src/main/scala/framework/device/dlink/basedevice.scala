package framework.device.dlink

import framework.ip.bus.axi4.{Params => ModuleParam}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}

trait HasDLink { def dlink: DLinkIO }

@instantiable
abstract class BaseDevice(b: ModuleParam) extends Module with HasDLink {
  val linkParams = b

  @public
  val dlink = IO(new DLinkIO(b))
}
