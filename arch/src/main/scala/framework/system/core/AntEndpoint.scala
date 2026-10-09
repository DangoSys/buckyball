package framework.system.core

import framework.system.GlobalConfigOps._
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.system.core.clink.{CLinkIO, HasCLink}
import framework.system.core.rocket.CpuParams

/** Typed Ant/NPU boundary; implementations own Ant and TLS, while the tile owns TSS. */
@instantiable
abstract class AntEndpoint(val b: framework.top.GlobalConfig) extends Module with HasCLink {
  val p         = b.antParams
  val local     = p.local
  val tracking  = p.tracking
  val bus       = p.bus
  val signature = p.signature
  implicit val cpuParams: CpuParams = p.cpu
  @public val clink = IO(new CLinkIO(b, local, tracking, bus))
}
