package framework.thread.tlink

import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
trait HasTLink {
  @public
  val tlink: TLinkIO
}
