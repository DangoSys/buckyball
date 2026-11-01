package framework.device.shm

import chisel3._
import chisel3.util._
import framework.ip.bus.axi4

/** Device-internal connection between communication nodes. */
class Link(axi: axi4.Params) extends Bundle {
  val tx             = new axi4.Port(axi)
  val rx             = Flipped(new axi4.Port(axi))
  val routeTarget    = Output(UInt(64.W))
  val routeAvailable = Input(Bool())
}

class StorageRequest(dataBits: Int) extends Bundle {
  val address = UInt(64.W)
  val write   = Bool()
  val data    = UInt(dataBits.W)
  val mask    = UInt((dataBits / 8).W)
}

class StorageResponse(dataBits: Int) extends Bundle {
  val data  = UInt(dataBits.W)
  val error = Bool()
}

/** Device storage acknowledges writes after the SRAM operation completes. */
class StorageAccess(dataBits: Int) extends Bundle {
  val request  = Decoupled(new StorageRequest(dataBits))
  val response = Flipped(Decoupled(new StorageResponse(dataBits)))
}
