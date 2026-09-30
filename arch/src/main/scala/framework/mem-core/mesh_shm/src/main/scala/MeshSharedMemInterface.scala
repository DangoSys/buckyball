package memcore.memory.mesh_shm

import chisel3._

class MeshClientRequest(p: MeshSharedMemParams) extends Bundle {
  val bank  = UInt(p.bankBits.W)
  val addr  = UInt(p.addressBits.W)
  val write = Bool()
  val data  = UInt(p.dataBits.W)
  val mask  = UInt(p.maskBits.W)
  val tag   = UInt(p.tagBits.W)
}

class MeshClientResponse(p: MeshSharedMemParams) extends Bundle {
  val data  = UInt(p.dataBits.W)
  val tag   = UInt(p.tagBits.W)
  val error = Bool()
}

class MeshPacket(p: MeshSharedMemParams) extends Bundle {
  val destRow   = UInt(p.rowBits.W)
  val destCol   = UInt(p.colBits.W)
  val sourceRow = UInt(p.rowBits.W)
  val sourceCol = UInt(p.colBits.W)
  val channel   = UInt(p.channelBits.W)
  val addr      = UInt(p.addressBits.W)
  val write     = Bool()
  val data      = UInt(p.dataBits.W)
  val mask      = UInt(p.maskBits.W)
  val tag       = UInt(p.tagBits.W)
  val error     = Bool()
}
