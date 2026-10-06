package memcore.memory.mesh_shm

import chisel3._

class MeshClientRequest(p: MeshSharedMemParams)
    extends memcore.bus.axi.Beat(p.dataBits, p.tagBits, p.bankBits, p.addressBits + 1) {
  def bank  = tdest
  def addr  = tuser(p.addressBits, 1)
  def write = tuser(0)
  def mask  = tkeep
  def tag   = tid
}

class MeshClientResponse(p: MeshSharedMemParams) extends memcore.bus.axi.Beat(p.dataBits, p.tagBits, 0, 2) {
  def tag   = tid
  def error = tuser(0)
  def write = tuser(1)
}

/** One row transaction: AXI-S payload plus routed event metadata in TUSER. */
class MeshPacket(p: MeshSharedMemParams)
    extends memcore.bus.axi.Beat(p.dataBits, p.tagBits, p.rowBits + p.colBits, p.packetUserBits) {
  def privateRequest = tuser(p.eventBits)
  def localBank      = tuser(p.eventBits + p.localBankBits, p.eventBits + 1)
  def core           = tuser(p.packetUserBits - 1, p.eventBits + p.localBankBits + 1)
  def destRow        = tdest(p.rowBits + p.colBits - 1, p.colBits)
  def destCol        = tdest(p.colBits - 1, 0)
  def error          = tuser(0)
  def write          = tuser(1)
  def addr           = tuser(p.addressBits + 1, 2)
  def channel        = tuser(p.addressBits + p.channelBits + 1, p.addressBits + 2)
  def sourceCol      = tuser(p.addressBits + p.channelBits + p.colBits + 1, p.addressBits + p.channelBits + 2)
  def sourceRow      =
    tuser(p.addressBits + p.channelBits + p.colBits + p.rowBits + 1, p.addressBits + p.channelBits + p.colBits + 2)
  def mask           = tkeep
  def tag            = tid
}
