package memcore.bus.axi

import chisel3._

/** Physical AXI4-Stream signals, directed from the master's perspective. */
class Port(
  dataBits: Int,
  idBits:   Int = 0,
  destBits: Int = 0,
  userBits: Int = 0)
    extends Bundle {
  require(dataBits > 0 && dataBits % 8 == 0)
  val tvalid = Output(Bool())
  val tready = Input(Bool())
  val tdata  = Output(UInt(dataBits.W))
  val tkeep  = Output(UInt((dataBits / 8).W))
  val tlast  = Output(Bool())
  val tid    = Output(UInt(idBits.W))
  val tdest  = Output(UInt(destBits.W))
  val tuser  = Output(UInt(userBits.W))
}

/**
 * AXI4-Stream payload signals.
 *
 * A `Decoupled[Beat]` is the full AXI-S channel: `valid` and `ready`
 * are TVALID and TREADY. This bundle exposes the remaining standard signal
 * names directly in generated RTL.
 */
class Beat(
  dataBits: Int,
  idBits:   Int = 0,
  destBits: Int = 0,
  userBits: Int = 0)
    extends Bundle {
  require(dataBits > 0 && dataBits % 8 == 0)
  val tdata = UInt(dataBits.W)
  val tkeep = UInt((dataBits / 8).W)
  val tlast = Bool()
  val tid   = UInt(idBits.W)
  val tdest = UInt(destBits.W)
  val tuser = UInt(userBits.W)

  // Internal aliases preserve concise datapath code while retaining AXI-S names.
  def data: UInt = tdata
  def keep: UInt = tkeep
  def last: Bool = tlast
  def id:   UInt = tid
  def dest: UInt = tdest
  def user: UInt = tuser
}
