package framework.memdomain.backend.shared

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig
import framework.memdomain.backend.banks.{SramReadResp, SramWriteResp}

object SharedMemLayout {

  def totalBank(b: GlobalConfig): Int = {
    require(b.memDomain.sharedEnable, "shared memory is disabled")
    require(b.memDomain.sharedBankNum > 0, "sharedBankNum must be > 0")
    b.memDomain.sharedBankNum
  }

  def channelPerHart(b: GlobalConfig): Int = {
    if (!b.memDomain.sharedEnable) {
      return 0
    }
    require(b.memDomain.nCores > 0, s"nCores(${b.memDomain.nCores}) must be > 0")
    require(
      b.memDomain.sharedInputChannels > 0,
      s"sharedInputChannels(${b.memDomain.sharedInputChannels}) must be > 0"
    )
    require(
      b.memDomain.sharedInputChannels % b.memDomain.nCores == 0,
      s"sharedInputChannels(${b.memDomain.sharedInputChannels}) must be divisible by nCores(${b.memDomain.nCores})"
    )
    if (b.memDomain.sharedInputChannels > 32) {
      require(
        b.memDomain.sharedInputChannels == b.memDomain.nCores,
        s"sharedInputChannels(${b.memDomain.sharedInputChannels}) must equal nCores(${b.memDomain.nCores}) when > 32"
      )
    }
    val ch = b.memDomain.sharedInputChannels / b.memDomain.nCores
    require(ch > 0, s"channelPerHart($ch) must be > 0")
    ch
  }

  def totalChannel(b: GlobalConfig): Int = if (b.memDomain.sharedEnable) b.memDomain.sharedInputChannels else 0
}

class SharedMemReadReq(val b: GlobalConfig) extends Bundle {
  val hartid   = UInt(b.tile.xLen.W)
  val pbank_id = UInt(log2Ceil(SharedMemLayout.totalBank(b)).W)
  val group_id = UInt(b.memDomain.groupIdWidth.W)
  val addr     = UInt(log2Ceil(b.memDomain.bankEntries).W)
}

class SharedMemWriteReq(val b: GlobalConfig) extends Bundle {
  val hartid   = UInt(b.tile.xLen.W)
  val pbank_id = UInt(log2Ceil(SharedMemLayout.totalBank(b)).W)
  val group_id = UInt(b.memDomain.groupIdWidth.W)
  val addr     = UInt(log2Ceil(b.memDomain.bankEntries).W)
  val mask     = Vec(b.memDomain.bankMaskLen, Bool())
  val data     = UInt(b.memDomain.bankWidth.W)
}

class SharedMemReadIO(val b: GlobalConfig) extends Bundle {
  val req  = Flipped(Decoupled(new SharedMemReadReq(b)))
  val resp = Decoupled(new SramReadResp(b))
}

class SharedMemWriteIO(val b: GlobalConfig) extends Bundle {
  val req  = Flipped(Decoupled(new SharedMemWriteReq(b)))
  val resp = Decoupled(new SramWriteResp(b))
}

@instantiable
class SharedMem(val b: GlobalConfig) extends Module {
  private val maskLen        = b.memDomain.bankMaskLen
  private val maskElem       = UInt((b.memDomain.bankWidth / maskLen).W)
  private val totalBanks     = SharedMemLayout.totalBank(b)
  private val minSharedLines = totalBanks * b.memDomain.bankEntries

  require(
    b.memDomain.sharedEntries >= minSharedLines,
    s"sharedEntries=${b.memDomain.sharedEntries} is too small, minimum=$minSharedLines"
  )

  @public
  val io = IO(new Bundle {
    val read  = new SharedMemReadIO(b)
    val write = new SharedMemWriteIO(b)
  })

  val mem = SyncReadMem(b.memDomain.sharedEntries, Vec(maskLen, maskElem))

  // Shared memory address mapping (group_id intentionally ignored):
  // shared_addr = pbank_id * bankEntries + local_addr.
  private def toSharedAddr(
    pbank_id:   UInt,
    _group_id:  UInt,
    local_addr: UInt
  ): UInt = {
    val pbankPart = pbank_id * b.memDomain.bankEntries.U
    pbankPart + local_addr
  }

  io.read.req.ready  := !io.write.req.valid
  io.write.req.ready := !io.read.req.valid

  val readReqFire = io.read.req.fire
  val readAddr    = toSharedAddr(io.read.req.bits.pbank_id, io.read.req.bits.group_id, io.read.req.bits.addr)

  when(readReqFire) {
    assert(io.read.req.bits.pbank_id < totalBanks.U, "SharedMem: pbank_id out of range")
    assert(io.read.req.bits.addr < b.memDomain.bankEntries.U, "SharedMem: local addr out of range")
  }

  val readData = mem.read(readAddr, readReqFire)

  io.read.resp.valid     := RegNext(readReqFire, init = false.B)
  io.read.resp.bits.data := readData.asUInt

  val writeReqFire = io.write.req.fire
  val writeAddr    = toSharedAddr(io.write.req.bits.pbank_id, io.write.req.bits.group_id, io.write.req.bits.addr)

  when(writeReqFire) {
    assert(io.write.req.bits.pbank_id < totalBanks.U, "SharedMem: pbank_id out of range")
    assert(io.write.req.bits.addr < b.memDomain.bankEntries.U, "SharedMem: local addr out of range")
    mem.write(
      writeAddr,
      io.write.req.bits.data.asTypeOf(Vec(maskLen, maskElem)),
      io.write.req.bits.mask
    )
  }

  io.write.resp.valid   := RegNext(writeReqFire, init = false.B)
  io.write.resp.bits.ok := RegNext(writeReqFire, init = false.B)
}
