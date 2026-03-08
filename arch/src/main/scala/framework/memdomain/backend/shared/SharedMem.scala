package framework.memdomain.backend.shared

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig
import framework.memdomain.backend.banks.{SramReadResp, SramWriteResp}

class SharedMemReadReq(val b: GlobalConfig) extends Bundle {
  val hartid   = UInt(b.core.xLen.W)
  val vbank_id = UInt(log2Up(b.memDomain.bankNum).W)
  val group_id = UInt(3.W)
  val addr     = UInt(log2Ceil(b.memDomain.bankEntries).W)
}

class SharedMemWriteReq(val b: GlobalConfig) extends Bundle {
  val hartid   = UInt(b.core.xLen.W)
  val vbank_id = UInt(log2Up(b.memDomain.bankNum).W)
  val group_id = UInt(3.W)
  val addr     = UInt(log2Ceil(b.memDomain.bankEntries).W)
  val mask     = Vec(b.memDomain.bankMaskLen, Bool())
  val data     = UInt(b.memDomain.bankWidth.W)
  val wmode    = Bool()
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
  private val hartLines      = b.memDomain.bankNum * b.memDomain.bankEntries
  private val minSharedLines = hartLines
  private val hartSlots      = math.max(1, b.memDomain.sharedEntries / hartLines)

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
  // shared_addr = (hart_slot * bankNum * bankEntries) + (vbank_id * bankEntries) + local_addr.
  // hart_slot is derived from hartid and bounded by available shared capacity.
  private def toSharedAddr(
    hartid:     UInt,
    vbank_id:   UInt,
    _group_id:  UInt,
    local_addr: UInt
  ): UInt = {
    val hartSlot  = if (hartSlots == 1) 0.U else (hartid(log2Ceil(hartSlots) - 1, 0) % hartSlots.U)
    val hartPart  = hartSlot * hartLines.U
    val vbankPart = vbank_id * b.memDomain.bankEntries.U
    hartPart + vbankPart + local_addr
  }

  io.read.req.ready  := !io.write.req.valid
  io.write.req.ready := !io.read.req.valid

  val readReqFire = io.read.req.fire
  val readAddr    =
    toSharedAddr(io.read.req.bits.hartid, io.read.req.bits.vbank_id, io.read.req.bits.group_id, io.read.req.bits.addr)

  when(readReqFire) {
    assert(io.read.req.bits.vbank_id < b.memDomain.bankNum.U, "SharedMem: vbank_id out of range")
    assert(io.read.req.bits.addr < b.memDomain.bankEntries.U, "SharedMem: local addr out of range")
  }

  val readData = mem.read(readAddr, readReqFire)

  io.read.resp.valid     := RegNext(readReqFire, init = false.B)
  io.read.resp.bits.data := readData.asUInt

  val writeReqFire = io.write.req.fire

  val writeAddr = toSharedAddr(
    io.write.req.bits.hartid,
    io.write.req.bits.vbank_id,
    io.write.req.bits.group_id,
    io.write.req.bits.addr
  )

  when(writeReqFire) {
    assert(io.write.req.bits.vbank_id < b.memDomain.bankNum.U, "SharedMem: vbank_id out of range")
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
