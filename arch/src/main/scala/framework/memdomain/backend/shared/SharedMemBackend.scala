package framework.memdomain.backend.shared

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.backend.{MemRequestIO, MTraceDPI}
import framework.memdomain.backend.accpipe.AccPipe
import framework.memdomain.frontend.outside_channel.MemConfigerIO
import framework.top.GlobalConfig

class SharedArbReq(b: GlobalConfig) extends Bundle {
  val src_channel = UInt(log2Up(b.memDomain.bankChannel).W)
  val is_write    = Bool()
  val vbank_id    = UInt(log2Up(b.memDomain.bankNum).W)
  val group_id    = UInt(3.W)
  val addr        = UInt(log2Ceil(b.memDomain.bankEntries).W)
  val mask        = Vec(b.memDomain.bankMaskLen, Bool())
  val data        = UInt(b.memDomain.bankWidth.W)
  val wmode       = Bool()
}

@instantiable
class SharedMemBackend(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val mem_req = Vec(b.memDomain.bankChannel, Flipped(new MemRequestIO(b)))
    val config  = Flipped(Decoupled(new MemConfigerIO(b)))

    // Query interface for frontend to get group count
    val query_vbank_id    = Input(UInt(8.W))
    val query_group_count = Output(UInt(4.W))
  })

  val accPipes:  Seq[Instance[AccPipe]]  = Seq.fill(b.memDomain.bankChannel)(Instantiate(new AccPipe(b)))
  val sharedMem: Instance[SharedMem]     = Instantiate(new SharedMem(b))
  val mtraces = Seq.fill(b.memDomain.bankChannel)(Module(new MTraceDPI))
  val sharedAllocTable = RegInit(VecInit(Seq.fill(b.memDomain.bankNum)(false.B)))

  for (mt <- mtraces) {
    mt.io.is_write := 0.U
    mt.io.channel  := 0.U
    mt.io.vbank_id := 0.U
    mt.io.group_id := 0.U
    mt.io.addr     := 0.U
    mt.io.data_lo  := 0.U
    mt.io.data_hi  := 0.U
    mt.io.enable   := false.B
  }

  io.config.ready := true.B
  when(io.config.valid) {
    val cfgVbankIdx = io.config.bits.vbank_id(log2Up(b.memDomain.bankNum) - 1, 0)
    when(io.config.bits.alloc === 2.U) {
      sharedAllocTable(cfgVbankIdx) := true.B
    }.elsewhen(io.config.bits.alloc === 0.U) {
      sharedAllocTable(cfgVbankIdx) := false.B
    }
  }

  // Shared memory treats each vbank as single-group storage.
  val queryVbankIdx = io.query_vbank_id(log2Up(b.memDomain.bankNum) - 1, 0)
  io.query_group_count := Mux(
    b.memDomain.sharedEnable.B && sharedAllocTable(queryVbankIdx),
    1.U(4.W),
    0.U(4.W)
  )

  for (i <- 0 until b.memDomain.bankChannel) {
    accPipes(i).io.mem_req.write <> io.mem_req(i).write
    accPipes(i).io.mem_req.read <> io.mem_req(i).read
    accPipes(i).io.mem_req.bank_id  := io.mem_req(i).bank_id
    accPipes(i).io.mem_req.group_id := io.mem_req(i).group_id
    accPipes(i).io.mem_req.is_shared := io.mem_req(i).is_shared

    accPipes(i).io.sramRead.req.ready  := false.B
    accPipes(i).io.sramRead.resp.valid := false.B
    accPipes(i).io.sramRead.resp.bits  := DontCare

    accPipes(i).io.sramWrite.req.ready  := false.B
    accPipes(i).io.sramWrite.resp.valid := false.B
    accPipes(i).io.sramWrite.resp.bits  := DontCare

    // Group dimension is intentionally disabled in shared backend.
    accPipes(i).io.is_multi := false.B
  }

  sharedMem.io.read.req.valid   := false.B
  sharedMem.io.read.req.bits    := DontCare
  sharedMem.io.read.resp.ready  := false.B
  sharedMem.io.write.req.valid  := false.B
  sharedMem.io.write.req.bits   := DontCare
  sharedMem.io.write.resp.ready := false.B

  // ---------------------------------------------------------------------------
  // Shared request routing arbiter
  // ---------------------------------------------------------------------------
  private def emitTrace(ch: Int, isWrite: UInt, addr: UInt, dataLo: UInt, dataHi: UInt, en: Bool): Unit = {
    mtraces(ch).io.is_write := isWrite
    mtraces(ch).io.channel  := ch.U
    mtraces(ch).io.vbank_id := io.mem_req(ch).bank_id
    mtraces(ch).io.group_id := io.mem_req(ch).group_id
    mtraces(ch).io.addr     := addr
    mtraces(ch).io.data_lo  := dataLo
    mtraces(ch).io.data_hi  := dataHi
    mtraces(ch).io.enable   := en
  }

  val sharedReqArb = Module(new RRArbiter(new SharedArbReq(b), b.memDomain.bankChannel))

  for (i <- 0 until b.memDomain.bankChannel) {
    val routeShared = b.memDomain.sharedEnable.B && sharedAllocTable(io.mem_req(i).bank_id)
    val useRead     = accPipes(i).io.sramRead.req.valid
    val useWrite    = !useRead && accPipes(i).io.sramWrite.req.valid

    when(io.mem_req(i).read.req.fire) {
      emitTrace(i, 0.U, io.mem_req(i).read.req.bits.addr, 0.U, 0.U, true.B)
    }
    when(io.mem_req(i).write.req.fire) {
      emitTrace(
        i,
        1.U,
        io.mem_req(i).write.req.bits.addr,
        io.mem_req(i).write.req.bits.data(63, 0),
        io.mem_req(i).write.req.bits.data(127, 64),
        true.B
      )
    }

    sharedReqArb.io.in(i).valid            := routeShared && (useRead || useWrite)
    sharedReqArb.io.in(i).bits.src_channel := i.U
    sharedReqArb.io.in(i).bits.is_write    := useWrite
    sharedReqArb.io.in(i).bits.vbank_id    := io.mem_req(i).bank_id
    sharedReqArb.io.in(i).bits.group_id    := 0.U
    sharedReqArb.io.in(i).bits.addr := Mux(
      useRead,
      accPipes(i).io.sramRead.req.bits.addr,
      accPipes(i).io.sramWrite.req.bits.addr
    )
    sharedReqArb.io.in(i).bits.mask        := Mux(
      useRead,
      VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B)),
      accPipes(i).io.sramWrite.req.bits.mask
    )
    sharedReqArb.io.in(i).bits.data        := Mux(useRead, 0.U, accPipes(i).io.sramWrite.req.bits.data)
    sharedReqArb.io.in(i).bits.wmode       := Mux(useRead, false.B, accPipes(i).io.sramWrite.req.bits.wmode)

    when(routeShared && useRead) {
      accPipes(i).io.sramRead.req.ready := sharedReqArb.io.in(i).ready
    }
    when(routeShared && useWrite) {
      accPipes(i).io.sramWrite.req.ready := sharedReqArb.io.in(i).ready
    }
  }

  // Keep one in-flight request per response type so source-channel mapping is exact.
  val readReqPending  = RegInit(false.B)
  val writeReqPending = RegInit(false.B)
  val readReqSrcCh    = Reg(UInt(log2Up(b.memDomain.bankChannel).W))
  val writeReqSrcCh   = Reg(UInt(log2Up(b.memDomain.bankChannel).W))

  val readRespBufValid = RegInit(false.B)
  val readRespBufSrcCh = Reg(UInt(log2Up(b.memDomain.bankChannel).W))
  val readRespBufData  = Reg(UInt(b.memDomain.bankWidth.W))

  val writeRespBufValid = RegInit(false.B)
  val writeRespBufSrcCh = Reg(UInt(log2Up(b.memDomain.bankChannel).W))
  val writeRespBufOk    = Reg(Bool())

  val arbIsWrite    = sharedReqArb.io.out.bits.is_write
  val canIssueRead  = !readReqPending && !readRespBufValid
  val canIssueWrite = !writeReqPending && !writeRespBufValid

  sharedMem.io.read.req.valid         := sharedReqArb.io.out.valid && !arbIsWrite && canIssueRead
  sharedMem.io.read.req.bits.vbank_id := sharedReqArb.io.out.bits.vbank_id
  sharedMem.io.read.req.bits.group_id := sharedReqArb.io.out.bits.group_id
  sharedMem.io.read.req.bits.addr     := sharedReqArb.io.out.bits.addr

  sharedMem.io.write.req.valid         := sharedReqArb.io.out.valid && arbIsWrite && canIssueWrite
  sharedMem.io.write.req.bits.vbank_id := sharedReqArb.io.out.bits.vbank_id
  sharedMem.io.write.req.bits.group_id := sharedReqArb.io.out.bits.group_id
  sharedMem.io.write.req.bits.addr     := sharedReqArb.io.out.bits.addr
  sharedMem.io.write.req.bits.mask     := sharedReqArb.io.out.bits.mask
  sharedMem.io.write.req.bits.data     := sharedReqArb.io.out.bits.data
  sharedMem.io.write.req.bits.wmode    := sharedReqArb.io.out.bits.wmode

  sharedReqArb.io.out.ready := Mux(
    arbIsWrite,
    canIssueWrite && sharedMem.io.write.req.ready,
    canIssueRead && sharedMem.io.read.req.ready
  )

  when(sharedMem.io.read.req.fire) {
    assert(!readReqPending, "SharedMemBackend: read request issued while previous read pending")
    readReqPending := true.B
    readReqSrcCh   := sharedReqArb.io.out.bits.src_channel
  }

  when(sharedMem.io.write.req.fire) {
    assert(!writeReqPending, "SharedMemBackend: write request issued while previous write pending")
    writeReqPending := true.B
    writeReqSrcCh   := sharedReqArb.io.out.bits.src_channel
  }

  // Always absorb SharedMem pulses into local response buffers.
  sharedMem.io.read.resp.ready  := !readRespBufValid
  sharedMem.io.write.resp.ready := !writeRespBufValid

  when(sharedMem.io.read.resp.fire) {
    assert(readReqPending, "SharedMemBackend: read response arrived without pending request")
    readRespBufValid := true.B
    readRespBufSrcCh := readReqSrcCh
    readRespBufData  := sharedMem.io.read.resp.bits.data
    readReqPending   := false.B
  }

  when(sharedMem.io.write.resp.fire) {
    assert(writeReqPending, "SharedMemBackend: write response arrived without pending request")
    writeRespBufValid := true.B
    writeRespBufSrcCh := writeReqSrcCh
    writeRespBufOk    := sharedMem.io.write.resp.bits.ok
    writeReqPending   := false.B
  }

  // ---------------------------------------------------------------------------
  // Response demux back to requesting channel
  // ---------------------------------------------------------------------------
  for (i <- 0 until b.memDomain.bankChannel) {
    when(readRespBufValid && (readRespBufSrcCh === i.U)) {
      accPipes(i).io.sramRead.resp.valid     := true.B
      accPipes(i).io.sramRead.resp.bits.data := readRespBufData

      when(accPipes(i).io.sramRead.resp.fire) {
        readRespBufValid := false.B
      }
    }

    when(writeRespBufValid && (writeRespBufSrcCh === i.U)) {
      accPipes(i).io.sramWrite.resp.valid   := true.B
      accPipes(i).io.sramWrite.resp.bits.ok := writeRespBufOk

      when(accPipes(i).io.sramWrite.resp.fire) {
        writeRespBufValid := false.B
      }
    }
  }
}
