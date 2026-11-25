package framework.ball.mmio_bank

import framework.generic.contract.ball.BallHostParam

import framework.generic.arbiter.{Arbiter, RRArbiter}
import chisel3.experimental.hierarchy.{Instance, Instantiate}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.mmio_bank.MetadataWrite

class MmioReadReq(link: BlinkParam) extends Bundle { val addr = UInt(log2Ceil(link.mmioBytes).W) }
class MmioReadResp(link: BlinkParam) extends Bundle { val data = UInt(link.mmioDataBits.W) }

class MmioRead(link: BlinkParam) extends Bundle {
  val req  = Flipped(Decoupled(new MmioReadReq(link)))
  val resp = Decoupled(new MmioReadResp(link))
}

class MmioWriteReq(link: BlinkParam) extends Bundle {
  val addr = UInt(log2Ceil(link.mmioBytes).W)
  val data = UInt(link.mmioDataBits.W)
}

/** Banked byte metadata SRAM; arbitration and response ownership remain local. */
@instantiable
class MmioBank(
  host:    BallHostParam,
  readers: Int,
  writers: Int)
    extends Module {
  require(host.mmioEnable && host.mmioBankWidth == 8 && host.link.mmioDataBits == 8)
  val bytes    = host.link.bankWidth / 8
  val bankBits = math.max(1, log2Ceil(host.mmioBankNum))
  val rowBits  = math.max(1, log2Ceil(host.mmioBankEntries))

  @public
  val io = IO(new Bundle {
    val upload = new MetadataWrite(host.link.bankWidth, host.link.mmioBytes)
    val reads  = Vec(readers, new MmioRead(host.link))
    val writes = Vec(writers, Flipped(Decoupled(new MmioWriteReq(host.link))))
  })

  val idle :: storing :: respond :: Nil = Enum(3)
  val state                             = RegInit(idle)
  val beat                              = Reg(chiselTypeOf(io.upload.request.bits))
  val cursor                            = RegInit(0.U(math.max(1, log2Ceil(bytes)).W))
  io.upload.request.ready             := state === idle
  io.upload.response.valid            := state === respond
  io.upload.response.bits             := true.B
  when(io.upload.request.fire) {
    assert(io.upload.request.bits.addr +& bytes.U <= host.link.mmioBytes.U)
    beat   := io.upload.request.bits
    cursor := 0.U
    state  := storing
  }
  when(state === storing) {
    when(cursor +& host.mmioBankNum.U >= bytes.U)(state := respond)
      .otherwise(cursor := cursor + host.mmioBankNum.U)
  }
  when(io.upload.response.fire)(state := idle)

  val readerBusy = RegInit(VecInit(Seq.fill(readers)(false.B)))
  val readReady  = Wire(Vec(readers, Vec(host.mmioBankNum, Bool())))
  val readValid  = Wire(Vec(readers, Vec(host.mmioBankNum, Bool())))
  val readData   = Wire(Vec(readers, Vec(host.mmioBankNum, UInt(8.W))))
  val writeReady = Wire(Vec(writers, Vec(host.mmioBankNum, Bool())))
  for (bank   <- 0 until host.mmioBankNum) {
    val memory  = SyncReadMem(host.mmioBankEntries, UInt(8.W))
    val writes  = Instantiate(new Arbiter(new MmioWriteReq(host.link), writers + 1))
    val offsets = (0 until host.mmioBankNum).map { offset =>
      val index = cursor +& offset.U
      index < bytes.U && (beat.addr + index) % host.mmioBankNum.U === bank.U
    }
    val index   = cursor +& PriorityEncoder(VecInit(offsets))
    writes.io.in(0).valid     := state === storing && VecInit(offsets).asUInt.orR
    writes.io.in(0).bits.addr := beat.addr + index
    writes.io.in(0).bits.data := Mux(beat.mask(index(log2Ceil(bytes) - 1, 0)), (beat.data >> (index << 3))(7, 0), 0.U)
    for (client <- 0 until writers) {
      val selected = io.writes(client).bits.addr % host.mmioBankNum.U === bank.U
      writes.io.in(client + 1).valid := io.writes(client).valid && selected
      writes.io.in(client + 1).bits  := io.writes(client).bits
      writeReady(client)(bank)       := selected && writes.io.in(client + 1).ready
    }
    writes.io.out.ready := true.B
    val pending = RegInit(false.B)
    val held    = RegInit(false.B)
    val owner   = Reg(UInt(math.max(1, log2Ceil(readers)).W))
    val value   = Reg(UInt(8.W))
    val reads   = Instantiate(new RRArbiter(new MmioReadReq(host.link), math.max(1, readers)))
    if (readers == 0) {
      reads.io.in(0).valid := false.B
      reads.io.in(0).bits  := 0.U.asTypeOf(reads.io.in(0).bits)
    }
    for (client <- 0 until readers) {
      val selected = io.reads(client).req.bits.addr % host.mmioBankNum.U === bank.U
      reads.io.in(client).valid := io.reads(client).req.valid && selected && !readerBusy(client)
      reads.io.in(client).bits  := io.reads(client).req.bits
      readReady(client)(bank)   := selected && !readerBusy(client) && reads.io.in(client).ready
      readValid(client)(bank)   := held && owner === client.U
      readData(client)(bank)    := value
    }
    reads.io.out.ready := !writes.io.out.valid && !pending && !held
    val raw     = memory.readWrite(
      (Mux(writes.io.out.valid, writes.io.out.bits.addr, reads.io.out.bits.addr) / host.mmioBankNum.U)(rowBits - 1, 0),
      writes.io.out.bits.data,
      writes.io.out.fire || reads.io.out.fire,
      writes.io.out.fire
    )
    pending                       := reads.io.out.fire
    when(reads.io.out.fire)(owner := reads.io.chosen)
    when(pending) { value := raw; held := true.B }
    if (readers > 0)
      when(held && (if (readers == 1) io.reads(0).resp.ready else io.reads(owner).resp.ready))(held := false.B)
  }
  for (client <- 0 until readers) {
    io.reads(client).req.ready                          := readReady(client).asUInt.orR
    io.reads(client).resp.valid                         := readValid(client).asUInt.orR
    io.reads(client).resp.bits.data                     := Mux1H(readValid(client), readData(client))
    when(io.reads(client).req.fire)(readerBusy(client)  := true.B)
    when(io.reads(client).resp.fire)(readerBusy(client) := false.B)
  }
  for (client <- 0 until writers) io.writes(client).ready := writeReady(client).asUInt.orR
}
