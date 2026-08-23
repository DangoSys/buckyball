package framework.memdomain.backend.mmio

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.backend.banks.SramWriteIO
import framework.top.GlobalConfig

/**
 * MmioPool: top-level MMIO subsystem wrapper.
 *
 * Instantiates:
 *   - Vec of MmioBanks (one per mmioBankNum)
 *   - MmioRouter (translates Ball requests to bank requests)
 *
 * External interfaces:
 *   - write: from MemLoader (mvin_mmio DMA path)
 *   - ballReq/ballResp/ballWriteReq: from Balls (via MemDomain wiring)
 */
@instantiable
class MmioPool(val b: GlobalConfig) extends Module {

  private val mmioReadPorts  = b.ballDomain.ballIdMappings.map(_.mmioReadBW).sum
  private val mmioWritePorts = b.ballDomain.ballIdMappings.map(_.mmioWriteBW).sum

  @public
  val io = IO(new Bundle {
    // DMA write path (from MemLoader)
    val write     = new SramWriteIO(b)
    val writeAddr = Input(UInt(log2Ceil(b.memDomain.mmioTotalBytes).W))

    // Ball read path (from MemDomain, one port per configured Blink line)
    val ballReq      =
      Vec(mmioReadPorts, Flipped(Decoupled(new MmioReadReq(b))))
    val ballResp     = Vec(mmioReadPorts, Decoupled(new MmioReadResp(b)))
    val ballWriteReq = Vec(mmioWritePorts, Flipped(Decoupled(new MmioWriteReq(b))))
  })

  // Instantiate components
  val banks  = Seq.fill(b.memDomain.mmioBankNum)(Instantiate(new MmioBank(b)))
  val router = Instantiate(new MmioRouter(b))

  // MMIO read addresses are already globally encoded; no region translation.
  router.io.ballReq <> io.ballReq
  io.ballResp <> router.io.ballResp

  // Wire router to banks (read path)
  for (i <- 0 until b.memDomain.mmioBankNum) {
    banks(i).io.read.req <> router.io.bankReadReq(i)
    banks(i).io.read.resp <> router.io.bankReadResp(i)
  }

  // A DMA beat is 128 bits. The globally encoded byte address is striped over
  // the homogeneous 8-bit MMIO banks, at most one byte per bank each cycle.
  val dmaBytes       = b.memDomain.bankWidth / 8
  val writeActive    = RegInit(false.B)
  val writeAddrReg   = RegInit(0.U(log2Ceil(b.memDomain.mmioTotalBytes).W))
  val writeDataReg   = RegInit(0.U(b.memDomain.bankWidth.W))
  val writeMaskReg   = RegInit(VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B)))
  val writeCursorReg = RegInit(0.U(log2Ceil(dmaBytes).W))

  io.write.req.ready := !writeActive
  when(io.write.req.fire) {
    assert(io.writeAddr + dmaBytes.U <= b.memDomain.mmioTotalBytes.U, "MmioPool: DMA write exceeds MMIO address space")
    writeActive    := true.B
    writeAddrReg   := io.writeAddr
    writeDataReg   := io.write.req.bits.data
    writeMaskReg   := io.write.req.bits.mask
    writeCursorReg := 0.U
  }

  val finalWriteGroup = writeActive &&
    (writeCursorReg +& b.memDomain.mmioBankNum.U >= dmaBytes.U)
  for (bankIdx <- 0 until b.memDomain.mmioBankNum) {
    val byteMatch      = VecInit((0 until b.memDomain.mmioBankNum).map { offset =>
      val byteIdx  = writeCursorReg +& offset.U
      val byteAddr = writeAddrReg + byteIdx
      byteIdx < dmaBytes.U && (byteAddr % b.memDomain.mmioBankNum.U === bankIdx.U)
    })
    val selectedOffset = PriorityEncoder(byteMatch)
    val selectedByte   = writeCursorReg +& selectedOffset
    val selectedAddr   = writeAddrReg + selectedByte
    val dmaWriteValid  = writeActive && byteMatch.asUInt.orR
    val dmaWriteAddr   = (selectedAddr / b.memDomain.mmioBankNum.U)(log2Ceil(b.memDomain.mmioBankEntries) - 1, 0)
    val selectedData   = (writeDataReg >> (selectedByte << 3.U))(7, 0)
    val dmaWriteData   = Mux(writeMaskReg(selectedByte), selectedData, 0.U)

    if (mmioWritePorts == 0) {
      banks(bankIdx).io.write.req.valid     := dmaWriteValid
      banks(bankIdx).io.write.req.bits.addr := dmaWriteAddr
      banks(bankIdx).io.write.req.bits.data := dmaWriteData
    } else {
      val ballMatches = VecInit((0 until mmioWritePorts).map { i =>
        io.ballWriteReq(i).valid &&
        (io.ballWriteReq(i).bits.addr % b.memDomain.mmioBankNum.U === bankIdx.U)
      })
      val ballIdx     = PriorityEncoder(ballMatches)
      val ballAddr    = io.ballWriteReq(ballIdx).bits.addr
      assert(
        PopCount(ballMatches) <= 1.U,
        "MmioPool: multiple Ball writes target one MMIO bank in a cycle"
      )
      assert(
        !(writeActive && ballMatches.asUInt.orR),
        "MmioPool: DMA and Ball write target one MMIO bank in a cycle"
      )
      banks(bankIdx).io.write.req.valid     := dmaWriteValid || ballMatches.asUInt.orR
      banks(bankIdx).io.write.req.bits.addr := Mux(
        dmaWriteValid,
        dmaWriteAddr,
        (ballAddr / b.memDomain.mmioBankNum.U)(log2Ceil(b.memDomain.mmioBankEntries) - 1, 0)
      )
      banks(bankIdx).io.write.req.bits.data := Mux(
        dmaWriteValid,
        dmaWriteData,
        io.ballWriteReq(ballIdx).bits.data
      )
    }
  }

  for (i <- 0 until mmioWritePorts) {
    val targetBank  = io.ballWriteReq(i).bits.addr % b.memDomain.mmioBankNum.U
    val targetReady = Mux1H(
      UIntToOH(targetBank, b.memDomain.mmioBankNum),
      banks.map(_.io.write.req.ready)
    )
    io.ballWriteReq(i).ready := !writeActive && targetReady
    assert(
      !(io.ballWriteReq(i).valid &&
        (io.ballWriteReq(i).bits.addr >= b.memDomain.mmioTotalBytes.U)),
      "MmioPool: Ball write exceeds the MMIO address space"
    )
  }

  io.write.resp.valid   := finalWriteGroup
  io.write.resp.bits.ok := true.B
  when(io.write.resp.fire) {
    writeActive := false.B
  }.elsewhen(writeActive && !finalWriteGroup) {
    writeCursorReg := writeCursorReg + b.memDomain.mmioBankNum.U
  }
}
