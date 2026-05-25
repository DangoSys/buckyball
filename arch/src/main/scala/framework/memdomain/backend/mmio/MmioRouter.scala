package framework.memdomain.backend.mmio

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

/**
 * MmioRouter: translates Ball mmioRead requests into MmioBank requests.
 *
 * Per Ball (parameter `ballNum` from ballDomain):
 *   1. Read meta_bank from current cmd (provided externally by the Router's parent).
 *   2. Look up regionTable[meta_bank].
 *   3. Compute absolute byte address = entry.mmio_addr + req.rel_addr.
 *   4. Decode (bank_id, entry, byte_offset):
 *        bytesPerRow  = mmioBankWidth / 8
 *        rowsPerBank  = mmioBankEntries
 *        bytesPerBank = bytesPerRow * rowsPerBank
 *        bank_id      = absAddr / bytesPerBank
 *        rowInBank    = (absAddr % bytesPerBank) / bytesPerRow
 *        byteOffset   = absAddr % bytesPerRow
 *   5. Route req to bankPorts(bank_id).
 *
 * Concurrency rule (enforced by software):
 *   At any time, each mmio_bank is accessed by at most one Ball.
 *   Therefore no arbitration is required — bank ports are driven by exactly the matching Ball.
 */
@instantiable
class MmioRouter(val b: GlobalConfig) extends Module {

  private val ballNum      = b.ballDomain.ballNum
  private val mmioBankNum  = b.memDomain.mmioBankNum
  private val bytesPerRow  = b.memDomain.mmioBankWidth / 8
  private val bytesPerBank = bytesPerRow * b.memDomain.mmioBankEntries

  @public
  val io = IO(new Bundle {
    // From Balls (via MemDomain wiring)
    val ballReq      = Vec(ballNum, Flipped(Decoupled(new MmioReadReq(b))))
    val ballResp     = Vec(ballNum, Decoupled(new MmioReadResp(b)))
    val ballMetaBank = Input(Vec(ballNum, UInt(log2Up(b.memDomain.bankNum).W)))

    // Region table lookup
    val tableLookupBank   =
      Output(Vec(ballNum, UInt(log2Up(b.memDomain.bankNum).W)))
    val tableLookupResult = Input(Vec(ballNum, new MmioRegionEntry(b)))

    // Bank ports (one per MmioBank)
    val bankReadReq  = Vec(mmioBankNum, Decoupled(new MmioBankReadReq(b)))
    val bankReadResp =
      Vec(mmioBankNum, Flipped(Decoupled(new MmioBankReadResp(b))))
  })

  io.tableLookupBank := io.ballMetaBank

  val ballAbsAddr = Wire(
    Vec(ballNum, UInt(log2Ceil(b.memDomain.mmioTotalBytes).W))
  )

  val ballBankId = Wire(Vec(ballNum, UInt(log2Ceil(mmioBankNum).W)))

  val ballRowInBank = Wire(
    Vec(ballNum, UInt(log2Ceil(b.memDomain.mmioBankEntries).W))
  )

  val ballByteOff = Wire(Vec(ballNum, UInt(log2Ceil(bytesPerRow).W)))
  val ballValid   = Wire(Vec(ballNum, Bool()))

  for (i <- 0 until ballNum) {
    val entry = io.tableLookupResult(i)
    ballValid(i)     := entry.valid
    ballAbsAddr(i)   := entry.mmio_addr + io.ballReq(i).bits.rel_addr
    ballBankId(i)    := (ballAbsAddr(i) / bytesPerBank.U)(
      log2Ceil(mmioBankNum) - 1,
      0
    )
    ballRowInBank(i) := ((ballAbsAddr(i) % bytesPerBank.U) / bytesPerRow.U)(
      log2Ceil(b.memDomain.mmioBankEntries) - 1,
      0
    )
    ballByteOff(i)   := (ballAbsAddr(i) % bytesPerRow.U)(
      log2Ceil(bytesPerRow) - 1,
      0
    )

    assert(
      !(io.ballReq(i).valid && !ballValid(i)),
      "MmioRouter: Ball %d issued mmioRead with no active region binding (meta_bank=%d)\n",
      i.U,
      io.ballMetaBank(i)
    )
  }

  // Per-bank request routing
  for (bankIdx <- 0 until mmioBankNum) {
    val ballMatches = VecInit((0 until ballNum).map { i =>
      io.ballReq(i).valid && ballValid(i) && (ballBankId(i) === bankIdx.U)
    })
    val matchedBall = PriorityEncoder(ballMatches)
    val anyMatch    = ballMatches.asUInt.orR

    io.bankReadReq(bankIdx).valid            := anyMatch
    io.bankReadReq(bankIdx).bits.addr        := ballRowInBank(matchedBall)
    io.bankReadReq(bankIdx).bits.byte_offset := ballByteOff(matchedBall)

    assert(
      PopCount(ballMatches) <= 1.U,
      "MmioRouter: bank %d hit by multiple Balls in the same cycle (software invariant violated)\n",
      bankIdx.U
    )
  }

  // Per-Ball req.ready
  for (i <- 0 until ballNum) {
    io.ballReq(i).ready := ballValid(i) && io.bankReadReq(ballBankId(i)).ready
  }

  // Response routing: each Ball has a small skid buffer holding in-flight bank_id
  val inFlightBankPerBall = Seq.fill(ballNum)(
    Module(
      new Queue(
        UInt(log2Ceil(mmioBankNum).W),
        entries = 2,
        pipe = true,
        flow = false
      )
    )
  )

  for (i <- 0 until ballNum) {
    inFlightBankPerBall(i).io.enq.valid := io.ballReq(i).fire
    inFlightBankPerBall(i).io.enq.bits  := ballBankId(i)
  }

  for (i <- 0 until ballNum) {
    val q          = inFlightBankPerBall(i)
    val srcBank    = q.io.deq.bits
    val srcRespVal = io.bankReadResp(srcBank).valid

    io.ballResp(i).valid     := q.io.deq.valid && srcRespVal
    io.ballResp(i).bits.data := io.bankReadResp(srcBank).bits.data

    q.io.deq.ready := io.ballResp(i).ready && srcRespVal
  }

  for (bankIdx <- 0 until mmioBankNum) {
    val anyExpect = VecInit((0 until ballNum).map { i =>
      inFlightBankPerBall(i).io.deq.valid &&
      (inFlightBankPerBall(i).io.deq.bits === bankIdx.U) &&
      io.ballResp(i).ready
    }).asUInt.orR
    io.bankReadResp(bankIdx).ready := anyExpect
  }
}
