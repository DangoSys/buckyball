package framework.memdomain.backend.mmio

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

/**
 * MmioRouter: translates Ball mmioRead requests into MmioBank requests.
 *
 * Per configured Blink MMIO line:
 *   1. Receive a globally encoded MMIO byte address.
 *   2. Decode the byte address:
 *        bank_id      = absAddr % mmioBankNum
 *        rowInBank    = absAddr / mmioBankNum
 *   3. Route req to bankPorts(bank_id).
 *
 * Concurrency rule (enforced by software):
 *   At any time, each mmio_bank is accessed by at most one requester.
 *   Therefore no arbitration is required — bank ports are driven by exactly the matching Ball.
 */
@instantiable
class MmioRouter(val b: GlobalConfig) extends Module {

  private val clientNum   = b.ballDomain.ballIdMappings.map(_.mmioReadBW).sum
  private val mmioBankNum = b.memDomain.mmioBankNum

  @public
  val io = IO(new Bundle {
    // From Balls (via MemDomain wiring)
    val ballReq  = Vec(clientNum, Flipped(Decoupled(new MmioReadReq(b))))
    val ballResp = Vec(clientNum, Decoupled(new MmioReadResp(b)))

    // Bank ports (one per MmioBank)
    val bankReadReq  = Vec(mmioBankNum, Decoupled(new MmioBankReadReq(b)))
    val bankReadResp =
      Vec(mmioBankNum, Flipped(Decoupled(new MmioBankReadResp(b))))
  })

  val ballAbsAddr = Wire(
    Vec(clientNum, UInt(log2Ceil(b.memDomain.mmioTotalBytes).W))
  )

  val ballBankId = Wire(Vec(clientNum, UInt(log2Ceil(mmioBankNum).W)))

  val ballRowInBank = Wire(
    Vec(clientNum, UInt(log2Ceil(b.memDomain.mmioBankEntries).W))
  )

  for (i <- 0 until clientNum) {
    ballAbsAddr(i)   := io.ballReq(i).bits.addr
    ballBankId(i)    := (ballAbsAddr(i) % mmioBankNum.U)(
      log2Ceil(mmioBankNum) - 1,
      0
    )
    ballRowInBank(i) := (ballAbsAddr(i) / mmioBankNum.U)(
      log2Ceil(b.memDomain.mmioBankEntries) - 1,
      0
    )

    assert(
      !(io.ballReq(i).valid && (ballAbsAddr(i) >= b.memDomain.mmioTotalBytes.U)),
      "MmioRouter: read exceeds the MMIO address space\n"
    )
  }

  // Per-bank request routing
  for (bankIdx <- 0 until mmioBankNum) {
    if (clientNum == 0) {
      io.bankReadReq(bankIdx).valid     := false.B
      io.bankReadReq(bankIdx).bits.addr := 0.U
    } else {
      val ballMatches = VecInit((0 until clientNum).map { i =>
        io.ballReq(i).valid && (ballBankId(i) === bankIdx.U)
      })
      val matchedBall = PriorityEncoder(ballMatches)
      val anyMatch    = ballMatches.asUInt.orR

      io.bankReadReq(bankIdx).valid     := anyMatch
      io.bankReadReq(bankIdx).bits.addr := ballRowInBank(matchedBall)

      assert(
        PopCount(ballMatches) <= 1.U,
        "MmioRouter: bank %d hit by multiple requesters in the same cycle (software invariant violated)\n",
        bankIdx.U
      )
    }
  }

  // Per-Ball req.ready
  for (i <- 0 until clientNum) {
    io.ballReq(i).ready := io.bankReadReq(ballBankId(i)).ready
  }

  // Response routing: each Ball has a small skid buffer holding in-flight bank_id
  val inFlightBankPerBall = Seq.fill(clientNum)(
    Module(
      new Queue(
        UInt(log2Ceil(mmioBankNum).W),
        entries = 2,
        pipe = true,
        flow = false
      )
    )
  )

  for (i <- 0 until clientNum) {
    inFlightBankPerBall(i).io.enq.valid := io.ballReq(i).fire
    inFlightBankPerBall(i).io.enq.bits  := ballBankId(i)
  }

  for (i <- 0 until clientNum) {
    val q          = inFlightBankPerBall(i)
    val srcBank    = q.io.deq.bits
    val srcRespVal = io.bankReadResp(srcBank).valid

    io.ballResp(i).valid     := q.io.deq.valid && srcRespVal
    io.ballResp(i).bits.data := io.bankReadResp(srcBank).bits.data

    q.io.deq.ready := io.ballResp(i).ready && srcRespVal
  }

  for (bankIdx <- 0 until mmioBankNum) {
    if (clientNum == 0) {
      io.bankReadResp(bankIdx).ready := false.B
    } else {
      val anyExpect = VecInit((0 until clientNum).map { i =>
        inFlightBankPerBall(i).io.deq.valid &&
        (inFlightBankPerBall(i).io.deq.bits === bankIdx.U) &&
        io.ballResp(i).ready
      }).asUInt.orR
      io.bankReadResp(bankIdx).ready := anyExpect
    }
  }
}
