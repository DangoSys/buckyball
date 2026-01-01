package framework.memdomain.backend

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.backend.banks.{SramBank, SramReadIO, SramWriteIO}
import framework.memdomain.backend.accpipe.AccPipe

/**
 * MemBackend: Backend memory manager
 * Manages the physical memory resources (Scratchpad + Accumulator Banks)
 *
 * Features:
 * - Instantiates bankNum SRAM banks and bankChannel AccPipes
 * - Supports up to bankChannel concurrent bank accesses per cycle
 * - All requests go through AccPipe, which handles both direct write mode and accumulator mode (read-modify-write)
 * - Assert checks ensure no bank_id conflicts in the same cycle
 */
class MemRequestIO(b: GlobalConfig) extends Bundle {
  val write   = Flipped(new SramWriteIO(b)) // Sender perspective: send write requests
  val read    = Flipped(new SramReadIO(b))  // Sender perspective: send read requests
  val bank_id = Output(UInt(log2Up(b.memDomain.bankNum).W))
}

@instantiable
class MemBackend(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    // Interface from midend - MemManager routes requests to AccPipes
    val mem_req = Vec(b.memDomain.bankChannel, Flipped(new MemRequestIO(b)))
  })

  val banks:    Seq[Instance[SramBank]] = Seq.fill(b.memDomain.bankNum)(Instantiate(new SramBank(b)))
  val accPipes: Seq[Instance[AccPipe]]  = Seq.fill(b.memDomain.bankChannel)(Instantiate(new AccPipe(b)))

  // -----------------------------------------------------------------------------
  // Request routing: All requests go through AccPipe
  // -----------------------------------------------------------------------------
  // Route all requests to AccPipe (AccPipe handles both wmode and read operations)
  for (i <- 0 until b.memDomain.bankChannel) {
    accPipes(i).io.write <> io.mem_req(i).write
    accPipes(i).io.read <> io.mem_req(i).read
    accPipes(i).io.bank_id := io.mem_req(i).bank_id
  }

  // -----------------------------------------------------------------------------
  // Bank conflict detection
  // -----------------------------------------------------------------------------
  // Check for bank_id conflicts across all requests (all go through AccPipe)
  for (i <- 0 until b.memDomain.bankChannel) {
    for (j <- (i + 1) until b.memDomain.bankChannel) {
      // Check write conflicts
      when(io.mem_req(i).write.req.valid && io.mem_req(j).write.req.valid) {
        assert(
          io.mem_req(i).bank_id =/= io.mem_req(j).bank_id,
          s"[MemManager]: Write Bank ID conflict between request $i and $j"
        )
      }
      // Check read conflicts
      when(io.mem_req(i).read.req.valid && io.mem_req(j).read.req.valid) {
        assert(
          io.mem_req(i).bank_id =/= io.mem_req(j).bank_id,
          s"[MemManager]: Read Bank ID conflict between request $i and $j"
        )
      }
    }
  }

  // -----------------------------------------------------------------------------
  // Initialize all AccPipe sram interfaces to default values
  // -----------------------------------------------------------------------------
  accPipes.foreach { accPipe =>
    accPipe.io.sramWrite.req.ready    := false.B
    accPipe.io.sramWrite.resp.valid   := false.B
    accPipe.io.sramWrite.resp.bits.ok := false.B

    accPipe.io.sramRead.req.ready      := false.B
    accPipe.io.sramRead.resp.valid     := false.B
    accPipe.io.sramRead.resp.bits.data := 0.U
  }

  // -----------------------------------------------------------------------------
  // Bank routing and connection
  // -----------------------------------------------------------------------------
  // Each bank connects to the AccPipe whose current_bank_id matches this bank's ID
  // Since we have conflict detection, each bank can have at most one request per cycle
  banks.zipWithIndex.foreach {
    case (bank, bankIdx) =>
      val bank_id = bankIdx.U

      // Aggregate write requests from all AccPipes for this bank
      val writeMatches = VecInit(accPipes.map { accPipe =>
        accPipe.io.sramWrite.req.valid && accPipe.io.busy && (accPipe.io.current_bank_id === bank_id)
      })
      val hasWriteReq  = writeMatches.asUInt.orR

      // Connect write - only one AccPipe can match per bank per cycle (conflict detection ensures this)
      when(hasWriteReq) {
        val matchedAccPipe = Mux1H(writeMatches.zip(accPipes).map {
          case (isMatch, accPipe) =>
            isMatch -> accPipe.io.sramWrite
        })
        bank.io.sramWrite.req.valid := matchedAccPipe.req.valid
        bank.io.sramWrite.req.bits   := matchedAccPipe.req.bits
        matchedAccPipe.req.ready     := bank.io.sramWrite.req.ready
        matchedAccPipe.resp.valid    := bank.io.sramWrite.resp.valid
        matchedAccPipe.resp.bits     := bank.io.sramWrite.resp.bits
        bank.io.sramWrite.resp.ready := matchedAccPipe.resp.ready
      }.otherwise {
        bank.io.sramWrite.req.valid  := false.B
        bank.io.sramWrite.req.bits   := 0.U.asTypeOf(bank.io.sramWrite.req.bits)
        bank.io.sramWrite.resp.ready := false.B
      }

      // Aggregate read requests from all AccPipes for this bank
      val readMatches = VecInit(accPipes.map { accPipe =>
        accPipe.io.sramRead.req.valid && accPipe.io.busy && (accPipe.io.current_bank_id === bank_id)
      })
      val hasReadReq  = readMatches.asUInt.orR

      // Connect read - only one AccPipe can match per bank per cycle
      when(hasReadReq) {
        val matchedAccPipe = Mux1H(readMatches.zip(accPipes).map {
          case (isMatch, accPipe) =>
            isMatch -> accPipe.io.sramRead
        })
        bank.io.sramRead.req.valid := matchedAccPipe.req.valid
        bank.io.sramRead.req.bits   := matchedAccPipe.req.bits
        matchedAccPipe.req.ready    := bank.io.sramRead.req.ready
        matchedAccPipe.resp.valid   := bank.io.sramRead.resp.valid
        matchedAccPipe.resp.bits    := bank.io.sramRead.resp.bits
        bank.io.sramRead.resp.ready := matchedAccPipe.resp.ready
      }.otherwise {
        bank.io.sramRead.req.valid  := false.B
        bank.io.sramRead.req.bits   := 0.U.asTypeOf(bank.io.sramRead.req.bits)
        bank.io.sramRead.resp.ready := false.B
      }
  }
}
