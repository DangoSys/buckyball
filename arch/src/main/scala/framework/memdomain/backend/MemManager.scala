package framework.memdomain.backend

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import framework.memdomain.MemDomainParam
import framework.memdomain.backend.banks.{SramBank, SramReadIO, SramWriteIO}
import framework.memdomain.backend.accpipe.AccPipe

/**
 * MemManager: Backend memory manager
 * Manages the physical memory resources (Scratchpad + Accumulator Banks)
 *
 * Features:
 * - Instantiates bankNum SRAM banks and bankChannel AccPipes
 * - Supports up to bankChannel concurrent bank accesses per cycle
 * - All requests go through AccPipe, which handles both direct write mode and accumulator mode (read-modify-write)
 * - Assert checks ensure no bank_id conflicts in the same cycle
 */
class MemRequestIO(parameter: MemDomainParam) extends Bundle {
  val write   = Flipped(new SramWriteIO(parameter.bankEntries, parameter.bankWidth, parameter.bankMaskLen))
  val read    = Flipped(new SramReadIO(parameter.bankEntries, parameter.bankWidth))
  val bank_id = Input(UInt(log2Up(parameter.bankNum).W))
}

@instantiable
class MemManager(val parameter: MemDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[MemDomainParam] {

  @public
  val io = IO(new Bundle {
    // Interface from midend (new architecture)
    val mem_req = Vec(parameter.bankChannel, new MemRequestIO(parameter))
  })

  // Instantiate bankNum SRAM banks
  val banks: Seq[Instance[SramBank]] = Seq.fill(parameter.bankNum) {
    Instantiate(new SramBank(parameter))
  }

  // Instantiate bankChannel accumulator pipes (all requests go through AccPipe)
  val accPipes: Seq[Instance[AccPipe]] = Seq.fill(parameter.bankChannel) {
    Instantiate(new AccPipe(parameter))
  }

  // -----------------------------------------------------------------------------
  // Request routing: All requests go through AccPipe
  // -----------------------------------------------------------------------------
  // Route all requests to AccPipe (AccPipe handles both wmode and read operations)
  for (i <- 0 until parameter.bankChannel) {
    accPipes(i).io.write <> io.mem_req(i).write
    accPipes(i).io.read <> io.mem_req(i).read
    accPipes(i).io.bank_id := io.mem_req(i).bank_id
  }

  // -----------------------------------------------------------------------------
  // Bank conflict detection
  // -----------------------------------------------------------------------------
  // Check for bank_id conflicts across all requests (all go through AccPipe)
  for (i <- 0 until parameter.bankChannel) {
    for (j <- (i + 1) until parameter.bankChannel) {
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
  // Bank routing and connection
  // -----------------------------------------------------------------------------
  // Each bank connects to the AccPipe whose current_bank_id matches this bank's ID
  // Since we have conflict detection, each bank can have at most one request per cycle
  banks.zipWithIndex.foreach {
    case (bank, bankIdx) =>
      val bank_id = bankIdx.U

      // Default: no write request
      bank.io.sramWrite.req.valid := false.B

      // Connect write request: find matching AccPipe and connect
      accPipes.foreach { accPipe =>
        val isMatch = accPipe.io.sramWrite.req.valid && accPipe.busy && (accPipe.current_bank_id === bank_id)
        when(isMatch) {
          bank.io.sramWrite.req <> accPipe.io.sramWrite.req
          accPipe.io.sramWrite.resp <> bank.io.sramWrite.resp
        }
      }

      // Default: no read request
      bank.io.sramRead.req.valid := false.B

      // Connect read request: find matching AccPipe and connect
      accPipes.foreach { accPipe =>
        val isMatch = accPipe.io.sramRead.req.valid && accPipe.busy && (accPipe.current_bank_id === bank_id)
        when(isMatch) {
          bank.io.sramRead.req <> accPipe.io.sramRead.req
          accPipe.io.sramRead.resp <> bank.io.sramRead.resp
        }
      }
  }
}
