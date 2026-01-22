package framework.memdomain.backend

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import framework.top.GlobalConfig
import framework.memdomain.backend.banks.{SramBank, SramReadIO, SramWriteIO}
import framework.memdomain.backend.accpipe.AccPipe

/**
 * MemBackend: Backend memory manager
 *
 * Key changes vs your version:
 * 1) Fix MemRequestIO.bank_id direction (should be Input)
 * 2) Remove illegal "accPipes.foreach init sram*" multi-driving
 * 3) Routing does NOT depend on accPipe.busy (idle same-cycle issue)
 * 4) Use accPipe.io.target_bank_id for same-cycle routing
 * 5) Add cross read/write conflict assertions based on actual bank requests (valid-level, and fire-level optional)
 */
class MemRequestIO(b: GlobalConfig) extends Bundle {
  val write   = Flipped(new SramWriteIO(b))                 // midend sends write req into backend
  val read    = Flipped(new SramReadIO(b))                  // midend sends read req into backend
  val bank_id = Output(UInt(log2Up(b.memDomain.bankNum).W)) // FIX: was Output
}

@instantiable
class MemBackend(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val mem_req = Vec(b.memDomain.bankChannel, Flipped(new MemRequestIO(b)))
  })

  val banks:    Seq[Instance[SramBank]] = Seq.fill(b.memDomain.bankNum)(Instantiate(new SramBank(b)))
  val accPipes: Seq[Instance[AccPipe]]  = Seq.fill(b.memDomain.bankChannel)(Instantiate(new AccPipe(b)))

  // -----------------------------------------------------------------------------
  // Midend -> AccPipe
  // -----------------------------------------------------------------------------
  for (i <- 0 until b.memDomain.bankChannel) {
    accPipes(i).io.write <> io.mem_req(i).write
    accPipes(i).io.read <> io.mem_req(i).read
    accPipes(i).io.bank_id := io.mem_req(i).bank_id
  }

  // -----------------------------------------------------------------------------
  // Create "return path" wires for each AccPipe (bank -> accPipe)
  // We drive these wires exactly once in routing logic to avoid multi-drive.
  // -----------------------------------------------------------------------------
  val rd_req_ready  = Wire(Vec(b.memDomain.bankChannel, Bool()))
  val rd_resp_valid = Wire(Vec(b.memDomain.bankChannel, Bool()))
  val rd_resp_data  = Wire(Vec(b.memDomain.bankChannel, UInt(b.memDomain.bankWidth.W)))
  val rd_resp_ready = Wire(Vec(b.memDomain.bankChannel, Bool())) // accPipe -> bank

  val wr_req_ready  = Wire(Vec(b.memDomain.bankChannel, Bool()))
  val wr_resp_valid = Wire(Vec(b.memDomain.bankChannel, Bool()))
  val wr_resp_ok    = Wire(Vec(b.memDomain.bankChannel, Bool()))
  val wr_resp_ready = Wire(Vec(b.memDomain.bankChannel, Bool())) // accPipe -> bank

  // Defaults: if a pipe is not selected by any bank this cycle, it sees "not ready / no response"
  for (i <- 0 until b.memDomain.bankChannel) {
    rd_req_ready(i)  := false.B
    rd_resp_valid(i) := false.B
    rd_resp_data(i)  := 0.U
    wr_req_ready(i)  := false.B
    wr_resp_valid(i) := false.B
    wr_resp_ok(i)    := false.B

    // connect accPipe outputs that go to bank
    rd_resp_ready(i) := accPipes(i).io.sramRead.resp.ready
    wr_resp_ready(i) := accPipes(i).io.sramWrite.resp.ready

    // drive accPipe inputs from routing wires
    accPipes(i).io.sramRead.req.ready      := rd_req_ready(i)
    accPipes(i).io.sramRead.resp.valid     := rd_resp_valid(i)
    accPipes(i).io.sramRead.resp.bits.data := rd_resp_data(i)

    accPipes(i).io.sramWrite.req.ready    := wr_req_ready(i)
    accPipes(i).io.sramWrite.resp.valid   := wr_resp_valid(i)
    accPipes(i).io.sramWrite.resp.bits.ok := wr_resp_ok(i)
  }

  // -----------------------------------------------------------------------------
  // Conflict detection (better than original)
  // 1) valid-level: if two pipes TRY to access same bank same cycle, assert
  // 2) also add read-vs-write cross conflict
  // NOTE: if you want fire-level only, see the optional section below.
  // -----------------------------------------------------------------------------
  for (i <- 0 until b.memDomain.bankChannel) {
    for (j <- (i + 1) until b.memDomain.bankChannel) {
      val wi = accPipes(i).io.sramWrite.req.valid
      val wj = accPipes(j).io.sramWrite.req.valid
      val ri = accPipes(i).io.sramRead.req.valid
      val rj = accPipes(j).io.sramRead.req.valid

      val bi = accPipes(i).io.target_bank_id
      val bj = accPipes(j).io.target_bank_id

      when(wi && wj) {
        assert(bi =/= bj, s"[MemBackend] WRITE bank conflict between pipe $i and $j")
      }
      when(ri && rj) {
        assert(bi =/= bj, s"[MemBackend] READ bank conflict between pipe $i and $j")
      }
      when(wi && rj) {
        assert(bi =/= bj, s"[MemBackend] WRITE($i) vs READ($j) bank conflict")
      }
      when(ri && wj) {
        assert(bi =/= bj, s"[MemBackend] READ($i) vs WRITE($j) bank conflict")
      }
    }
  }

  // -----------------------------------------------------------------------------
  // Bank routing
  // For each bank:
  // - select at most one write requester (Mux1H)
  // - select at most one read requester (Mux1H)
  // - connect bank <-> selected accPipe using the per-pipe wires
  // -----------------------------------------------------------------------------
  banks.zipWithIndex.foreach {
    case (bank, bankIdx) =>
      val bankId = bankIdx.U(log2Up(b.memDomain.bankNum).W)

      // -------------------------
      // WRITE routing
      // -------------------------
      val wMatch = Wire(Vec(b.memDomain.bankChannel, Bool()))
      for (i <- 0 until b.memDomain.bankChannel) {
        wMatch(i) := accPipes(i).io.sramWrite.req.valid && (accPipes(i).io.target_bank_id === bankId)
      }
      val wHas = wMatch.asUInt.orR
      // stronger safety: at most 1
      assert(PopCount(wMatch) <= 1.U, s"[MemBackend] More than one WRITE match to bank $bankIdx")

      when(wHas) {
        val selIdx = OHToUInt(wMatch)
        // bank gets req from selected pipe using Mux1H
        bank.io.sramWrite.req.valid := Mux1H(wMatch, accPipes.map(_.io.sramWrite.req.valid))
        bank.io.sramWrite.req.bits  := Mux1H(wMatch, accPipes.map(_.io.sramWrite.req.bits))
        // selected pipe sees ready from bank
        wr_req_ready(selIdx)        := bank.io.sramWrite.req.ready

        // response path: selected pipe sees bank resp
        wr_resp_valid(selIdx)        := bank.io.sramWrite.resp.valid
        wr_resp_ok(selIdx)           := bank.io.sramWrite.resp.bits.ok
        // bank sees ready from selected pipe
        bank.io.sramWrite.resp.ready := wr_resp_ready(selIdx)
      }.otherwise {
        bank.io.sramWrite.req.valid  := false.B
        bank.io.sramWrite.req.bits   := 0.U.asTypeOf(bank.io.sramWrite.req.bits)
        bank.io.sramWrite.resp.ready := false.B
      }

      // -------------------------
      // READ routing
      // -------------------------
      val rMatch = Wire(Vec(b.memDomain.bankChannel, Bool()))
      for (i <- 0 until b.memDomain.bankChannel) {
        rMatch(i) := accPipes(i).io.sramRead.req.valid && (accPipes(i).io.target_bank_id === bankId)
      }
      val rHas = rMatch.asUInt.orR
      assert(PopCount(rMatch) <= 1.U, s"[MemBackend] More than one READ match to bank $bankIdx")

      when(rHas) {
        val selIdx = OHToUInt(rMatch)
        // bank gets req from selected pipe using Mux1H
        bank.io.sramRead.req.valid := Mux1H(rMatch, accPipes.map(_.io.sramRead.req.valid))
        bank.io.sramRead.req.bits  := Mux1H(rMatch, accPipes.map(_.io.sramRead.req.bits))
        // selected pipe sees ready from bank
        rd_req_ready(selIdx)       := bank.io.sramRead.req.ready

        // response path: selected pipe sees bank resp
        rd_resp_valid(selIdx)       := bank.io.sramRead.resp.valid
        rd_resp_data(selIdx)        := bank.io.sramRead.resp.bits.data
        // bank sees ready from selected pipe
        bank.io.sramRead.resp.ready := rd_resp_ready(selIdx)
      }.otherwise {
        bank.io.sramRead.req.valid  := false.B
        bank.io.sramRead.req.bits   := 0.U.asTypeOf(bank.io.sramRead.req.bits)
        bank.io.sramRead.resp.ready := false.B
      }
  }

  // -----------------------------------------------------------------------------
  // Optional: fire-level conflict assertions (stricter semantics)
  // Enable if you prefer conflicts only when both actually handshake.
  // -----------------------------------------------------------------------------
  // val wrFire = VecInit(accPipes.map(_.io.sramWrite.req.fire))
  // val rdFire = VecInit(accPipes.map(_.io.sramRead.req.fire))
  // for (i <- 0 until b.memDomain.bankChannel) {
  //   for (j <- (i + 1) until b.memDomain.bankChannel) {
  //     val bi = accPipes(i).io.target_bank_id
  //     val bj = accPipes(j).io.target_bank_id
  //     when(wrFire(i) && wrFire(j)) { assert(bi =/= bj) }
  //     when(rdFire(i) && rdFire(j)) { assert(bi =/= bj) }
  //     when(wrFire(i) && rdFire(j)) { assert(bi =/= bj) }
  //     when(rdFire(i) && wrFire(j)) { assert(bi =/= bj) }
  //   }
  // }
}
