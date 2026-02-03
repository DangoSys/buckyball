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
  val write   = Flipped(new SramWriteIO(b)) // midend sends write req into backend
  val read    = Flipped(new SramReadIO(b))  // midend sends read req into backend
  val bank_id = Output(UInt(log2Up(b.memDomain.bankNum).W))
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

      // =========================================================
      // WRITE routing (req: combinational; resp: owner-registered)
      // =========================================================
      val wMatch = Wire(Vec(b.memDomain.bankChannel, Bool()))
      for (i <- 0 until b.memDomain.bankChannel) {
        wMatch(i) := (accPipes(i).io.target_bank_id === bankId) &&
        accPipes(i).io.sramWrite.req.valid
      }
      val wHas = wMatch.asUInt.orR
      val wSel = OHToUInt(wMatch)
      assert(PopCount(wMatch) <= 1.U, s"[MemBackend] More than one WRITE match to bank $bankIdx")

      // owner regs for write resp
      val wOwnerValid = RegInit(false.B)
      val wOwnerIdx   = RegInit(0.U(log2Up(b.memDomain.bankChannel).W))

      // default bank write req
      bank.io.sramWrite.req.valid := false.B
      bank.io.sramWrite.req.bits  := 0.U.asTypeOf(bank.io.sramWrite.req.bits)

      // issue req from selected pipe
      when(wHas) {
        bank.io.sramWrite.req.valid := true.B
        bank.io.sramWrite.req.bits  := Mux1H(wMatch, accPipes.map(_.io.sramWrite.req.bits))
        // ready to selected pipe
        wr_req_ready(wSel)          := bank.io.sramWrite.req.ready

        when(bank.io.sramWrite.req.fire) {
          wOwnerValid := true.B
          wOwnerIdx   := wSel
        }
      }

      // resp routes by owner (NOT by wHas)
      // drive selected pipe resp wires
      when(wOwnerValid) {
        wr_resp_valid(wOwnerIdx) := bank.io.sramWrite.resp.valid
        wr_resp_ok(wOwnerIdx)    := bank.io.sramWrite.resp.bits.ok
      }
      // ready from owner pipe
      bank.io.sramWrite.resp.ready := Mux(wOwnerValid, wr_resp_ready(wOwnerIdx), false.B)

      when(bank.io.sramWrite.resp.fire && wOwnerValid) {
        wOwnerValid := false.B
      }

      // =========================================================
      // READ routing (req: combinational; resp: owner-registered)
      // =========================================================
      val rMatch        = Wire(Vec(b.memDomain.bankChannel, Bool()))
      // owner regs for read resp
      val rOwnerValid   = RegInit(false.B)
      val rOwnerIdx     = RegInit(0.U(log2Up(b.memDomain.bankChannel).W))
      val canAcceptRead = !rOwnerValid
      for (i <- 0 until b.memDomain.bankChannel) {
        rMatch(i) := canAcceptRead && (accPipes(i).io.target_bank_id === bankId) && accPipes(i).io.sramRead.req.valid
      }
      val rHas = rMatch.asUInt.orR
      val rSel = OHToUInt(rMatch)
      // stronger safety: at most 1
      assert(PopCount(rMatch) <= 1.U, s"[MemBackend] More than one READ match to bank $bankIdx")

      // default bank read req
      bank.io.sramRead.req.valid := false.B
      bank.io.sramRead.req.bits  := 0.U.asTypeOf(bank.io.sramRead.req.bits)

      // issue req from selected pipe
      when(rHas) {
        bank.io.sramRead.req.valid := true.B
        bank.io.sramRead.req.bits  := Mux1H(rMatch, accPipes.map(_.io.sramRead.req.bits))
        // ready to selected pipe
        rd_req_ready(rSel)         := bank.io.sramRead.req.ready

        when(bank.io.sramRead.req.fire) {
          rOwnerValid := true.B
          rOwnerIdx   := rSel
        }
      }

      // resp routes by owner (NOT by rHas)
      when(rOwnerValid) {
        rd_resp_valid(rOwnerIdx) := bank.io.sramRead.resp.valid
        rd_resp_data(rOwnerIdx)  := bank.io.sramRead.resp.bits.data
      }
      // bank sees ready from owner pipe
      bank.io.sramRead.resp.ready := Mux(rOwnerValid, rd_resp_ready(rOwnerIdx), false.B)

      when(bank.io.sramRead.resp.fire && rOwnerValid) {
        rOwnerValid := false.B
      }
  }
}
