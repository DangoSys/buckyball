package framework.balldomain.prototype.trace

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.memdomain.backend.banks.SramBank
import framework.top.GlobalConfig

/**
 * Trace — TraceBall inner processing unit.
 *
 * Handles cycle counter management (START/STOP/READ).
 */
@instantiable
class Trace(val b: GlobalConfig) extends Module {

  val ballMapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "TraceBall")
    .getOrElse(throw new IllegalArgumentException("TraceBall not found in config"))

  val inBW  = ballMapping.inBW
  val outBW = ballMapping.outBW

  val bankWidth = b.memDomain.bankWidth

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  // ============================================================
  // Constants
  // ============================================================
  val CTR_START = 0.U(4.W)
  val CTR_STOP  = 1.U(4.W)
  val CTR_READ  = 2.U(4.W)

  val NUM_COUNTERS = 16

  // ============================================================
  // State machine
  // ============================================================
  val idle :: sCounter :: complete :: Nil = Enum(3)
  val state                               = RegInit(idle)

  // ============================================================
  // Registers
  // ============================================================
  val rob_id_reg     = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val is_sub_reg     = RegInit(false.B)
  val sub_rob_id_reg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))

  // Counter-specific registers
  val subcmd_reg  = RegInit(0.U(4.W))
  val ctr_id_reg  = RegInit(0.U(4.W))
  val payload_reg = RegInit(0.U(56.W))

  // Cycle counters: 16 independent counters
  val cycleCounter = RegInit(0.U(64.W))
  cycleCounter := cycleCounter + 1.U

  val ctrStartCycle = RegInit(VecInit(Seq.fill(NUM_COUNTERS)(0.U(64.W))))
  val ctrTag        = RegInit(VecInit(Seq.fill(NUM_COUNTERS)(0.U(56.W))))
  val ctrActive     = RegInit(VecInit(Seq.fill(NUM_COUNTERS)(false.B)))

  // ============================================================
  // Private SramBank (staging buffer for future use)
  // ============================================================
  val privBank = Module(new SramBank(b))

  // Default: private bank idle
  privBank.io.sramRead.req.valid       := false.B
  privBank.io.sramRead.req.bits.addr   := 0.U
  privBank.io.sramRead.resp.ready      := false.B
  privBank.io.sramWrite.req.valid      := false.B
  privBank.io.sramWrite.req.bits.addr  := 0.U
  privBank.io.sramWrite.req.bits.data  := 0.U
  privBank.io.sramWrite.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
  privBank.io.sramWrite.req.bits.wmode := false.B
  privBank.io.sramWrite.resp.ready     := true.B

  // ============================================================
  // DPI-C modules
  // ============================================================
  val ctraceDpi = Module(new CTraceDPI)
  ctraceDpi.io.subcmd  := 0.U
  ctraceDpi.io.ctr_id  := 0.U
  ctraceDpi.io.tag     := 0.U
  ctraceDpi.io.elapsed := 0.U
  ctraceDpi.io.cycle   := cycleCounter
  ctraceDpi.io.enable  := false.B

  // ============================================================
  // External bank port defaults
  // ============================================================
  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id           := rob_id_reg
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).bank_id          := 0.U
    io.bankRead(i).group_id         := 0.U
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
  }

  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id            := rob_id_reg
    io.bankWrite(i).ball_id           := 0.U
    io.bankWrite(i).bank_id           := 0.U
    io.bankWrite(i).group_id          := 0.U
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := false.B
  }

  // ============================================================
  // Command interface
  // ============================================================
  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := false.B
  io.cmdResp.bits.rob_id     := rob_id_reg
  io.cmdResp.bits.is_sub     := is_sub_reg
  io.cmdResp.bits.sub_rob_id := sub_rob_id_reg

  // ============================================================
  // State machine
  // ============================================================
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        rob_id_reg     := io.cmdReq.bits.rob_id
        is_sub_reg     := io.cmdReq.bits.is_sub
        sub_rob_id_reg := io.cmdReq.bits.sub_rob_id

        val cmd = io.cmdReq.bits.cmd
        val rs2 = cmd.rs2

        // Counter fields from rs2
        subcmd_reg  := rs2(3, 0)
        ctr_id_reg  := rs2(7, 4)
        payload_reg := rs2(63, 8)

        state := sCounter
      }
    }

    // ----------------------------------------------------------
    // Counter instruction: execute and complete in 1 cycle
    // ----------------------------------------------------------
    is(sCounter) {
      val cid = ctr_id_reg

      ctraceDpi.io.subcmd := subcmd_reg
      ctraceDpi.io.ctr_id := cid

      switch(subcmd_reg) {
        is(CTR_START) {
          ctrStartCycle(cid) := cycleCounter
          ctrTag(cid)        := payload_reg
          ctrActive(cid)     := true.B

          ctraceDpi.io.tag     := payload_reg
          ctraceDpi.io.elapsed := 0.U
          ctraceDpi.io.enable  := true.B
        }
        is(CTR_STOP) {
          val elapsed = cycleCounter - ctrStartCycle(cid)
          ctrActive(cid) := false.B

          ctraceDpi.io.tag     := ctrTag(cid)
          ctraceDpi.io.elapsed := elapsed
          ctraceDpi.io.enable  := true.B
        }
        is(CTR_READ) {
          val current = cycleCounter - ctrStartCycle(cid)

          ctraceDpi.io.tag     := ctrTag(cid)
          ctraceDpi.io.elapsed := current
          ctraceDpi.io.enable  := true.B
        }
      }

      state := complete
    }

    // ----------------------------------------------------------
    // Complete: fire cmdResp
    // ----------------------------------------------------------
    is(complete) {
      io.cmdResp.valid       := true.B
      io.cmdResp.bits.rob_id := rob_id_reg
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }

  // ============================================================
  // Status
  // ============================================================
  io.status.idle    := state === idle
  io.status.running := state =/= idle && state =/= complete
}
