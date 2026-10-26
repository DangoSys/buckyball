package examples.balls.trace

import framework.acc.balldomain.configs.BallRouteParam

import chisel3.experimental.hierarchy.{Instance, Instantiate}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import framework.acc.balldomain.ball.blink.{BallCommand, BallResponse}
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.balldomain.ball.blink.BallStatus
import framework.acc.memdomain.{BankRead, BankWrite}

/**
 * Trace — TraceBall inner processing unit.
 *
 * Handles cycle counter management (START/STOP/READ).
 */
@instantiable
class Trace(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val routingParam:  BallDomainParam,
  val instance:      BallRouteParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits

  val ballMapping = instance

  val inBW  = ballMapping.inBW
  val outBW = ballMapping.outBW

  val bankWidth = memParam.bankWidth

  @public
  val io = IO(new Bundle {

    val cmdReq = Flipped(Decoupled(new BallCommand(frontendParam, memParam)))

    val cmdResp = Decoupled(new BallResponse(frontendParam))

    val bankRead = Vec(
      inBW,
      Flipped(new BankRead(
        frontendParam,
        memParam
      ))
    )

    val bankWrite = Vec(
      outBW,
      Flipped(new BankWrite(
        frontendParam,
        memParam
      ))
    )

    val status = new BallStatus
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
  val rob_id_reg = RegInit(0.U(log2Up(frontendParam.rob_entries).W))

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
  // DPI-C modules
  // ============================================================
  val ctraceDpi = Instantiate(new CTraceDPI)
  ctraceDpi.io.clock   := clock
  ctraceDpi.io.reset   := reset.asBool
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
    io.bankRead(i).bank_id          := 0.U
    io.bankRead(i).group_id         := 0.U
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
  }

  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id           := rob_id_reg
    io.bankWrite(i).bank_id          := 0.U
    io.bankWrite(i).group_id         := 0.U
    io.bankWrite(i).io.req.valid     := false.B
    io.bankWrite(i).io.req.bits.addr := 0.U
    io.bankWrite(i).io.req.bits.data := 0.U
    io.bankWrite(i).io.req.bits.mask := VecInit(
      Seq.fill(memParam.bankMaskLen)(0.U(1.W))
    )
    io.bankWrite(i).io.resp.ready    := false.B
  }

  // ============================================================
  // Command interface
  // ============================================================
  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := rob_id_reg

  // ============================================================
  // State machine
  // ============================================================
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        rob_id_reg := io.cmdReq.bits.rob_id

        val cmd = io.cmdReq.bits
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
