package framework.acc.balldomain.sub_rob

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.frontend.decoder.{DomainId, PostGDCmd}
import framework.acc.balldomain.sub_rob.configs.Params

@instantiable
class SubROB(val frontendParam: FrontendParam, val memParam: MemDomainParam, val ballParam: BallDomainParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits

  val p = Params(
    depth = frontendParam.sub_rob_depth,
    masterEntries = frontendParam.rob_entries,
    balls = ballParam.balls.size
  )

  val subRobDepth = p.depth
  val subIdBits   = log2Up(subRobDepth * 4)
  val robIdBits   = log2Up(p.masterEntries)
  val ballIdBits  = log2Up(p.balls)
  val rowIdBits   = log2Up(subRobDepth)

  @public
  val io = IO(new Bundle {
    // Ball → SubROB: write a row of sub-instructions
    val write            = Flipped(Decoupled(new SubRobRow(frontendParam, memParam, ballParam)))
    // SubROB → GlobalRS: issue independent Ball and memory slots
    val issue            = Vec(2, Decoupled(new PostGDCmd(frontendParam, memParam)))
    val issueSubId       = Output(Vec(2, UInt(subIdBits.W)))
    val issueMasterRobId = Output(UInt(robIdBits.W))
    // GlobalRS → SubROB: a sub-instruction completed (sub_rob_id)
    val subComplete      = Flipped(Decoupled(UInt(subIdBits.W)))
    // SubROB → main ROB: all sub-instructions done
    val masterComplete   = Decoupled(UInt(robIdBits.W))
    // Status
    val occupied         = Output(Bool())
    val lockedBallId     = Output(UInt(ballIdBits.W))
  })

  // -------------------------------------------------------------------------
  // Storage
  // -------------------------------------------------------------------------
  val sram = SyncReadMem(subRobDepth, new SubRobRow(frontendParam, memParam, ballParam))

  val writePtr     = RegInit(0.U(rowIdBits.W))
  val readPtr      = RegInit(0.U(rowIdBits.W))
  val rowCount     = RegInit(0.U((log2Up(subRobDepth) + 1).W))
  val lockedBallId = RegInit(0.U(ballIdBits.W))
  val masterRobId  = RegInit(0.U(robIdBits.W))

  def nextPtr(p: UInt): UInt = Mux(p === (subRobDepth - 1).U, 0.U, p + 1.U)

  val occupied   = RegInit(false.B)
  val lastQueued = RegInit(false.B)
  val isFull     = rowCount === subRobDepth.U
  val ballMatch  =
    !occupied || (io.write.bits.ball_id === lockedBallId && io.write.bits.master_rob_id === masterRobId)

  // -------------------------------------------------------------------------
  // FSM
  // -------------------------------------------------------------------------
  // States:
  //   sIdle      - nothing to do (rowCount == 0)
  //   sReadReq   - issue SRAM read request for readPtr row
  //   sReadWait  - wait 1 cycle for sramRaw to be latched into sramData
  //   sReadResp  - sramData valid; dispatch one slot per domain per cycle
  //   sWaitSlots - all slots issued; wait for all subCompletes
  //   sWaitMaster - all rows done; fire masterComplete then return to sIdle
  val sIdle :: sReadReq :: sReadWait :: sReadResp :: sWaitSlots :: sWaitMaster :: Nil =
    Enum(6)
  val state                                                                           = RegInit(sIdle)

  // Write path: accept if not full AND ball_id matches (or not occupied yet)
  io.write.ready := !isFull && !lastQueued && ballMatch && state =/= sReadReq && state =/= sWaitMaster

  when(io.write.fire) {
    occupied                            := true.B
    when(io.write.bits.last)(lastQueued := true.B)
    for (slot <- io.write.bits.slots) {
      when(slot.valid) {
        assert(
          slot.cmd.domain_id === DomainId.BALL || slot.cmd.domain_id === DomainId.MEM,
          "SubROB supports Ball and memory slots"
        )
      }
    }
    when(occupied) {
      assert(
        io.write.bits.master_rob_id === masterRobId,
        "SubROB row master_rob_id mismatch"
      )
    }
    writePtr := nextPtr(writePtr)
    when(!occupied) {
      lockedBallId := io.write.bits.ball_id
      masterRobId  := io.write.bits.master_rob_id
    }
  }

  // SyncReadMem: read issued in sReadReq (cycle N), data valid on cycle N+1.
  // We latch it into sramData with RegEnable so it stays stable during sReadResp
  // and sWaitSlots, even when the read port is not enabled.
  val sramReadEn = state === sReadReq

  val sramRaw = sram.readWrite(
    Mux(io.write.fire, writePtr, readPtr),
    io.write.bits,
    sramReadEn || io.write.fire,
    io.write.fire
  )

  val dataFresh =
    RegNext(sramReadEn, false.B) // pulse on first cycle of sReadResp
  val sramData   = RegEnable(sramRaw, dataFresh)
  val readPtrReg = RegEnable(readPtr, sramReadEn)

  // Per-row tracking registers (reset when moving to next row)
  // slotIssued: have we already sent this slot to a domain?
  // slotDone:   have we received subComplete for this slot?
  val slotIssued = RegInit(VecInit(Seq.fill(4)(false.B)))
  val slotDone   = RegInit(VecInit(Seq.fill(4)(false.B)))

  // Combinational: which slots still need to be issued?
  val slotNeedsIssue = VecInit(
    (0 until 4).map(i => sramData.slots(i).valid && !slotIssued(i))
  )

  val selected = Wire(Vec(2, UInt(2.W)))
  for (port <- 0 until 2) {
    val domain     = if (port == 0) DomainId.BALL else DomainId.MEM
    val candidates =
      VecInit(
        (0 until 4).map(slot => slotNeedsIssue(slot) && sramData.slots(slot).cmd.domain_id === domain)
      )
    selected(port)       := PriorityEncoder(candidates.asUInt)
    io.issue(port).valid := state === sReadResp && candidates.asUInt.orR
    io.issue(port).bits  := sramData.slots(selected(port)).cmd
    io.issueSubId(port)  := readPtrReg * 4.U + selected(port)
  }

  // Combinational: are all valid slots done?
  val allSlotsDone = VecInit(
    (0 until 4).map(i => !sramData.slots(i).valid || slotDone(i))
  ).asUInt.andR

  io.issueMasterRobId := masterRobId

  // subComplete: always ready, mark the corresponding slot done
  io.subComplete.ready := true.B
  when(io.subComplete.fire) {
    val subId   = io.subComplete.bits
    val subRow  = subId / 4.U
    val slotIdx = subId(1, 0)
    assert(
      state === sReadResp || state === sWaitSlots || state === sWaitMaster,
      "SubROB subComplete arrived in invalid state"
    )
    assert(subRow === readPtrReg, "SubROB subComplete points to non-active row")
    slotDone(slotIdx) := true.B
  }

  // masterComplete
  io.masterComplete.valid := state === sWaitMaster
  io.masterComplete.bits  := masterRobId

  // Status outputs
  io.occupied     := occupied
  io.lockedBallId := lockedBallId

  // -------------------------------------------------------------------------
  // FSM transitions
  // -------------------------------------------------------------------------
  // Helper: advance past current row (called when row is complete)
  val rowAdvance = WireDefault(false.B)

  def advanceRow(): Unit = {
    readPtr              := nextPtr(readPtr)
    rowAdvance           := true.B
    slotIssued.foreach(_ := false.B)
    slotDone.foreach(_   := false.B)
  }

  switch(state) {
    is(sIdle) {
      when(rowCount =/= 0.U)(state := sReadReq)
    }
    is(sReadReq) {
      // SRAM read issued, data available on sramRaw next cycle
      state := sReadWait
    }
    is(sReadWait) {
      // sramRaw is valid; RegEnable latches it into sramData at this clock edge.
      // sramData will be stable starting next cycle (sReadResp).
      state := sReadResp
    }
    is(sReadResp) {
      // Fire issued slot
      for (port <- 0 until 2) {
        when(io.issue(port).fire)(slotIssued(selected(port)) := true.B)
      }
      // Check if all slots have been issued
      // After marking current slot issued, recalculate
      val allIssued = VecInit(
        (0 until 4).map(i =>
          !sramData.slots(i).valid || slotIssued(i) ||
            VecInit(
              (0 until 2).map(port => io.issue(port).fire && selected(port) === i.U)
            ).asUInt.orR
        )
      ).asUInt.andR

      when(allIssued) {
        // Check fast path: all already done (e.g., empty row or immediate completion)
        val allDoneNow = VecInit(
          (0 until 4).map(i => !sramData.slots(i).valid || slotDone(i))
        ).asUInt.andR
        when(allDoneNow) {
          // Fast path: skip sWaitSlots
          advanceRow()
          state := Mux(
            sramData.last,
            sWaitMaster,
            Mux(rowCount === 1.U && !io.write.fire, sIdle, sReadReq)
          )
        }.otherwise {
          state := sWaitSlots
        }
      }
      // else: stay in sReadResp to dispatch remaining slots
    }
    is(sWaitSlots) {
      when(allSlotsDone) {
        advanceRow()
        state := Mux(
          sramData.last,
          sWaitMaster,
          Mux(rowCount === 1.U && !io.write.fire, sIdle, sReadReq)
        )
      }
    }
    is(sWaitMaster) {
      when(io.masterComplete.fire) {
        occupied     := false.B
        lastQueued   := false.B
        lockedBallId := 0.U
        masterRobId  := 0.U
        state        := sIdle
      }
    }
  }
  when(io.write.fire =/= rowAdvance) {
    rowCount := Mux(io.write.fire, rowCount + 1.U, rowCount - 1.U)
  }

}

class SubRobSlot(val frontendParam: FrontendParam, val memParam: MemDomainParam) extends Bundle {
  val commandDataBits = frontendParam.commandDataBits
  val valid           = Bool()
  val cmd             = new PostGDCmd(frontendParam, memParam)
}

class SubRobRow(val frontendParam: FrontendParam, val memParam: MemDomainParam, val ballParam: BallDomainParam)
    extends Bundle {
  val commandDataBits = frontendParam.commandDataBits
  val slots           = Vec(4, new SubRobSlot(frontendParam, memParam))
  val ball_id         = UInt(log2Up(ballParam.balls.size).W)
  val master_rob_id   = UInt(log2Up(frontendParam.rob_entries).W)
  val last            = Bool()
}
