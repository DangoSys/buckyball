package framework.frontend.globalrs

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig
import framework.balldomain.blink.SubRobRow
import framework.frontend.decoder.PostGDCmd

@instantiable
class SubROB(val b: GlobalConfig) extends Module {
  val subRobDepth = b.frontend.sub_rob_depth
  val subIdBits   = log2Up(subRobDepth * 4)
  val robIdBits   = log2Up(b.frontend.rob_entries)
  val ballIdBits  = log2Up(b.ballDomain.ballNum)
  val rowIdBits   = log2Up(subRobDepth)

  @public
  val io = IO(new Bundle {
    // Ball → SubROB: write a row of sub-instructions
    val write            = Flipped(Decoupled(new SubRobRow(b)))
    // SubROB → GlobalRS: issue one slot at a time (PostGDCmd, domain_id determines target)
    val issue            = Decoupled(new PostGDCmd(b))
    val issueSubId       = Output(UInt(subIdBits.W))
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
  val sram = SyncReadMem(subRobDepth, new SubRobRow(b))

  val writePtr     = RegInit(0.U(rowIdBits.W))
  val readPtr      = RegInit(0.U(rowIdBits.W))
  val rowCount     = RegInit(0.U((log2Up(subRobDepth) + 1).W))
  val lockedBallId = RegInit(0.U(ballIdBits.W))
  val masterRobId  = RegInit(0.U(robIdBits.W))

  def nextPtr(p: UInt): UInt = Mux(p === (subRobDepth - 1).U, 0.U, p + 1.U)

  val occupied  = rowCount > 0.U
  val isFull    = rowCount === subRobDepth.U
  val ballMatch = !occupied || (io.write.bits.ball_id === lockedBallId)

  // Write path: accept if not full AND ball_id matches (or not occupied yet)
  io.write.ready := !isFull && ballMatch

  when(io.write.fire) {
    sram.write(writePtr, io.write.bits)
    writePtr := nextPtr(writePtr)
    rowCount := rowCount + 1.U
    when(!occupied) {
      lockedBallId := io.write.bits.ball_id
      masterRobId  := io.write.bits.master_rob_id
    }
  }

  // -------------------------------------------------------------------------
  // FSM
  // -------------------------------------------------------------------------
  // States:
  //   sIdle      - nothing to do (rowCount == 0)
  //   sReadReq   - issue SRAM read request for readPtr row
  //   sReadWait  - wait 1 cycle for sramRaw to be latched into sramData
  //   sReadResp  - sramData valid; dispatch valid slots one per cycle
  //   sWaitSlots - all slots issued; wait for all subCompletes
  //   sWaitMaster - all rows done; fire masterComplete then return to sIdle
  val sIdle :: sReadReq :: sReadWait :: sReadResp :: sWaitSlots :: sWaitMaster :: Nil = Enum(6)
  val state                                                                           = RegInit(sIdle)

  // SyncReadMem: read issued in sReadReq (cycle N), data valid on cycle N+1.
  // We latch it into sramData with RegEnable so it stays stable during sReadResp
  // and sWaitSlots, even when the read port is not enabled.
  val sramReadEn = state === sReadReq
  val sramRaw    = sram.read(readPtr, sramReadEn)
  val dataFresh  = RegNext(sramReadEn, false.B) // pulse on first cycle of sReadResp
  val sramData   = RegEnable(sramRaw, dataFresh)
  val readPtrReg = RegEnable(readPtr, sramReadEn)

  // Per-row tracking registers (reset when moving to next row)
  // slotIssued: have we already sent this slot to a domain?
  // slotDone:   have we received subComplete for this slot?
  val slotIssued = RegInit(VecInit(Seq.fill(4)(false.B)))
  val slotDone   = RegInit(VecInit(Seq.fill(4)(false.B)))

  // Combinational: which slots still need to be issued?
  val slotNeedsIssue = VecInit((0 until 4).map(i => sramData.slots(i).valid && !slotIssued(i)))
  val hasSlotToIssue = slotNeedsIssue.asUInt.orR
  val firstSlotIdx   = PriorityEncoder(slotNeedsIssue.asUInt)

  // Combinational: are all valid slots done?
  val allSlotsDone = VecInit((0 until 4).map(i => !sramData.slots(i).valid || slotDone(i))).asUInt.andR

  // Issue output (valid only in sReadResp when there's a slot to issue)
  io.issue.valid      := (state === sReadResp) && hasSlotToIssue
  io.issue.bits       := sramData.slots(firstSlotIdx).cmd
  io.issueSubId       := readPtrReg * 4.U + firstSlotIdx
  io.issueMasterRobId := masterRobId

  // subComplete: always ready, mark the corresponding slot done
  io.subComplete.ready := true.B
  when(io.subComplete.fire) {
    val subId   = io.subComplete.bits
    val slotIdx = subId(1, 0)
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
  def advanceRow(): Unit = {
    readPtr              := nextPtr(readPtr)
    rowCount             := rowCount - 1.U
    slotIssued.foreach(_ := false.B)
    slotDone.foreach(_   := false.B)
  }

  switch(state) {
    is(sIdle) {
      when(occupied)(state := sReadReq)
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
      when(io.issue.fire) {
        slotIssued(firstSlotIdx) := true.B
      }
      // Check if all slots have been issued
      // After marking current slot issued, recalculate
      val allIssued = VecInit((0 until 4).map(i =>
        !sramData.slots(i).valid || slotIssued(i) ||
          (io.issue.fire && firstSlotIdx === i.U)
      )).asUInt.andR

      when(allIssued) {
        // Check fast path: all already done (e.g., empty row or immediate completion)
        val allDoneNow = VecInit((0 until 4).map(i => !sramData.slots(i).valid || slotDone(i))).asUInt.andR
        when(allDoneNow) {
          // Fast path: skip sWaitSlots
          advanceRow()
          state := Mux(rowCount === 1.U, sWaitMaster, sReadReq)
        }.otherwise {
          state := sWaitSlots
        }
      }
      // else: stay in sReadResp to dispatch remaining slots
    }
    is(sWaitSlots) {
      when(allSlotsDone) {
        advanceRow()
        state := Mux(rowCount === 1.U, sWaitMaster, sReadReq)
      }
    }
    is(sWaitMaster) {
      when(io.masterComplete.fire) {
        lockedBallId := 0.U
        masterRobId  := 0.U
        state        := sIdle
      }
    }
  }
}
