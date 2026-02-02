// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2022 Jiuyang Liu <liu@jiuyang.me>
// Port to buckyball framework

package framework.gpdomain.sequencer

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.util._
import chisel3.util.experimental.decode.DecodeBundle
import framework.top.GlobalConfig
import framework.frontend.globalrs.{GlobalRsComplete, GlobalRsIssue}
import framework.gpdomain.sequencer.decoder.{Decoder, DomainDecoder}

/**
 * Sequencer for GP Domain
 *
 * Handles instruction issue, scheduling, token management, and coordination
 * between lanes, LSU, and mask unit.
 *
 * Ported from T1's sequencer logic
 */
@instantiable
class Sequencer(val b: GlobalConfig) extends Module {
  val chainingSize = b.gpDomain.chainingSize
  val laneNumber   = b.gpDomain.laneNumber
  val vLen         = b.gpDomain.vLen
  val dLen         = b.gpDomain.dLen
  val eLen         = b.gpDomain.eLen
  val xLen         = 32 // Fixed to 32 for now

  val instructionIndexBits = log2Ceil(chainingSize) + 1
  val chaining1HBits       = 2 << log2Ceil(chainingSize)
  val datapathWidth        = b.gpDomain.laneScale * eLen
  val vlMaxBits            = log2Ceil(vLen * 8 / 8) + 1 // vLen in bits / sewMin
  val regNumBits           = 5                          // 32 vector registers
  val vrfOffsetBits        = log2Ceil(vLen / datapathWidth / laneNumber)

  @public
  val io = IO(new Bundle {
    // Interface with global RS
    val global_issue_i    = Flipped(Decoupled(new GlobalRsIssue(b)))
    val global_complete_o = Decoupled(new GlobalRsComplete(b))

    // TODO: Interface with lanes (placeholder)
    // val laneRequest = Vec(laneNumber, Decoupled(new LaneRequest(...)))
    // val laneResponse = Vec(laneNumber, Flipped(Decoupled(new LaneResponse(...))))

    // TODO: Interface with LSU (placeholder)
    // val lsuRequest = Decoupled(new LSURequestInterface(...))
    // val lsuReport = Flipped(Decoupled(new LastReportBundle(...)))

    // TODO: Interface with mask unit (placeholder)
    // val maskUnitRequest = Decoupled(new MaskUnitExeReq(...))
    // val maskUnitReport = Flipped(Decoupled(new LastReportBundle(...)))

    // Status
    val busy = Output(Bool())
  })

  // ===========================================================================
  // Decoder
  // ===========================================================================
  val decoder: Instance[DomainDecoder] = Instantiate(new DomainDecoder(b))
  decoder.io.inst_i := io.global_issue_i.bits.cmd.cmd
  val decoded = decoder.io.decoded_o

  // ===========================================================================
  // Token Manager
  // ===========================================================================
  val tokenManager: Instance[TokenManager] = Instantiate(new TokenManager(b))

  // ===========================================================================
  // Instruction Counter
  // ===========================================================================
  val instructionCounter:     UInt = RegInit(0.U(instructionIndexBits.W))
  val nextInstructionCounter: UInt = instructionCounter + 1.U

  // ===========================================================================
  // Request Register (1-deep queue)
  // ===========================================================================
  val requestReg: ValidIO[InstructionPipeBundle] = RegInit(
    0.U.asTypeOf(Valid(new InstructionPipeBundle(xLen, vLen, instructionIndexBits, vlMaxBits)))
  )

  val requestRegDequeue = Wire(Decoupled(io.global_issue_i.bits.cloneType))

  // Latch instruction when issued
  when(io.global_issue_i.fire) {
    requestReg.bits.instruction      := io.global_issue_i.bits.cmd.cmd.raw_inst
    requestReg.bits.rs1Data          := io.global_issue_i.bits.cmd.cmd.rs1Data
    requestReg.bits.rs2Data          := io.global_issue_i.bits.cmd.cmd.rs2Data
    requestReg.bits.vl               := 0.U // TODO: extract from CSR
    requestReg.bits.vstart           := 0.U // TODO: extract from CSR
    requestReg.bits.vtype            := 0.U // TODO: extract from CSR
    requestReg.bits.decodeResult     := decoded
    requestReg.bits.instructionIndex := instructionCounter
    requestReg.bits.vdIsV0           := (io.global_issue_i.bits.cmd.cmd.raw_inst(11, 7) === 0.U) &&
    (io.global_issue_i.bits.cmd.cmd.raw_inst(6) || !io.global_issue_i.bits.cmd.cmd.raw_inst(5))
    requestReg.bits.writeByte        := 0.U // TODO: calculate based on vl and sew

    instructionCounter := nextInstructionCounter
  }

  // Request register valid update
  requestReg.valid := Mux(
    io.global_issue_i.fire ^ requestRegDequeue.fire,
    io.global_issue_i.fire,
    requestReg.valid
  )

  // Manually maintain dequeue interface
  requestRegDequeue.bits  := io.global_issue_i.bits
  requestRegDequeue.valid := requestReg.valid

  // Decode result alias
  val decodeResult: DecodeBundle = requestReg.bits.decodeResult

  // Instruction type detection
  val isLoadStoreType: Bool = !requestRegDequeue.bits.cmd.cmd.raw_inst(6) && requestRegDequeue.valid
  val isStoreType:     Bool = !requestRegDequeue.bits.cmd.cmd.raw_inst(6) && requestRegDequeue.bits.cmd.cmd.raw_inst(5)
  val maskType:        Bool = !requestRegDequeue.bits.cmd.cmd.raw_inst(25)

  // ===========================================================================
  // Instruction Slots (State Machine Registers)
  // ===========================================================================
  val instructionFinished: Vec[Vec[Bool]] = Wire(Vec(laneNumber, Vec(chainingSize, Bool())))
  val vxsatReportVec:      Vec[UInt]      = Wire(Vec(laneNumber, UInt(chainingSize.W)))
  val vxsatReport = vxsatReportVec.reduce(_ | _)

  // Initialize dummy values (will be connected to lanes later)
  instructionFinished.foreach(_.foreach(_ := false.B))
  vxsatReportVec.foreach(_ := 0.U)

  val slots: Seq[InstructionControl] = Seq.tabulate(chainingSize) { index =>
    val control = RegInit(
      (-1.S(new InstructionControl(instructionIndexBits, laneNumber).getWidth.W))
        .asTypeOf(new InstructionControl(instructionIndexBits, laneNumber))
    )

    // Execution finished check
    val laneAndLSUFinish: Bool = control.endTag.asUInt.andR
    val v0WriteFinish = !ohCheck(tokenManager.v0WriteValid, control.record.instructionIndex, chainingSize)

    // LSU finished (placeholder)
    val lsuFinished: Bool = false.B // TODO: connect to LSU
    val vxsatUpdate = ohCheck(vxsatReport, control.record.instructionIndex, chainingSize)

    // Instruction allocation to this slot
    val instructionToSlotOH: UInt = Wire(UInt(chainingSize.W))
    when(instructionToSlotOH(index)) {
      // Instruction metadata
      control.record.instructionIndex := requestReg.bits.instructionIndex
      control.record.isLoadStore      := isLoadStoreType
      control.record.maskType         := maskType
      control.record.gather           := false.B // TODO: decode
      control.record.pop              := false.B // TODO: decode

      // Control signals
      control.state.idle          := false.B
      control.state.wLast         := false.B
      control.state.sCommit       := false.B
      control.state.wMaskUnitLast := true.B // TODO: check maskUnit requirement

      control.vxsat := false.B

      // Initialize endTag
      control.endTag := VecInit(Seq.fill(laneNumber)(false.B) :+ !isLoadStoreType)
    }.otherwise {
      // State machine updates
      when(laneAndLSUFinish && v0WriteFinish) {
        control.state.wLast := true.B
      }

      // TODO: Add retire logic
      // when(responseCounter === control.record.instructionIndex && retire) {
      //   control.state.sCommit := true.B
      // }

      when(control.state.sCommit && control.state.wMaskUnitLast) {
        control.state.idle := true.B
      }

      // Update endTag from lanes
      control.endTag.zip(instructionFinished.map(_(index)) :+ lsuFinished).foreach {
        case (d, c) =>
          d := d || c
      }

      when(vxsatUpdate) {
        control.vxsat := true.B
      }
    }

    control
  }

  // ===========================================================================
  // Slot Allocation Logic
  // ===========================================================================
  val slotFree:    Vec[Bool] = VecInit(slots.map(_.state.idle))
  val allSlotFree: Bool      = slotFree.asUInt.andR
  val freeOR:      Bool      = slotFree.asUInt.orR

  // Special instructions go to last slot
  val specialInstruction: Bool = false.B // TODO: decode special instructions
  val slotReady:          Bool = Mux(specialInstruction, slots.last.state.idle, freeOR)

  // Select slot for new instruction
  val instructionToSlotOH: UInt = Mux(
    specialInstruction,
    UIntToOH(chainingSize.U),
    PriorityEncoderOH(slotFree.asUInt)
  )

  // ===========================================================================
  // Token Manager Connections
  // ===========================================================================
  tokenManager.instructionIssue.valid                 := requestRegDequeue.valid
  tokenManager.instructionIssue.bits.instructionIndex := requestReg.bits.instructionIndex
  tokenManager.instructionIssue.bits.writeV0          := requestReg.bits.vdIsV0
  tokenManager.instructionIssue.bits.useV0AsMask      := maskType
  tokenManager.instructionIssue.bits.isLoadStore      := isLoadStoreType
  tokenManager.instructionIssue.bits.toLane           := !isLoadStoreType
  tokenManager.instructionIssue.bits.toMask           := false.B // TODO: detect mask unit instructions

  // LSU write v0 connections (placeholder)
  tokenManager.lsuWriteV0.foreach { port =>
    port.valid := false.B
    port.bits  := 0.U
  }

  // Instruction finish signals (placeholder)
  tokenManager.instructionFinish.foreach(_ := 0.U)
  tokenManager.maskUnitFree                := true.B

  // ===========================================================================
  // Issue Logic
  // ===========================================================================
  // Can issue when:
  // 1. There's a free slot
  // 2. Token manager allows (no v0 conflicts)
  // 3. All lanes are ready (TODO)
  val canIssue: Bool = slotReady && tokenManager.issueAllow

  io.global_issue_i.ready := canIssue && io.global_complete_o.ready
  requestRegDequeue.ready := canIssue

  // ===========================================================================
  // Completion Logic (Placeholder)
  // ===========================================================================
  io.global_complete_o.valid       := io.global_issue_i.valid
  io.global_complete_o.bits.rob_id := io.global_issue_i.bits.rob_id

  // ===========================================================================
  // Status
  // ===========================================================================
  io.busy := !allSlotFree
}
